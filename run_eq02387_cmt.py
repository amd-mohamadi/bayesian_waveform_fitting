"""Waveform-fitting MT inversion for the small CAPE event eq02387 (Mw ~2).

Adapted from run_regional_cmt.py for a much smaller, local event. Differences:
  * Green's functions: the qseis store built from the CAPE path-averaged 1-D model
    cape_pathavg_eq02387_600m.tvel (store id forge_eq02387_qseis_pathavg_600m).
  * Data come from cape_events/eq02387/invdata.pkl (ENZ velocity traces), rotated
    to ZRT (vertical/radial/transverse) before fitting.
  * CAP-style windowing: the OBSERVED phase window is cut around the invdata pick
    (P pick -> Z, S pick -> R & T), while the SYNTHETIC window is cut around the
    modeled (store) arrival -- where the synthetic energy actually is. A per-trace
    cross-correlation shift then aligns them. Cutting the synthetic on the pick
    instead would land its window off the synthetic arrival (flat traces).
  * Observed and synthetic are both velocity.
  * Optional first-motion (Pz) polarity term: observed polarity comes from the
    picker CSV (signed phase_polarity, sign=up/down, |.|=confidence); the synthetic
    coefficient is read off the RAW GF basis at the modeled onset and scored with a
    probit. Evaluated at the fixed onset, it is immune to the autoshift, so it stops
    the shift from "cheating" by aligning onto an opposite-polarity cycle.
    Controlled by --polarity-weight (0 disables) / --polarity-sigma.

The invdata FORK S pick is a mispick (~6.4 s); it is overridden to 1.25 s after the
origin -- the actual S arrival read off report/eq02387_picks_check.png.

Run from the repo root:
    conda run -n pymc python run_eq02387_cmt.py --mode real
    conda run -n pymc python run_eq02387_cmt.py --mode synthetic   # recovery test
"""
import argparse
import os

import numpy as np
from pyrocko import moment_tensor as pmt

from src import invdata as invio
from src.forward import GFForward
from src import dataset as ds
from src import model as M
from src.sampler import run_smc

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "store_superdirs": [os.path.join(HERE, "gf_stores")],
    "store_id": "forge_eq02387_qseis_pathavg_600m",
    "invdata": os.path.join(HERE, "cape_events", "eq02387", "invdata.pkl"),
    "picks_file": os.path.join(HERE, "picks_amp.csv"),
    "polarity_cache": os.path.join(HERE, "cape_events", "eq02387", "pz_polarity.csv"),
    "smti_reference_mt6": os.path.join(
        HERE, "reference_solutions", "eq02387_smti_blackjax_ppol_ampratio_mt6.npz"),
}
# grond DC reference solution (grond_eq02387_dc_reference_run.md) for comparison
GROND_DC_REF = dict(strike=77.76, dip=87.65, rake=-24.06, mw=1.915)
PHASE_OF = {"Z": "P", "R": "S", "T": "S"}
FORK = ("UU", "FORK", "01")


def m6_to_mt(m6):
    return pmt.MomentTensor(mnn=m6[0], mee=m6[1], mdd=m6[2], mne=m6[3], mnd=m6[4], med=m6[5])


def m6_from_sdr_mw(strike, dip, rake, mw):
    m = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mw).m()  # NED 3x3
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])


def report(m6_ref, ref_label, posterior, weights):
    m6 = posterior["m6"]
    mean_m6 = np.average(m6, axis=0, weights=weights)
    mt_est = m6_to_mt(mean_m6)
    s1, d1, r1 = mt_est.both_strike_dip_rake()[0]
    mw_est = float(np.average(posterior["mw"], weights=weights))
    mw_std = float(np.sqrt(np.average((posterior["mw"] - mw_est) ** 2, weights=weights)))
    kagan = None
    print("\n================ POSTERIOR ================")
    print(f"estimated  strike={s1:6.1f}  dip={d1:5.1f}  rake={r1:6.1f}  Mw={mw_est:.2f} +/- {mw_std:.2f}")
    if m6_ref is not None:
        mt_ref = m6_to_mt(m6_ref)
        s1t, d1t, r1t = mt_ref.both_strike_dip_rake()[0]
        kagan = pmt.kagan_angle(mt_ref, mt_est)
        print(f"{ref_label:10s} strike={s1t:6.1f}  dip={d1t:5.1f}  rake={r1t:6.1f}  Mw={mt_ref.magnitude:.2f}")
        print(f"Kagan angle (mechanism error vs {ref_label}): {kagan:.1f} deg")
    print("===========================================\n")
    return mean_m6, mw_est, kagan


def build_window_specs(forward, picks_rel, args, exclude_set):
    """Per-target (aligned to forward.meta) anchors/windows/filters + a pruned observed.

    Returns (anchors, wins, filts, noise_anchors, dropped) and prints a summary.
    Targets that are excluded are removed from ``observed`` so the dataset builder
    drops them (see build_dataset_real_picks).
    """
    filt_p = dict(fmin=args.p_fmin, fmax=args.p_fmax, order=args.order, tfade=args.p_tfade)
    filt_s = dict(fmin=args.s_fmin, fmax=args.s_fmax, order=args.order, tfade=args.s_tfade)
    win_p, win_s = (args.p_pre, args.p_post), (args.s_pre, args.s_post)
    if abs((args.p_pre + args.p_post) - (args.s_pre + args.s_post)) > 1e-9:
        raise SystemExit("P and S total window length (pre+post) must be equal "
                         f"(P={args.p_pre + args.p_post}, S={args.s_pre + args.s_post})")

    obs_anchors, wins, filts, noise_anchors, dropped = [], [], [], [], []
    print(f"\n{'target':18s}{'phase':>6s}{'pick_s':>8s}{'score':>6s}  use")
    for m in forward.meta:
        net, sta, loc, cha = m["nslc"]
        stc = (net, sta, loc)
        ph = PHASE_OF[cha]
        pr = picks_rel.get(stc, {})
        p_off, s_off = pr.get("P"), pr.get("S")
        off = p_off if ph == "P" else s_off
        score = pr.get("p_score") if ph == "P" else pr.get("s_score")

        excluded = (
            off is None or p_off is None
            or (args.min_score > 0 and score is not None and score < args.min_score)
            or f"{net}.{sta}.{loc}.{cha}" in exclude_set
            or f"{net}.{sta}" in exclude_set
        )
        obs_anchors.append(off if off is not None else (p_off if p_off is not None else 0.0))
        wins.append(win_p if ph == "P" else win_s)
        filts.append(filt_p if ph == "P" else filt_s)
        noise_anchors.append(p_off if p_off is not None else 0.0)
        if excluded:
            dropped.append((net, sta, loc, cha))
        sc = f"{score:.2f}" if score is not None else "  - "
        os_ = f"{off:.2f}" if off is not None else "  - "
        print(f"{net}.{sta}.{loc}.{cha:5s}{ph:>6s}{os_:>8s}{sc:>6s}  {'no ' if excluded else 'yes'}")
    return obs_anchors, wins, filts, noise_anchors, dropped


def model_anchors(forward):
    """Per-target source-relative model traveltime (anyP for Z, anyS for N/E)."""
    from pyrocko import orthodrome
    out = []
    for m in forward.meta:
        st, cha = m["station"], m["nslc"][3]
        dist = orthodrome.distance_accurate50m(forward.event.lat, forward.event.lon,
                                               st.lat, st.lon)
        phase_id = "anyP" if cha == "Z" else "anyS"
        out.append(float(forward.store.t(phase_id, (forward.event.depth, dist))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synthetic"], default="real")
    # phase windows: observed cut at picks, synthetic at modeled arrivals.
    # P and S total length (pre+post) must match so all targets share N.
    ap.add_argument("--p-pre", type=float, default=0.125)
    ap.add_argument("--p-post", type=float, default=0.275)
    ap.add_argument("--s-pre", type=float, default=0.125)
    ap.add_argument("--s-post", type=float, default=0.275)
    ap.add_argument("--order", type=int, default=3, help="Butterworth filter order")
    ap.add_argument("--p-fmin", type=float, default=2.0)
    ap.add_argument("--p-fmax", type=float, default=30.0)
    ap.add_argument("--p-tfade", type=float, default=0.1)
    ap.add_argument("--s-fmin", type=float, default=1.0)
    ap.add_argument("--s-fmax", type=float, default=20.0)
    ap.add_argument("--s-tfade", type=float, default=0.1)

    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop picks with confidence below this (0 keeps all)")
    ap.add_argument("--exclude", default="",
                    help="comma list of NET.STA or NET.STA.LOC.CHA to drop")
    # noise / covariance
    ap.add_argument("--noise-len", type=float, default=0.8)
    ap.add_argument("--noise-gap", type=float, default=0.2)
    ap.add_argument("--structure", choices=["variance", "exponential"], default="variance")
    ap.add_argument("--group-by", choices=["global", "channel", "station", "trace"], default="station")
    # source-type / magnitude priors
    ap.add_argument("--dc", action="store_true", help="constrain to double couple")
    ap.add_argument("--gamma-beta", default="2,2")
    ap.add_argument("--delta-beta", default="2,2")
    ap.add_argument("--mw-bounds", default="1.5,3.0")
    ap.add_argument("--hp-bounds", default="-0.1,8.0")
    ap.add_argument("--max-shift-sec", type=float, default=0.05,
                    help="per-trace CC time-shift half-window in s (absorbs pick/model offset)")
    ap.add_argument("--lag-penalty", type=float, default=0.0)
    # first-motion (Pz) polarity likelihood (real mode); weight 0 disables
    ap.add_argument("--polarity-weight", type=float, default=1.0,
                    help="scale of the Pz first-motion polarity log-likelihood (0 disables)")
    ap.add_argument("--polarity-sigma", type=float, default=0.4,
                    help="probit width on the unit-normalised radiation amplitude")
    # sampler
    ap.add_argument("--num-particles", type=int, default=600)
    ap.add_argument("--mcmc-steps", type=int, default=10)
    ap.add_argument("--kernel", choices=["rmh", "hmc", "mwg"], default="mwg")
    ap.add_argument("--mwg-mechanism-steps", type=int, default=None,
                    help="inner RMH moves for the (kappa,h,sigma) block when --kernel mwg")
    ap.add_argument("--waste-free", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target-ess", type=float, default=0.85)
    ap.add_argument("--snr", type=float, default=10.0, help="synthetic-mode SNR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gf-npz", default=None,
                    help="OpenSWPC reciprocity GF store npz; replaces the qseis store")
    ap.add_argument("--components", default="Z,R,T",
                    help="comma list of components to invert, e.g. Z or Z,T")
    ap.add_argument("--sample-location", action="store_true",
                    help="sample the source location over the green-point cloud "
                         "(discrete nearest-node; requires --gf-npz)")
    ap.add_argument("--loc-radius-m", type=float, default=None,
                    help="limit the location prior box to catalog +/- this many "
                         "meters per axis (default: the whole cloud)")
    ap.add_argument("--synth-pid", type=int, default=None,
                    help="synthetic mode: generate the data from this green point "
                         "instead of the nearest one (location-recovery test)")
    ap.add_argument("--out", default="runs/eq02387_cmt_posterior.npz")
    ap.add_argument("--outdir", default="report/eq02387_cmt")
    args = ap.parse_args()

    mw_bounds = tuple(float(x) for x in args.mw_bounds.split(","))
    hp_bounds = tuple(float(x) for x in args.hp_bounds.split(","))
    gamma_beta = tuple(float(x) for x in args.gamma_beta.split(","))
    delta_beta = tuple(float(x) for x in args.delta_beta.split(","))
    exclude_set = {s.strip() for s in args.exclude.split(",") if s.strip()}
    components = tuple(s.strip().upper() for s in args.components.split(",") if s.strip())
    if not components or any(c not in ("Z", "R", "T") for c in components):
        raise SystemExit(f"--components must be a subset of Z,R,T (got {args.components!r})")

    event, stations, observed, picks_rel, raw = invio.load_invdata(PATHS["invdata"])
    # FORK invdata S pick is a mispick (~6.4 s); the true S is at 1.25 s after origin
    # (read off report/eq02387_picks_check.png).
    if FORK in picks_rel:
        picks_rel[FORK]["S"] = 1.25
    print(f"event eq02387: lat={event.lat:.3f} lon={event.lon:.3f} depth={event.depth:.0f} m  "
          f"Mw_cat~{event.magnitude}")

    # rotate observed ENZ -> ZRT; station R/T channel orientations are set to match
    # so the engine synthesises radial/transverse directly (P -> Z, S -> R & T).
    stations, observed = invio.to_zrt(event, stations, observed)

    if (args.sample_location or args.synth_pid is not None) and not args.gf_npz:
        raise SystemExit("--sample-location / --synth-pid require --gf-npz")

    if args.gf_npz:
        from src.forward import NpzGFForward
        src_d = raw["source"]
        source_xyz_km = (src_d["x"] / 1e3, src_d["y"] / 1e3, src_d["z"] / 1e3)
        forward = NpzGFForward(args.gf_npz, event, stations, source_xyz_km,
                               channels=components,
                               all_points=args.sample_location or args.synth_pid is not None)
        print(f"forward: 3D GF store {args.gf_npz} (point offset "
              f"{forward.point_offset_m:.0f} m, deltat={forward.deltat:.4f}s)")
    else:
        forward = GFForward(PATHS["store_superdirs"], PATHS["store_id"], event, stations,
                            channels=components, quantity="velocity")
        print(f"forward: {len(stations)} stations x {','.join(components)} -> "
              f"{len(forward.targets)} targets "
              f"(quantity=velocity, deltat={forward.deltat:.4f}s)")

    obs_anchors, wins, filts, noise_anchors, dropped = build_window_specs(
        forward, picks_rel, args, exclude_set)
    for nslc in dropped:
        observed.pop(nslc, None)
    if args.gf_npz:
        # CAP-style, like the 1D path: observed windows on picks, synthetic
        # windows on the synthetic's own arrivals -- detected from the basis
        # energy since the 3D store has no traveltime table. (The 3D model's
        # S runs ~100-150 ms earlier than the picks even though P matches, so
        # pick-anchored synthetic windows clip the synthetic S.)
        from src.forward import basis_onset_anchors
        basis_anchors = basis_onset_anchors(forward, obs_anchors)
        print("windows: observed on picks, synthetic on basis-detected onsets (CAP-style)")
        print(f"{'target':18s}{'phase':>6s}{'pick_s':>8s}{'onset_s':>9s}{'diff_ms':>9s}")
        for m, oa, ba in zip(forward.meta, obs_anchors, basis_anchors):
            net, sta, loc, cha = m["nslc"]
            print(f"{net}.{sta}.{loc}.{cha:5s}{PHASE_OF[cha]:>6s}{oa:8.2f}{ba:9.2f}"
                  f"{(ba - oa) * 1e3:9.0f}")
    else:
        # CAP-style: observed cut at the picks, synthetic cut at the modeled arrival
        # (where the synthetic energy is); the per-trace CC shift then aligns them.
        basis_anchors = model_anchors(forward)
        print("windows: observed on picks, synthetic on modeled arrivals (CAP-style)")

    # First-motion (Pz) polarity term (real mode only; synthetic recovery uses the
    # waveform alone). Observed polarity comes from the picker CSV; the synthetic
    # first-motion coefficient is read off the raw GF basis at the modeled onset.
    polarity, pol_by_station, tP_by = None, None, None
    if args.mode == "real" and args.polarity_weight > 0:
        pol_by_station = invio.load_pz_polarity(
            raw.get("event_id", "eq02387"), raw["station_ids"],
            PATHS["picks_file"], PATHS["polarity_cache"])
        # 3D store has no traveltime table: read the first motion just before the
        # P pick (pick lags the true onset by up to ~70 ms)
        tP_by = ({stc: pr["P"] - 0.05 for stc, pr in picks_rel.items()
                  if pr.get("P") is not None} if args.gf_npz else None)
        polarity = ds.build_polarity_coeffs(
            forward, pol_by_station,
            n_fm_sec=0.12 if args.gf_npz else 0.05, tP_by_station=tP_by)
        if polarity is None:
            print("polarity: no observed Pz polarities found -> term disabled")
        else:
            polarity["sigma"] = args.polarity_sigma
            polarity["weight"] = args.polarity_weight
            print(f"\npolarity (Pz): {len(polarity['meta'])} picks, "
                  f"sigma={args.polarity_sigma}, weight={args.polarity_weight}")
            print(f"  {'station':10s}{'obs_pol':>8s}{'inc':>7s}")
            for pm, inc in zip(polarity["meta"], polarity["inc"]):
                print(f"  {pm['station']:10s}{pm['polarity']:+8.2f}{inc:7.3f}")

    m6_ref = m6_from_sdr_mw(**GROND_DC_REF)
    if args.mode == "synthetic":
        synth_anchors, synth_pid = basis_anchors, None
        if args.synth_pid is not None:
            # location-recovery test: data generated from a non-nearest green point
            from src.forward import basis_onset_anchors
            synth_pid = args.synth_pid
            synth_anchors = basis_onset_anchors(forward, obs_anchors, pid=synth_pid)
            print(f"synthetic data from green point {synth_pid} at "
                  f"{forward.points_xyz_km[synth_pid]} km "
                  f"(nearest to catalog is {forward.pid})")
        dataset = ds.build_dataset_synthetic_picks(
            forward, m6_ref, synth_anchors, wins, filts,
            snr=args.snr, seed=args.seed, structure=args.structure, group_by=args.group_by,
            pid=synth_pid)
    else:
        dataset = ds.build_dataset_real_picks(
            forward, observed, event.time, basis_anchors, wins, filts, noise_anchors,
            obs_anchors=obs_anchors, noise_len=args.noise_len, noise_gap=args.noise_gap,
            structure=args.structure, group_by=args.group_by)

    location = None
    if args.sample_location:
        keep_nslc = {m["nslc"] for m in dataset.meta}
        print(f"\nlocation sampling: windowing the basis at {forward.n_points} "
              "green points ...")
        basis_all, a_pol_all = ds.build_location_stack(
            forward, obs_anchors, wins, filts, keep_nslc,
            pol_by_station=pol_by_station if polarity is not None else None,
            n_fm_sec=0.12, tP_by_station=tP_by)
        axes, pid_lut = forward.grid()
        lo = np.array([a[0] for a in axes])
        hi = np.array([a[-1] for a in axes])
        if args.loc_radius_m is not None:
            r_km = args.loc_radius_m / 1e3
            cat = np.asarray(source_xyz_km)
            lo, hi = np.maximum(lo, cat - r_km), np.minimum(hi, cat + r_km)
        location = dict(basis_all=basis_all, a_pol_all=a_pol_all, axes=axes,
                        pid_lut=pid_lut, lo=lo, hi=hi)
        print(f"location: {pid_lut.shape} node grid, "
              f"x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] "
              f"z[{lo[2]:.3f},{hi[2]:.3f}] km (uniform prior, nearest-node snap)")
    max_shift = int(round(args.max_shift_sec / forward.deltat))
    print(f"dataset: T={dataset.basis.shape[0]} targets, N={dataset.basis.shape[2]} samples, "
          f"G={dataset.n_groups} noise groups, autoshift=+/-{max_shift} samples "
          f"({args.max_shift_sec:.2f}s)")

    import jax
    logprior_fn, loglik_fn = M.build_logdensities(
        dataset, mw_bounds=mw_bounds, hp_bounds=hp_bounds, dc=args.dc,
        max_shift=max_shift, lag_penalty_coef=args.lag_penalty,
        gamma_beta=gamma_beta, delta_beta=delta_beta, polarity=polarity,
        location=location)
    if not args.dc:
        print(f"source-type prior: gamma~Beta{gamma_beta}, delta~Beta{delta_beta} (centred on DC)")
    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    particles = M.init_particles(kinit, args.num_particles, dataset.n_groups, dc=args.dc,
                                 gamma_beta=gamma_beta, delta_beta=delta_beta,
                                 sample_location=args.sample_location)

    result = run_smc(logprior_fn, loglik_fn, particles, key,
                     kernel=args.kernel, num_mcmc_steps=args.mcmc_steps,
                     mwg_mechanism_steps=args.mwg_mechanism_steps,
                     target_ess=args.target_ess, waste_free=args.waste_free)

    posterior = M.extract_posterior(
        result["particles"], result["weights"], mw_bounds, hp_bounds, dc=args.dc,
        loc_bounds=(location["lo"], location["hi"]) if location is not None else None)

    # Reference mechanism: synthetic recovery compares against the generating
    # (grond DC) tensor; real runs compare against the SMTI BlackJAX
    # polarity/amplitude-ratio solution (independent data types; the grond 1-D
    # waveform solution inherits too much velocity-model error to referee).
    ref_label = "true" if args.mode == "synthetic" else "grond DC"
    if args.mode == "real" and os.path.exists(PATHS["smti_reference_mt6"]):
        from src import smti_style_plots as ssp
        _ref = np.load(PATHS["smti_reference_mt6"])
        u = ssp.from_smti_mt6(np.median(_ref["mt6"], axis=1))
        # scale the unit tensor to the catalog Ml so the reference row shows a
        # meaningful magnitude (|m6| = sqrt(2) M0 for our NED 6-vector)
        m6_ref = np.sqrt(2.0) * pmt.magnitude_to_moment(2.64) * u / np.linalg.norm(u)
        ref_label = "SMTI ref"
    mean_m6, mw_est, kagan = report(m6_ref, ref_label, posterior, result["weights"])

    # Representative (MAP) sample: the highest-posterior particle. The posterior-mean
    # 6-vector is not a valid mechanism when the posterior is broad, so the waveform
    # fit is drawn from the MAP sample (recovers per-trace fits the mean smears out).
    logdens = np.asarray(jax.vmap(
        lambda p: logprior_fn(p) + loglik_fn(p))(result["particles"]))
    idx_map = int(np.argmax(logdens))
    map_m6 = posterior["m6"][idx_map]
    map_mt = m6_to_mt(map_m6)
    s1m, d1m, r1m = map_mt.both_strike_dip_rake()[0]
    mw_map = float(posterior["mw"][idx_map])
    kagan_map = float(pmt.kagan_angle(m6_to_mt(m6_ref), map_mt)) if m6_ref is not None else None
    print(f"MAP sample strike={s1m:6.1f}  dip={d1m:5.1f}  rake={r1m:6.1f}  Mw={mw_map:.2f}"
          + (f"   Kagan={kagan_map:.1f} deg" if kagan_map is not None else ""))

    map_pid = None
    if location is not None:
        loc = posterior["loc_xyz"]                       # (n, 3) km
        w = result["weights"]
        mean_xyz = np.average(loc, axis=0, weights=w)
        std_xyz = np.sqrt(np.average((loc - mean_xyz) ** 2, axis=0, weights=w))
        map_xyz = loc[idx_map]
        ijk = tuple(int(np.argmin(np.abs(a - map_xyz[q])))
                    for q, a in enumerate(location["axes"]))
        map_pid = int(location["pid_lut"][ijk])
        map_node = forward.points_xyz_km[map_pid]
        ref_xyz = (forward.points_xyz_km[args.synth_pid]
                   if args.mode == "synthetic" and args.synth_pid is not None
                   else np.asarray(source_xyz_km))
        ref_lab = "true point" if args.mode == "synthetic" else "catalog"
        print(f"location mean xyz = ({mean_xyz[0]:.3f}, {mean_xyz[1]:.3f}, "
              f"{mean_xyz[2]:.3f}) km  +/- ({std_xyz[0]*1e3:.0f}, {std_xyz[1]*1e3:.0f}, "
              f"{std_xyz[2]*1e3:.0f}) m")
        print(f"location MAP node {map_pid} at ({map_node[0]:.3f}, {map_node[1]:.3f}, "
              f"{map_node[2]:.3f}) km; offset from {ref_lab} = "
              f"({(map_node[0]-ref_xyz[0])*1e3:+.0f}, {(map_node[1]-ref_xyz[1])*1e3:+.0f}, "
              f"{(map_node[2]-ref_xyz[2])*1e3:+.0f}) m")

    if polarity is not None:
        m6n = mean_m6 / (np.linalg.norm(mean_m6) + 1e-30)
        X = polarity["a_pol"] @ m6n            # signed by obs: >0 = predicted agrees
        n_ok = int(np.sum(X > 0))
        print(f"polarity check (posterior mean): {n_ok}/{len(X)} picks agree")
        print(f"  {'station':10s}{'obs_pol':>8s}{'pred_X':>9s}  agree")
        for pm, x in zip(polarity["meta"], X):
            print(f"  {pm['station']:10s}{pm['polarity']:+8.2f}{x:+9.3f}  "
                  f"{'yes' if x > 0 else 'NO'}")

    if args.mode == "real":
        m6_grond = m6_from_sdr_mw(**GROND_DC_REF)
        print(f"secondary comparison, grond DC (1-D waveform): Kagan = "
              f"{pmt.kagan_angle(m6_to_mt(m6_grond), m6_to_mt(mean_m6)):.1f} deg")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    extra = ({"loc_xyz": posterior["loc_xyz"], "map_pid": map_pid}
             if location is not None else {})
    np.savez(args.out, m6=posterior["m6"], weights=result["weights"],
             mw=posterior["mw"], kappa=posterior["kappa"], h=posterior["h"],
             sigma=posterior["sigma"], hp=posterior["hp"], m6_ref=m6_ref,
             map_m6=map_m6, idx_map=idx_map, **extra)
    print(f"saved posterior -> {args.out}")

    from src import plotting
    os.makedirs(args.outdir, exist_ok=True)
    lc = None
    if location is not None:
        # draw the waveform fit / window plots at the MAP location, not the
        # catalog-nearest node the dataset was initially built on
        dataset.basis = location["basis_all"][map_pid]
        from src.forward import basis_onset_anchors
        basis_anchors = basis_onset_anchors(forward, obs_anchors, pid=map_pid)
        ref_xyz = (forward.points_xyz_km[args.synth_pid]
                   if args.mode == "synthetic" and args.synth_pid is not None
                   else np.asarray(source_xyz_km))
        lc = plotting.plot_location_posterior(
            posterior["loc_xyz"], result["weights"], ref_xyz,
            posterior["loc_xyz"][idx_map],
            os.path.join(args.outdir, f"{args.mode}_location.png"))
    bb = plotting.plot_fuzzy_beachball(
        posterior["m6"], m6_ref,
        os.path.join(args.outdir, f"{args.mode}_beachball.png"),
        title=f"eq02387 {args.mode} ({len(posterior['m6'])} samples)",
        ref_label=ref_label)
    wf = plotting.plot_waveform_fit(
        dataset, map_m6, forward.deltat, max_shift,
        os.path.join(args.outdir, f"{args.mode}_waveform_fit.png"), mw=mw_map,
        kagan=kagan_map, label="MAP sample")
    ft = None
    smti_paths = []
    if args.mode == "real":
        ft = plotting.plot_full_trace_windows(
            forward, observed, event.time, picks_rel, PHASE_OF,
            obs_anchors, basis_anchors, wins, filts, map_m6,
            os.path.join(args.outdir, "real_full_traces.png"), pid=map_pid)

        # SMTI-style figures: station-annotated beachballs (median + GMM mode),
        # Kaverina/Hudson HDI diagrams, arviz-style parameter posteriors.
        from src import smti_style_plots as ssp
        from pyrocko import gf as pyrocko_gf
        if pol_by_station is None:
            pol_by_station = invio.load_pz_polarity(
                raw.get("event_id", "eq02387"), raw["station_ids"],
                PATHS["picks_file"], PATHS["polarity_cache"])
        qseis_store = pyrocko_gf.LocalEngine(
            store_superdirs=PATHS["store_superdirs"]).get_store(PATHS["store_id"])
        station_geom = ssp.station_takeoff_azimuth(event, stations, qseis_store)

        bb_mm = ssp.plot_median_and_mode_beachballs(
            posterior["m6"], station_geom, pol_by_station, args.outdir, prefix=args.mode)
        print(f"posterior median vs mode Kagan: "
              f"{pmt.kagan_angle(m6_to_mt(bb_mm['median_m6']), m6_to_mt(bb_mm['mode_m6'])):.1f} deg")
        smti_paths += [bb_mm["median_path"], bb_mm["mode_path"]]

        kh = ssp.plot_kaverina_hudson(posterior["m6"], args.outdir, prefix=args.mode)
        smti_paths += [kh["kaverina_path"], kh["hudson_path"]]

        av = ssp.plot_arviz_posteriors(posterior, args.outdir, prefix=args.mode, dc=args.dc)
        smti_paths += [p for k, p in av.items() if k.endswith("_path")]

    print("saved plots ->\n  " + "\n  ".join(p for p in (bb, wf, ft, lc, *smti_paths) if p))


if __name__ == "__main__":
    main()
