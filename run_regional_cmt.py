"""End-to-end driver: waveform-fitting MT inversion on the Grond regional CMT example.

Run from the repo root:
    conda run -n pymc python run_regional_cmt.py --mode synthetic
    conda run -n pymc python run_regional_cmt.py --mode real

Synthetic mode generates observed waveforms from the event.txt moment tensor (a
recovery test); real mode fits the prepared GE-network displacement data.
"""
import argparse
import os

import numpy as np
from pyrocko import moment_tensor as pmt

from src import data as data_io
from src.forward import GFForward
from src import dataset as ds
from src import model as M
from src.sampler import run_smc

EX = os.path.join(os.path.dirname(__file__), "grond", "examples", "example_regional_cmt")
EVENT_DIR = os.path.join(EX, "data", "events", "gfz2018pmjk")
PATHS = {
    "store_superdirs": [os.path.abspath(os.path.join(EX, "gf_stores"))],
    "store_id": "crust2_j3",
    "stations": os.path.join(EVENT_DIR, "waveforms", "stations.prepared.txt"),
    "obs_dir": os.path.join(EVENT_DIR, "waveforms", "prepared"),
    "event": os.path.join(EVENT_DIR, "event.txt"),
}


def m6_to_mt(m6):
    return pmt.MomentTensor(mnn=m6[0], mee=m6[1], mdd=m6[2], mne=m6[3], mnd=m6[4], med=m6[5])


def report(m6_true, posterior, weights):
    m6 = posterior["m6"]
    mean_m6 = np.average(m6, axis=0, weights=weights)
    mt_est = m6_to_mt(mean_m6)
    s1, d1, r1 = mt_est.both_strike_dip_rake()[0]
    mw_est = float(np.average(posterior["mw"], weights=weights))
    mw_std = float(np.sqrt(np.average((posterior["mw"] - mw_est) ** 2, weights=weights)))
    kagan = None

    print("\n================ POSTERIOR ================")
    print(f"estimated  strike={s1:6.1f}  dip={d1:5.1f}  rake={r1:6.1f}  Mw={mw_est:.2f} +/- {mw_std:.2f}")
    if m6_true is not None:
        mt_true = m6_to_mt(m6_true)
        s1t, d1t, r1t = mt_true.both_strike_dip_rake()[0]
        kagan = pmt.kagan_angle(mt_true, mt_est)
        print(f"true       strike={s1t:6.1f}  dip={d1t:5.1f}  rake={r1t:6.1f}  Mw={mt_true.magnitude:.2f}")
        print(f"Kagan angle (mechanism error): {kagan:.1f} deg")
    print("===========================================\n")
    return mean_m6, mw_est, kagan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--channels", default="Z,R,T")
    ap.add_argument("--fmin", type=float, default=0.01)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--tfade", type=float, default=20.0)
    ap.add_argument("--t-pre", type=float, default=20.0)
    ap.add_argument("--t-post", type=float, default=300.0)
    ap.add_argument("--num-particles", type=int, default=600)
    ap.add_argument("--mcmc-steps", type=int, default=10)
    ap.add_argument("--kernel", choices=["rmh", "hmc", "nuts"], default="rmh")
    ap.add_argument("--waste-free", action=argparse.BooleanOptionalAction, default=True,
                    help="waste-free SMC: keep every rejuvenation state (faster tempering "
                         "on large datasets); needs num-particles divisible by mcmc-steps")
    ap.add_argument("--hmc-int-steps", type=int, default=8,
                    help="HMC leapfrog steps per trajectory (kernel=hmc only)")
    ap.add_argument("--target-ess", type=float, default=0.6)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--dc", action="store_true", help="constrain to double couple")
    ap.add_argument("--gamma-beta", default="3,3",
                    help="Beta(alpha,beta) prior on lune longitude gamma; symmetric >1 "
                         "regularises toward double couple (ignored under --dc)")
    ap.add_argument("--delta-beta", default="3,3",
                    help="Beta(alpha,beta) prior on lune latitude delta; symmetric >1 "
                         "regularises toward double couple (ignored under --dc)")
    ap.add_argument("--mw-bounds", default="5.7,6.2",
                    help="magnitude prior (catalog-centred, grond regional-CMT style)")
    ap.add_argument("--hp-bounds", default="-2.0,10.0",
                    help="log-noise-scale hyperparameter prior; upper bound must exceed the "
                         "largest per-trace ML hp or quiet traces stay unwhitened and SMC stalls")
    ap.add_argument("--group-by", choices=["global", "channel", "station", "trace"], default=None,
                    help="noise-hyperparameter grouping (default: channel for synthetic, trace for real)")
    ap.add_argument("--max-shift-sec", type=float, default=None,
                    help="per-trace CC time-shift half-window in s (default: 0 synthetic, 15 real)")
    ap.add_argument("--lag-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/regional_cmt_posterior.npz")
    ap.add_argument("--outdir", default="report",
                    help="directory for the beachball + waveform-fit PNGs")
    args = ap.parse_args()

    channels = tuple(args.channels.split(","))
    mw_bounds = tuple(float(x) for x in args.mw_bounds.split(","))
    hp_bounds = tuple(float(x) for x in args.hp_bounds.split(","))
    gamma_beta = tuple(float(x) for x in args.gamma_beta.split(","))
    delta_beta = tuple(float(x) for x in args.delta_beta.split(","))
    filt = dict(fmin=args.fmin, fmax=args.fmax, order=args.order, tfade=args.tfade)
    group_by = args.group_by or ("channel" if args.mode == "synthetic" else "trace")
    max_shift_sec = args.max_shift_sec if args.max_shift_sec is not None \
        else (0.0 if args.mode == "synthetic" else 5.0)
    if args.mode == "real" and hp_bounds[0] < 0.0:
        print("note: real-data hp lower bound below 0 can force very small beta steps; "
              "try --hp-bounds 0,10 or --hp-bounds 2,10 for faster tempering.")

    event = data_io.load_event(PATHS["event"])
    stations = data_io.load_stations(PATHS["stations"])
    m6_true = data_io.event_m6_ned(event)
    print(f"event: lat={event.lat:.3f} lon={event.lon:.3f} depth={event.depth:.0f} m  "
          f"Mw_true={m6_to_mt(m6_true).magnitude:.2f}")

    forward = GFForward(PATHS["store_superdirs"], PATHS["store_id"], event, stations, channels)
    print(f"forward: {len(stations)} stations x {len(channels)} chan -> {len(forward.targets)} targets")

    if args.mode == "synthetic":
        dataset = ds.build_dataset_synthetic(
            forward, m6_true, args.t_pre, args.t_post, filt,
            snr=args.snr, seed=args.seed, group_by=group_by)
    else:
        observed = data_io.load_observed(PATHS["obs_dir"])
        dataset = ds.build_dataset_real(
            forward, observed, event.time, args.t_pre, args.t_post, filt,
            group_by=group_by)
    max_shift = int(round(max_shift_sec / forward.deltat))
    print(f"dataset: T={dataset.basis.shape[0]} targets, N={dataset.basis.shape[2]} samples, "
          f"G={dataset.n_groups} noise groups, autoshift=+/-{max_shift} samples ({max_shift_sec:.0f}s)")

    import jax
    logprior_fn, loglik_fn = M.build_logdensities(
        dataset, mw_bounds=mw_bounds, hp_bounds=hp_bounds, dc=args.dc,
        max_shift=max_shift, lag_penalty_coef=args.lag_penalty,
        gamma_beta=gamma_beta, delta_beta=delta_beta)
    if not args.dc:
        print(f"source-type prior: gamma~Beta{gamma_beta}, delta~Beta{delta_beta} "
              f"(centred on double couple)")
    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    particles = M.init_particles(kinit, args.num_particles, dataset.n_groups, dc=args.dc,
                                 gamma_beta=gamma_beta, delta_beta=delta_beta)

    result = run_smc(logprior_fn, loglik_fn, particles, key,
                     kernel=args.kernel, num_mcmc_steps=args.mcmc_steps,
                     target_ess=args.target_ess, waste_free=args.waste_free,
                     hmc_num_integration_steps=args.hmc_int_steps)

    posterior = M.extract_posterior(result["particles"], result["weights"],
                                    mw_bounds, hp_bounds, dc=args.dc)
    mean_m6, mw_est, kagan = report(m6_true, posterior, result["weights"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, m6=posterior["m6"], weights=result["weights"],
             mw=posterior["mw"], kappa=posterior["kappa"], h=posterior["h"],
             sigma=posterior["sigma"], hp=posterior["hp"],
             m6_true=m6_true if m6_true is not None else [])
    print(f"saved posterior -> {args.out}")

    from src import plotting
    os.makedirs(args.outdir, exist_ok=True)
    bb = plotting.plot_fuzzy_beachball(
        posterior["m6"], m6_true,
        os.path.join(args.outdir, f"{args.mode}_beachball.png"),
        title=f"{args.mode} run ({len(posterior['m6'])} posterior samples)")
    wf = plotting.plot_waveform_fit(
        dataset, mean_m6, forward.deltat, max_shift,
        os.path.join(args.outdir, f"{args.mode}_waveform_fit.png"), mw=mw_est, kagan=kagan)
    print(f"saved plots ->\n  {bb}\n  {wf}")


if __name__ == "__main__":
    main()
