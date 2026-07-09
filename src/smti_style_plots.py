"""Bridge to SMTI's polarity/amplitude-ratio MT plotting toolkit, ported from
``cape_inversion_blackjax.py`` (the SMTI BlackJAX polarity+amplitude-ratio
inversion script) for side-by-side comparison with our own waveform-fit
posterior: beachball-with-station-polarities, posterior median/GMM-mode
mechanisms, Kaverina/Hudson HDI diagrams, and arviz-style parameter posteriors.

SMTI/src uses package-relative imports (``from ..forward_model import ...``),
so it is loaded here as an isolated package under the alias ``"smti_src"`` --
never added to sys.path as ``"src"``, which would collide with this project's
own top-level ``src`` package.
"""
import importlib
import importlib.util
import os
import sys

import numpy as np
from pyrocko import cake, orthodrome

_HERE = os.path.dirname(os.path.abspath(__file__))
_SMTI_SRC = os.path.abspath(os.path.join(_HERE, "..", "SMTI", "src"))

_module_cache = {}


def _smti(modname):
    """Import an SMTI/src submodule (e.g. "plot.plot_classes") under the
    isolated "smti_src" package alias, memoized."""
    if "smti_src" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "smti_src", os.path.join(_SMTI_SRC, "__init__.py"),
            submodule_search_locations=[_SMTI_SRC])
        module = importlib.util.module_from_spec(spec)
        sys.modules["smti_src"] = module
        spec.loader.exec_module(module)
    key = f"smti_src.{modname}"
    if key not in _module_cache:
        _module_cache[key] = importlib.import_module(key)
    return _module_cache[key]


def to_smti_mt6(m6_ned, normalize=True):
    """Our NED m6 [mnn,mee,mdd,mne,mnd,med] (shape (n,6) or (6,)) -> SMTI's
    MT6 convention [Mxx,Myy,Mzz,sqrt2 Mxy,sqrt2 Mxz,sqrt2 Myz] (x=N,y=E,z=D,
    same NED axes), shape (6,n). Verified against SMTI's own MT33_MT6 (which
    additionally unit-normalises): only the off-diagonal sqrt(2) scaling and
    6-vector order differ.
    """
    m6 = np.atleast_2d(np.asarray(m6_ned, dtype=float))     # (n,6)
    sq2 = np.sqrt(2.0)
    out = np.stack([m6[:, 0], m6[:, 1], m6[:, 2],
                    sq2 * m6[:, 3], sq2 * m6[:, 4], sq2 * m6[:, 5]], axis=0)  # (6,n)
    if normalize:
        out = out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-30)
    return out


def from_smti_mt6(mt6_smti):
    """Inverse of ``to_smti_mt6``: SMTI's MT6 (x=N,y=E,z=D, sqrt2 off-diag)
    -> our NED m6 [mnn,mee,mdd,mne,mnd,med]. Accepts shape (6,) or (6,n);
    returns (6,) or (n,6) respectively.
    """
    mt6 = np.asarray(mt6_smti, dtype=float)
    sq2 = np.sqrt(2.0)
    if mt6.ndim == 1:
        return np.array([mt6[0], mt6[1], mt6[2], mt6[3] / sq2, mt6[4] / sq2, mt6[5] / sq2])
    return np.stack([mt6[0], mt6[1], mt6[2], mt6[3] / sq2, mt6[4] / sq2, mt6[5] / sq2], axis=1)


def station_takeoff_azimuth(event, stations, qseis_store):
    """Per-station azimuth [deg] and P take-off angle [deg, 0=down/180=up] at
    the source, ray-traced through ``qseis_store``'s 1-D earth model. Used
    purely for station-marker placement on the focal sphere -- the
    inversion's own likelihood may use a different (e.g. 3-D) store.
    """
    mod = qseis_store.config.earthmodel_1d
    out = {}
    for st in stations:
        dist_m = orthodrome.distance_accurate50m(event.lat, event.lon, st.lat, st.lon)
        dist_deg = dist_m * cake.m2d
        d_receiver = getattr(st, "depth", 0.0) or 0.0
        phase = cake.PhaseDef("p" if event.depth > d_receiver else "P")
        rays = mod.arrivals(distances=[dist_deg], phases=[phase],
                            zstart=event.depth, zstop=d_receiver)
        takeoff = float(rays[0].takeoff_angle()) if rays else 90.0
        out[st.station] = (orthodrome.azimuth(event, st) % 360.0, takeoff)
    return out


def plot_beachball_with_stations(m6_ned, station_geom, pol_by_station, outpath,
                                 title=None, figsize=(5, 5)):
    """Beachball (SMTI's fault-plane P-amplitude _AmplitudePlot) with
    P-polarity station markers -- port of cape_inversion_blackjax.py's
    bb_median / bb_mode plot block.

    station_geom : {station_short_name: (azimuth_deg, takeoff_deg)}
    pol_by_station : {(net,sta,loc): signed polarity in [-1,1]}, matched to
        station_geom by short station name.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = _smti("plot.plot_classes")
    mtc = _smti("moment_tesnor_conversion")
    sp = _smti("plot.spherical_projection")

    mt6_smti = to_smti_mt6(m6_ned).reshape(6, 1)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    amp_plot = pc._AmplitudePlot(
        None, fig, mt6_smti, phase="P", projection="equalarea", lower=True,
        full_sphere=False, colormap="bwr", axis_lines=False, fault_plane=True,
        nodal_line=False, TNP=False, text=False, show=False, resolution=720)
    amp_plot.plot()
    ax = amp_plot.ax
    ax.set_axis_off()
    ax.set_aspect("equal")

    pol_by_name = {k[1]: v for k, v in pol_by_station.items()}
    names, az_l, to_l, pol_l = [], [], [], []
    for sta_short, (az, to) in station_geom.items():
        pol = pol_by_name.get(sta_short)
        if pol is None or pol == 0.0:
            continue
        names.append(sta_short); az_l.append(az); to_l.append(to); pol_l.append(pol)

    if names:
        v = mtc.toa_vec(np.asarray(az_l), np.asarray(to_l), radians=False)
        x = np.squeeze(np.array(v[0, :]))
        y = np.squeeze(np.array(v[1, :]))
        z = np.squeeze(np.array(v[2, :]))
        x_all, y_all, z_all = np.append(x, -x), np.append(y, -y), np.append(z, -z)
        pol_all = np.append(pol_l, pol_l)

        y_proj, x_proj = sp.equal_area(x_all, y_all, z_all, lower=True, full_sphere=False)
        idx = ~np.isnan(x_proj)
        Xp, Yp, pol_plot = x_proj[idx], y_proj[idx], pol_all[idx]
        names_plot = np.append(names, names)[idx]

        for i in range(len(Xp)):
            ax.text(Xp[i], Yp[i], names_plot[i], fontsize=7,
                   verticalalignment="bottom", horizontalalignment="left")
        up, down = pol_plot > 0, pol_plot < 0
        if np.any(up):
            ax.scatter(Xp[up], Yp[up], c="r", marker="o", edgecolors="k", s=50, label="P up")
        if np.any(down):
            ax.scatter(Xp[down], Yp[down], c="b", marker="o", edgecolors="k", s=50, label="P down")
        if np.any(up) or np.any(down):
            ax.legend(loc="upper right", bbox_to_anchor=(1.06, 1.04), fontsize=8,
                     frameon=False, borderaxespad=0.01, handletextpad=0.01, labelspacing=0.2)

    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_median_and_mode_beachballs(posterior_m6, station_geom, pol_by_station,
                                    outdir, prefix="real"):
    """Posterior median (component-wise) and GMM-mode moment tensors, each
    rendered with P-polarity stations -- SMTI's bb_median / bb_mode analogs.

    Returns dict(median_m6, mode_m6, median_path, mode_path).
    """
    util = _smti("utilities")
    smti_samples = to_smti_mt6(posterior_m6)          # (6, n), SMTI convention
    m6_median = np.median(posterior_m6, axis=0)        # our convention, (6,)
    _, idx_mode = util.find_map_mt6_gmm(smti_samples)
    m6_mode = posterior_m6[idx_mode]                   # same index, our convention

    out = {"median_m6": m6_median, "mode_m6": m6_mode}
    for tag, m6 in (("median", m6_median), ("mode", m6_mode)):
        out[f"{tag}_path"] = plot_beachball_with_stations(
            m6, station_geom, pol_by_station,
            os.path.join(outdir, f"{prefix}_bb_{tag}.png"),
            title=f"posterior {tag}")
    return out


def plot_kaverina_hudson(posterior_m6, outdir, prefix="real", hdi_prob=0.90,
                         max_samples=1500, seed=0):
    """Kaverina and Hudson diagrams with HDI density contours over the
    posterior MT samples (SMTI's kaverina_hdi_90 / hudson_hdi_90 analogs).
    Density contours only -- no median/mode/reference markers or legend.
    """
    pc = _smti("plot.plot_classes")
    samples = to_smti_mt6(posterior_m6)      # (6, n)
    n = samples.shape[1]
    if n > max_samples:
        idx = np.random.default_rng(seed).choice(n, max_samples, replace=False)
        samples = samples[:, idx]

    kav_path = os.path.join(outdir, f"{prefix}_kaverina_hdi_{int(hdi_prob * 100)}.png")
    hud_path = os.path.join(outdir, f"{prefix}_hudson_hdi_{int(hdi_prob * 100)}.png")
    pc.plot_kaverina_with_hdi(samples, hdi_prob=hdi_prob, output_path=kav_path,
                              grid_size=150, n_contour_levels=15, cmap="jet",
                              show_samples=False)
    pc.plot_hudson_with_hdi(samples, hdi_prob=hdi_prob, output_path=hud_path,
                            grid_size=150, n_contour_levels=15, cmap="jet",
                            show_samples=False)
    return {"kaverina_path": kav_path, "hudson_path": hud_path}


def plot_arviz_posteriors(posterior, outdir, prefix="real", dc=False, hdi_prob=0.90):
    """arviz-style marginal posterior plots for the Tape mechanism parameters
    (kappa, h, sigma) and, if not --dc, the source-type lune coordinates
    (gamma, delta) -- SMTI's posterior_dc.png / posterior_non_dc.png analogs.
    ``posterior`` is the dict returned by ``model.extract_posterior``.
    """
    import arviz as az
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dc_vars = [v for v in ("kappa", "h", "sigma") if v in posterior]
    non_dc_vars = [] if dc else [v for v in ("gamma", "delta") if v in posterior]
    all_vars = dc_vars + non_dc_vars
    idata = az.from_dict(posterior={
        v: np.asarray(posterior[v], dtype=float).reshape(1, -1) for v in all_vars})

    out = {}
    if dc_vars:
        az.plot_posterior(idata, var_names=dc_vars, hdi_prob=hdi_prob)
        p = os.path.join(outdir, f"{prefix}_posterior_dc.png")
        plt.gcf().savefig(p, dpi=150, bbox_inches="tight")
        plt.close("all")
        out["dc_path"] = p
    if non_dc_vars:
        az.plot_posterior(idata, var_names=non_dc_vars, hdi_prob=hdi_prob)
        p = os.path.join(outdir, f"{prefix}_posterior_non_dc.png")
        plt.gcf().savefig(p, dpi=150, bbox_inches="tight")
        plt.close("all")
        out["non_dc_path"] = p

    summary_df = az.summary(idata, var_names=all_vars, hdi_prob=hdi_prob)
    summary_path = os.path.join(outdir, f"{prefix}_arviz_summary.csv")
    summary_df.to_csv(summary_path)
    out["summary_csv"] = summary_path
    out["summary"] = summary_df
    return out
