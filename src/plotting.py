"""Posterior visualizations: fuzzy beachball and waveform-fit overlays."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyrocko.plot import beachball
from pyrocko import moment_tensor as pmt


def m6_to_mt(m6):
    return pmt.MomentTensor(mnn=m6[0], mee=m6[1], mdd=m6[2], mne=m6[3], mnd=m6[4], med=m6[5])


def best_shift(obs, syn, max_shift):
    """Lag k (samples) minimizing ||obs - shift(syn, k)||, and the shifted synthetic."""
    if max_shift <= 0:
        return 0, syn
    n = len(syn)
    best_k, best_m, best_s = 0, np.inf, syn
    for k in range(-max_shift, max_shift + 1):
        s = np.zeros_like(syn)
        if k >= 0:
            s[k:] = syn[:n - k]
        else:
            s[:n + k] = syn[-k:]
        m = np.sum((obs - s) ** 2)
        if m < best_m:
            best_k, best_m, best_s = k, m, s
    return best_k, best_s


def plot_fuzzy_beachball(m6_samples, m6_true, outpath, n_show=500,
                         beachball_type="full", title="Posterior moment tensor"):
    """Fuzzy beachball from the posterior ensemble; mean (black) and true (blue) overlaid."""
    idx = np.linspace(0, len(m6_samples) - 1, min(n_show, len(m6_samples))).astype(int)
    mts = [m6_to_mt(m6_samples[i]) for i in idx]
    mean_mt = m6_to_mt(np.mean(m6_samples, axis=0))

    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_axis_off(); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    beachball.plot_fuzzy_beachball_mpl_pixmap(
        mts, ax, best_mt=mean_mt, beachball_type=beachball_type,
        position=(0.0, 0.0), size=2.0, size_units="data",
        color_t="red", color_p="white", edgecolor="black",
        best_color="black", grid_resolution=200)
    ax.set_title(f"{title}\nfuzzy posterior ({len(mts)} samples), black = mean", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_axis_off(); ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_aspect("equal")
    if m6_true is not None:
        beachball.plot_beachball_mpl(
            m6_to_mt(m6_true), ax2, beachball_type=beachball_type,
            position=(0.0, 0.0), size=2.0, size_units="data",
            color_t="blue", color_p="white", edgecolor="black", linewidth=1.5)
        kagan = pmt.kagan_angle(m6_to_mt(m6_true), mean_mt)
        ax2.set_title(f"true mechanism\nKagan(mean, true) = {kagan:.1f}deg", fontsize=10)
    else:
        beachball.plot_beachball_mpl(
            mean_mt, ax2, beachball_type=beachball_type, position=(0.0, 0.0),
            size=2.0, size_units="data", color_t="black", color_p="white", edgecolor="black")
        ax2.set_title("posterior-mean mechanism", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_full_trace_windows(forward, observed, event_time, picks_rel, phase_of,
                            obs_anchors, basis_anchors, wins, filts, m6, outpath,
                            t_max=3.6, pid=None):
    """Full-length obs vs synthetic (independently normalized) with the fit
    windows shaded -- the guard against windows missing the synthetic energy.

    Blue span = observed window (pick-anchored); orange span = synthetic window
    (basis/model-anchored); green/magenta dotted = P/S picks.
    """
    from .processing import process_trace
    trs = (forward.basis_pyrocko_traces(pid) if pid is not None
           else forward.basis_pyrocko_traces())
    tgrid = np.arange(0.0, t_max, forward.deltat)
    stations = sorted({m["nslc"][1] for m in forward.meta})
    chans = sorted({m["channel"] for m in forward.meta})
    nr, nc = len(stations), len(chans)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 1.6 * nr),
                             squeeze=False, sharex=True)
    for jt, m in enumerate(forward.meta):
        nslc = m["nslc"]
        i, j = stations.index(nslc[1]), chans.index(m["channel"])
        ax = axes[i][j]
        filt = filts[jt]
        syn = m6 @ np.stack([process_trace(trs[jt][k], tgrid, **filt) for k in range(6)])
        tr = observed.get(nslc)
        if tr is not None:
            obs = process_trace(tr, event_time + tgrid, **filt)
            ax.plot(tgrid, obs / (np.abs(obs).max() + 1e-30), "k", lw=0.6)
        ax.plot(tgrid, syn / (np.abs(syn).max() + 1e-30), "r", lw=0.6)
        t_pre, t_post = wins[jt]
        ax.axvspan(obs_anchors[jt] - t_pre, obs_anchors[jt] + t_post,
                   color="tab:blue", alpha=0.15)
        ax.axvspan(basis_anchors[jt] - t_pre, basis_anchors[jt] + t_post,
                   color="tab:orange", alpha=0.15)
        pr = picks_rel.get(nslc[:3], {})
        if pr.get("P") is not None:
            ax.axvline(pr["P"], color="g", lw=0.8, ls=":")
        if pr.get("S") is not None:
            ax.axvline(pr["S"], color="m", lw=0.8, ls=":")
        ax.set_yticks([]); ax.set_ylim(-1.2, 1.2)
        ax.text(0.01, 0.85, f"{nslc[1]}.{m['channel']}", transform=ax.transAxes, fontsize=8)
    for j, c in enumerate(chans):
        axes[0][j].set_title(c, fontsize=10)
        axes[-1][j].set_xlabel("source-relative time [s]", fontsize=8)
    fig.suptitle("Full traces (indep. normalized): obs (black) vs synthetic (red); "
                 "blue = observed window, orange = synthetic window, dotted = P/S picks",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_location_posterior(loc_xyz, weights, cat_xyz, map_xyz, outpath):
    """Source-location posterior over the green cloud: marginal x/y/z histograms
    + x-y scatter. Green = catalog hypocenter, red dashed/cross = MAP sample."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    labels = ["x east [km]", "y north [km]", "z depth [km]"]
    for q in range(3):
        ax = axes[q]
        ax.hist(loc_xyz[:, q], bins=30, weights=weights, color="tab:blue", alpha=0.75)
        ax.axvline(cat_xyz[q], color="g", lw=1.2, label="catalog")
        ax.axvline(map_xyz[q], color="r", lw=1.2, ls="--", label="MAP")
        ax.set_xlabel(labels[q]); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    ax = axes[3]
    ax.scatter(loc_xyz[:, 0], loc_xyz[:, 1], s=6, c=weights, cmap="viridis", alpha=0.6)
    ax.plot(cat_xyz[0], cat_xyz[1], "g*", ms=13, label="catalog")
    ax.plot(map_xyz[0], map_xyz[1], "r+", ms=13, mew=2, label="MAP")
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.suptitle("Source-location posterior (discrete green-point grid)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_waveform_fit(dataset, m6, deltat, max_shift, outpath, mw=None, kagan=None,
                      label="posterior mean"):
    """Observed vs synthetic MT overlays, one panel per station-channel.

    ``label`` names which MT is drawn (e.g. "posterior mean" or "MAP sample").
    """
    synth = np.einsum("k,tkn->tn", m6, dataset.basis)
    obs = dataset.data
    meta = dataset.meta
    N = obs.shape[1]
    t = np.arange(N) * deltat

    stations = sorted({m["nslc"][1] for m in meta})
    chans = sorted({m["channel"] for m in meta})
    nr, nc = len(stations), len(chans)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 1.25 * nr), squeeze=False, sharex=True)

    for jt, m in enumerate(meta):
        i = stations.index(m["nslc"][1]); j = chans.index(m["channel"])
        ax = axes[i][j]
        o, s = obs[jt], synth[jt]
        lag, s_al = best_shift(o, s, max_shift)
        vr = 1.0 - np.sum((o - s_al) ** 2) / (np.sum(o ** 2) + 1e-30)
        amp = np.max(np.abs(o)) or 1.0
        ax.plot(t, o / amp, "k", lw=0.7)
        ax.plot(t, s_al / amp, "r", lw=0.7)
        ax.set_yticks([]); ax.set_ylim(-1.3, 1.3)
        ax.text(0.02, 0.86, f"{m['nslc'][1]}.{m['channel']}  VR={vr:.2f}  dt={lag*deltat*1e3:+.0f}ms",
                transform=ax.transAxes, fontsize=7, va="top")
    for j, c in enumerate(chans):
        axes[0][j].set_title(c, fontsize=10)
        axes[-1][j].set_xlabel("time in window [s]", fontsize=8)

    sup = f"Waveform fit: observed (black) vs synthetic (red, {label})"
    if mw is not None:
        sup += f"   Mw={mw:.2f}"
    if kagan is not None:
        sup += f"   Kagan={kagan:.1f}deg"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return outpath
