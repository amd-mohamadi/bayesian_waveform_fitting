import matplotlib.pyplot as plt

from src_smc_mti.tape import MT33_MT6

try:
    from src.plot.plot_classes import _AmplitudePlot
except Exception:
    try:
        from src_smc_mti.plot.plot_classes import _AmplitudePlot
    except Exception:
        _AmplitudePlot = None


def plot_amplitude_beachball(mt33, output_path, title=None):
    """Plot a single MT beachball using the CAPE-style _AmplitudePlot."""
    if _AmplitudePlot is None:
        print("Warning: _AmplitudePlot unavailable; skipping beachball plot.")
        return

    mt6 = MT33_MT6(mt33).reshape(6, 1)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    amp_plot = _AmplitudePlot(
        None,
        fig,
        mt6,
        phase="P",
        projection="equalarea",
        lower=True,
        full_sphere=False,
        colormap="bwr",
        axis_lines=False,
        fault_plane=True,
        nodal_line=False,
        TNP=False,
        text=False,
        show=False,
        resolution=200,
    )
    amp_plot.plot()
    ax = amp_plot.ax
    ax.set_axis_off()
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
