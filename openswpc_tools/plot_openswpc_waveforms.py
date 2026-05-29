import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read


FNAME_RE = re.compile(
    r"^(?P<title>.+)\.3d\.(?P<station>st\d+)\.(?P<quantity>[UV])(?P<comp>[xyz])\.sac$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OpenSWPC SAC waveforms (U/V) by station and component"
    )
    parser.add_argument(
        "--wav-dir", required=True, help="Directory with OpenSWPC SAC files"
    )
    parser.add_argument(
        "--quantity",
        default="V",
        choices=["U", "V", "u", "v"],
        help="Plot displacement (U) or velocity (V)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <wav-dir>/openswpc_waveforms_<quantity>.png)",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=0,
        help="Maximum number of stations to plot (0 = all)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each trace by its peak absolute amplitude",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def station_key(station: str) -> tuple[int, str]:
    try:
        return int(station[2:]), station
    except Exception:
        return 10**9, station


def load_grouped_traces(wav_dir: Path, quantity: str):
    grouped = {}
    title = None
    qty = quantity.upper()

    for fp in sorted(wav_dir.glob("*.sac")):
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        if m.group("quantity").upper() != qty:
            continue

        st = m.group("station").lower()
        comp = m.group("comp").lower()
        if title is None:
            title = m.group("title")

        tr = read(str(fp))[0]
        grouped.setdefault(st, {})[comp] = tr

    if title is None:
        title = "openswpc"

    return grouped, title


def main() -> None:
    args = parse_args()
    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        raise FileNotFoundError(f"Waveform directory does not exist: {wav_dir}")

    grouped, title = load_grouped_traces(wav_dir, args.quantity)
    if not grouped:
        raise RuntimeError(
            f"No SAC files found for quantity={args.quantity.upper()} in {wav_dir}"
        )

    stations = sorted(grouped.keys(), key=station_key)
    if args.max_stations > 0:
        stations = stations[: args.max_stations]

    comps = ["x", "y", "z"]
    fig_h = max(6.0, 1.8 + 0.28 * len(stations))
    fig, axes = plt.subplots(3, 1, figsize=(12, fig_h), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ci, comp in enumerate(comps):
        ax = axes[ci]
        for si, st in enumerate(stations):
            tr = grouped.get(st, {}).get(comp)
            if tr is None:
                continue

            y = tr.data.astype(np.float64)
            if args.normalize:
                peak = np.max(np.abs(y))
                if peak > 0:
                    y = y / peak

            t = tr.times()
            y_off = y + float(si)
            ax.plot(t, y_off, lw=0.9, color="tab:blue")

        ax.set_ylabel(f"{args.quantity.upper()}{comp} + offset")
        ax.grid(alpha=0.25)

    # station labels on left of bottom panel using y offsets
    ax_last = axes[-1]
    for si, st in enumerate(stations):
        ax_last.text(
            0.0,
            float(si),
            st,
            fontsize=7,
            va="center",
            ha="right",
            transform=ax_last.get_yaxis_transform(),
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"OpenSWPC waveforms: {title} | quantity={args.quantity.upper()} | stations={len(stations)}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = (
        Path(args.output)
        if args.output
        else wav_dir / f"openswpc_waveforms_{args.quantity.upper()}.png"
    )
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved waveform plot: {out}")


if __name__ == "__main__":
    main()
