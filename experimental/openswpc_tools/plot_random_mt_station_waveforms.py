import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read


FNAME_RE = re.compile(
    r"^(?P<title>.+)\.3d\.(?P<label>st\d+)\.(?P<quantity>[UV])(?P<comp>[xyz])\.sac$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one OpenSWPC random-MT synthetic waveform figure per station"
    )
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantity", default="V", choices=("V", "U", "v", "u"))
    parser.add_argument(
        "--suffix",
        default=".synthetic_random_mt",
        help="Filename suffix before .png. Use empty string to write station_id.png.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    wav_dir = case_dir / "out" / "wav"
    meta_path = case_dir / "station_order.json"
    if not meta_path.exists():
        meta_path = case_dir / "case_metadata.json"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing OpenSWPC waveform directory: {wav_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing station metadata: {case_dir / 'station_order.json'} or {case_dir / 'case_metadata.json'}"
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    label_to_station = {
        str(item["label"]).lower(): str(item["station_id"])
        for item in metadata.get("stations", [])
    }
    grouped: dict[str, dict[str, object]] = {}
    qty = args.quantity.upper()
    title = "random_mt"

    for fp in sorted(wav_dir.glob("*.sac")):
        m = FNAME_RE.match(fp.name)
        if not m or m.group("quantity").upper() != qty:
            continue
        label = m.group("label").lower()
        comp = m.group("comp").lower()
        title = m.group("title")
        grouped.setdefault(label, {})[comp] = read(str(fp))[0]

    if not grouped:
        raise RuntimeError(f"No {qty} SAC waveforms found in {wav_dir}")

    comp_order = [("z", "Z"), ("y", "N"), ("x", "E")]
    written = []
    for label, station_id in label_to_station.items():
        traces = grouped.get(label, {})
        if not traces:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        peak = 0.0
        for comp, _ in comp_order:
            tr = traces.get(comp)
            if tr is not None:
                peak = max(peak, float(np.max(np.abs(tr.data))))
        scale = peak if peak > 0 else 1.0

        for ax, (comp, comp_label) in zip(axes, comp_order):
            tr = traces.get(comp)
            if tr is None:
                ax.text(0.5, 0.5, f"Missing {comp_label}", ha="center", va="center")
            else:
                t = tr.times()
                y = np.asarray(tr.data, dtype=float) / scale
                ax.plot(t, y, color="tab:red", lw=1.0)
                ax.set_xlim(0.0, t[-1] if len(t) else 0.0)
            ax.set_ylabel(f"{qty}{comp_label}")
            ax.grid(alpha=0.25)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{station_id} synthetic random MT | {title} | peak={peak:.3e}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out = out_dir / f"{station_id}{args.suffix}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        written.append(out)

    for out in written:
        print(out)
    print(f"Saved {len(written)} station waveform plots")


if __name__ == "__main__":
    main()
