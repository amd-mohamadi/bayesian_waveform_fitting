import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_inversion import select_station_geometry
from src_smc_mti.forward import FastSynthesizer
from src_smc_mti.io import load_stations_from_xml, load_velocity_model


def random_tape_params(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(-np.pi / 6.0, np.pi / 6.0),
            rng.uniform(-np.pi / 2.0, np.pi / 2.0),
            rng.uniform(0.0, 2.0 * np.pi),
            rng.uniform(0.0, 1.0),
            rng.uniform(-np.pi / 2.0, np.pi / 2.0),
        ],
        dtype=float,
    )


def plot_station(t: np.ndarray, y: np.ndarray, station_id: str, params: np.ndarray, out: Path) -> None:
    labels = ["Z", "X/E", "Y/N"]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    scale = peak if peak > 0 else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y[i] / scale, color="tab:red", lw=1.0)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{station_id} Axitra random MT | peak={peak:.3e}\n"
        f"gamma={params[0]:.3f}, delta={params[1]:.3f}, kappa={params[2]:.3f}, "
        f"h={params[3]:.3f}, sigma={params[4]:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fast Axitra random-MT synthetic waveform plots for a CAPE event"
    )
    parser.add_argument("--event-dir", default="cape_events/eq02387")
    parser.add_argument("--stations-dir", default="stations")
    parser.add_argument("--velocity-model", default="../forge.tvel")
    parser.add_argument("--output-dir", default="waveforms/eq02387")
    parser.add_argument("--seed", type=int, default=2387)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--fmax", type=float, default=50.0)
    parser.add_argument("--source-target-freq-hz", type=float, default=30.0)
    parser.add_argument("--source-delay", type=float, default=0.0)
    parser.add_argument(
        "--suffix",
        default=".axitra_random_mt",
        help="Filename suffix before .png. Use empty string to write station_id.png.",
    )
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    inv_path = event_dir / "invdata.pkl"
    with inv_path.open("rb") as f:
        invdata = pickle.load(f)

    source = invdata["source"]
    source_loc = (float(source["x"]), float(source["y"]), float(source["z"]))
    m0 = 10.0 ** (1.5 * float(source["mw"]) + 9.1)

    stations_all, codes_all, _, _ = load_stations_from_xml(args.stations_dir)
    stations, station_ids = select_station_geometry(
        stations_all, codes_all, [str(sid) for sid in invdata["station_ids"]]
    )
    velocity_model = load_velocity_model(args.velocity_model)
    mt_params = random_tape_params(args.seed)

    synth = FastSynthesizer(
        velocity_model,
        stations,
        source_loc,
        duration=float(args.duration),
        fmax=float(args.fmax),
        t0=1.0 / (np.pi * float(args.source_target_freq_hz)),
    )
    synth.setup()
    try:
        waveforms = synth.synthesize_batch(
            mt_params.reshape(1, 5),
            m0,
            source_delay=float(args.source_delay),
        )[0]
    finally:
        synth.cleanup()

    npts = waveforms.shape[-1]
    t = np.linspace(0.0, float(args.duration), npts, endpoint=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, sid in enumerate(station_ids):
        out = out_dir / f"{sid}{args.suffix}.png"
        plot_station(t, waveforms[i], sid, mt_params, out)
        written.append(out)

    print("Random Tape parameters:", " ".join(f"{x:.8g}" for x in mt_params))
    print(f"M0: {m0:.6e}")
    print(f"Axitra output shape: {waveforms.shape} (station, component, time)")
    for out in written:
        print(out)


if __name__ == "__main__":
    main()
