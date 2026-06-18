"""Generate posterior visualizations from a saved run, reconstructing the dataset.

The SMC posterior is loaded from the .npz; the (deterministic) dataset is rebuilt
from the same window/filter args, so no re-inversion is needed.

    conda run -n pymc python -m src.plot_results --mode real  --posterior runs/real_adapt.npz
    conda run -n pymc python -m src.plot_results --mode synthetic --posterior runs/syn_adapt.npz --t-post 150
"""
import argparse
import os

import numpy as np
from pyrocko import moment_tensor as pmt

from . import data as data_io
from .forward import GFForward
from . import dataset as ds
from . import plotting
from ..run_regional_cmt import PATHS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="real")
    ap.add_argument("--posterior", default="runs/real_adapt.npz")
    ap.add_argument("--channels", default="Z,R,T")
    ap.add_argument("--fmin", type=float, default=0.01)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--tfade", type=float, default=20.0)
    ap.add_argument("--t-pre", type=float, default=20.0)
    ap.add_argument("--t-post", type=float, default=300.0)
    ap.add_argument("--max-shift-sec", type=float, default=None)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beachball-type", default="full")
    ap.add_argument("--outdir", default="report")
    args = ap.parse_args()

    channels = tuple(args.channels.split(","))
    filt = dict(fmin=args.fmin, fmax=args.fmax, order=args.order, tfade=args.tfade)
    max_shift_sec = args.max_shift_sec if args.max_shift_sec is not None \
        else (0.0 if args.mode == "synthetic" else 15.0)

    event = data_io.load_event(PATHS["event"])
    stations = data_io.load_stations(PATHS["stations"])
    m6_true = data_io.event_m6_ned(event)
    forward = GFForward(PATHS["store_superdirs"], PATHS["store_id"], event, stations, channels)
    max_shift = int(round(max_shift_sec / forward.deltat))

    if args.mode == "synthetic":
        dataset = ds.build_dataset_synthetic(
            forward, m6_true, args.t_pre, args.t_post, filt,
            snr=args.snr, seed=args.seed, group_by="channel")
    else:
        observed = data_io.load_observed(PATHS["obs_dir"])
        dataset = ds.build_dataset_real(
            forward, observed, event.time, args.t_pre, args.t_post, filt, group_by="trace")

    post = np.load(args.posterior)
    m6, w = post["m6"], post["weights"]
    w = w / w.sum()
    mean_m6 = np.average(m6, axis=0, weights=w)
    mean_mt = plotting.m6_to_mt(mean_m6)
    mw = mean_mt.magnitude
    kagan = pmt.kagan_angle(plotting.m6_to_mt(m6_true), mean_mt) if m6_true is not None else None

    os.makedirs(args.outdir, exist_ok=True)
    bb = plotting.plot_fuzzy_beachball(
        m6, m6_true, os.path.join(args.outdir, f"{args.mode}_beachball.png"),
        beachball_type=args.beachball_type,
        title=f"{args.mode} run ({len(m6)} posterior samples)")
    wf = plotting.plot_waveform_fit(
        dataset, mean_m6, forward.deltat, max_shift,
        os.path.join(args.outdir, f"{args.mode}_waveform_fit.png"), mw=mw, kagan=kagan)
    print(f"saved:\n  {bb}\n  {wf}")


if __name__ == "__main__":
    main()
