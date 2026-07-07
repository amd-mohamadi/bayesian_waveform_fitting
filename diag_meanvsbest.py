"""Diagnostic: for a saved posterior, compare per-trace waveform VR of the
posterior-MEAN tensor (what the plot draws) vs the single best-fitting sample.

Rebuilds the real-data dataset exactly as run_eq02387_cmt.main() does, then scores
each m6 sample by total autoshift L2 misfit and reports FORK.Z / FORU.Z VRs.
"""
import argparse, os
import numpy as np

import run_eq02387_cmt as R
from src import invdata as invio
from src.forward import GFForward
from src import dataset as ds
from src.plotting import best_shift

ap = argparse.ArgumentParser()
ap.add_argument("--posterior", default="runs/eq02387_mt_mwg.npz")
ap.add_argument("--max-shift-sec", type=float, default=0.1)
ap.add_argument("--polarity-weight", type=float, default=100.0)
args0 = ap.parse_args()

# --- mirror main() dataset construction (real mode) ---
class A: pass
a = A()
a.p_pre, a.p_post, a.s_pre, a.s_post = 0.1, 0.3, 0.1, 0.3
a.order = 3
a.p_fmin, a.p_fmax, a.p_tfade = 2.0, 30.0, 0.1
a.s_fmin, a.s_fmax, a.s_tfade = 1.0, 20.0, 0.1
a.min_score = 0.0
a.noise_len, a.noise_gap = 0.8, 0.2
a.structure, a.group_by = "variance", "station"

event, stations, observed, picks_rel, raw = invio.load_invdata(R.PATHS["invdata"])
if R.FORK in picks_rel:
    picks_rel[R.FORK]["S"] = 1.25
stations, observed = invio.to_zrt(event, stations, observed)
forward = GFForward(R.PATHS["store_superdirs"], R.PATHS["store_id"], event, stations,
                    channels=("Z", "R", "T"), quantity="velocity")
obs_anchors, wins, filts, noise_anchors, dropped = R.build_window_specs(
    forward, picks_rel, a, set())
for nslc in dropped:
    observed.pop(nslc, None)
basis_anchors = R.model_anchors(forward)
dataset = ds.build_dataset_real_picks(
    forward, observed, event.time, basis_anchors, wins, filts, noise_anchors,
    obs_anchors=obs_anchors, noise_len=a.noise_len, noise_gap=a.noise_gap,
    structure=a.structure, group_by=a.group_by)

deltat = forward.deltat
max_shift = int(round(args0.max_shift_sec / deltat))
B = dataset.basis     # (T,6,N)
D = dataset.data      # (T,N)
W = dataset.weights   # (T,N) diag
meta = dataset.meta
labels = [f"{m['nslc'][1]}.{m['channel']}" for m in meta]

post = np.load(args0.posterior)
m6 = post["m6"]; w = post["weights"]
mean_m6 = np.average(m6, axis=0, weights=w)

def synth(m):           # (T,N)
    return np.einsum("k,tkn->tn", m, B)

def total_misfit(m):    # weighted autoshift L2 over all traces
    s = synth(m); tot = 0.0
    for t in range(D.shape[0]):
        _, s_al = best_shift(D[t], s[t], max_shift)
        r = (D[t] - s_al) * W[t]
        tot += float(np.sum(r ** 2))
    return tot

def per_trace_vr(m):
    s = synth(m); out = {}
    for t in range(D.shape[0]):
        _, s_al = best_shift(D[t], s[t], max_shift)
        vr = 1.0 - np.sum((D[t] - s_al) ** 2) / (np.sum(D[t] ** 2) + 1e-30)
        out[labels[t]] = vr
    return out

# best single sample by total weighted misfit
mis = np.array([total_misfit(m6[i]) for i in range(len(m6))])
ibest = int(np.argmin(mis))
print(f"\nposterior: {args0.posterior}  ({len(m6)} samples)")
print(f"mean-m6 total weighted misfit : {total_misfit(mean_m6):.4g}")
print(f"best-sample  (#{ibest}) misfit : {mis[ibest]:.4g}   (min over samples)")

vm = per_trace_vr(mean_m6); vb = per_trace_vr(m6[ibest])
print(f"\n{'trace':10s}{'VR(mean)':>10s}{'VR(best)':>10s}")
for lb in labels:
    star = "  <--" if lb in ("FORK.Z", "FORU.Z") else ""
    print(f"{lb:10s}{vm[lb]:10.2f}{vb[lb]:10.2f}{star}")

# Z-only mean VR summary
zlabels = [lb for lb in labels if lb.endswith(".Z")]
print(f"\nZ-mean VR: mean-m6={np.mean([vm[l] for l in zlabels]):.3f}  "
      f"best={np.mean([vb[l] for l in zlabels]):.3f}")
