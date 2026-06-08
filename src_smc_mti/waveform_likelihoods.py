import numpy as np
import torch
import torch.nn.functional as F

try:
    from tslearn.metrics import soft_dtw as tslearn_soft_dtw
    from tslearn.metrics.soft_dtw_fast import (
        _njit_soft_dtw_batch as tslearn_njit_soft_dtw_batch,
    )
except Exception:
    tslearn_soft_dtw = None
    tslearn_njit_soft_dtw_batch = None


class L2Likelihood:
    """Simple L2 least-squares waveform likelihood for debugging."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def compute_log_likelihood(
        self, synthetics: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        """
        Compute L2 log-likelihood: -0.5 * sum((syn - obs)^2) / sigma^2.
        """
        obs = observation[np.newaxis, ...]
        residuals = synthetics - obs
        ss_res = np.sum(residuals**2, axis=(1, 2, 3))
        return -0.5 * ss_res / (self.sigma**2)


class SoftDTWLikelihood:
    """
    Normalized Soft-DTW likelihood for waveform traces.

    Inputs:
      synthetics: (B, N, C, T)
      observation: (N, C, T)
    """

    def __init__(
        self,
        sigma=0.05,
        gamma=0.05,
        device="cpu",
        downsample_to=48,
        warp_radius=None,
        l2_weight=0.15,
        corr_weight=0.10,
        amp_weight=0.12,
        aggregation="median",
        normalize_mode="demean",
        backend="torch",
        eps=1e-8,
    ):
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.downsample_to = downsample_to
        self.warp_radius = None if warp_radius is None else int(warp_radius)
        self.l2_weight = float(l2_weight)
        self.corr_weight = float(corr_weight)
        self.amp_weight = float(amp_weight)
        if backend not in ("torch", "tslearn"):
            raise ValueError("backend must be 'torch' or 'tslearn'")
        if backend == "tslearn" and tslearn_soft_dtw is None:
            raise RuntimeError(
                "Soft-DTW backend 'tslearn' requested but tslearn is unavailable."
            )
        self.backend = backend
        if aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        self.aggregation = aggregation
        if normalize_mode not in ("none", "demean", "zscore"):
            raise ValueError("normalize_mode must be 'none', 'demean', or 'zscore'")
        self.normalize_mode = normalize_mode
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        if self.backend == "tslearn" and tslearn_njit_soft_dtw_batch is not None:
            self._warmup_tslearn_njit()

    def _zscore(self, x):
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / std

    def _soft_dtw_batch(self, x, y):
        bsz, npts = x.shape
        dmat = (x[:, :, None] - y[:, None, :]) ** 2

        if self.warp_radius is not None:
            ii = torch.arange(npts, device=x.device)
            jj = torch.arange(npts, device=x.device)
            outside = (ii[:, None] - jj[None, :]).abs() > self.warp_radius
            dmat = dmat + outside.to(dmat.dtype)[None, :, :] * 1e6

        r = torch.full(
            (bsz, npts + 1, npts + 1),
            float("inf"),
            device=x.device,
            dtype=x.dtype,
        )
        r[:, 0, 0] = 0.0

        g = self.gamma
        for i in range(1, npts + 1):
            for j in range(1, npts + 1):
                a = r[:, i - 1, j - 1]
                b = r[:, i - 1, j]
                c = r[:, i, j - 1]
                stacked = torch.stack((-a / g, -b / g, -c / g), dim=1)
                softmin = -g * torch.logsumexp(stacked, dim=1)
                r[:, i, j] = dmat[:, i - 1, j - 1] + softmin

        return r[:, npts, npts]

    def _soft_dtw_batch_tslearn(self, x, y):
        if tslearn_soft_dtw is None:
            raise RuntimeError("tslearn Soft-DTW backend is unavailable")

        x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64)
        bsz, npts = x_np.shape

        if tslearn_njit_soft_dtw_batch is not None:
            dmat = (x_np[:, :, None] - y_np[:, None, :]) ** 2
            if self.warp_radius is not None:
                ii = np.arange(npts)
                jj = np.arange(npts)
                outside = np.abs(ii[:, None] - jj[None, :]) > self.warp_radius
                dmat += outside[None, :, :] * 1e6
            r = np.empty((bsz, npts + 2, npts + 2), dtype=np.float64)
            tslearn_njit_soft_dtw_batch(dmat, r, float(self.gamma))
            out = r[:, npts, npts].astype(np.float32)
        else:
            out = np.empty(bsz, dtype=np.float32)
            for i in range(bsz):
                out[i] = float(tslearn_soft_dtw(x_np[i], y_np[i], gamma=self.gamma))

        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    def _warmup_tslearn_njit(self):
        if tslearn_njit_soft_dtw_batch is None:
            return
        d = np.zeros((1, 2, 2), dtype=np.float64)
        r = np.empty((1, 4, 4), dtype=np.float64)
        tslearn_njit_soft_dtw_batch(d, r, float(max(self.gamma, 1e-6)))

    def _soft_dtw_dispatch(self, x, y):
        if self.backend == "tslearn":
            return self._soft_dtw_batch_tslearn(x, y)
        return self._soft_dtw_batch(x, y)

    def compute_log_likelihood(self, synthetics, observation):
        if synthetics.ndim == 3:
            synthetics = synthetics[np.newaxis, ...]

        bsz, nsta, ncomp, npts = synthetics.shape
        ntr = nsta * ncomp

        syn = torch.from_numpy(synthetics).float().to(self.device)
        obs = torch.from_numpy(observation).float().to(self.device)

        if self.downsample_to is not None and self.downsample_to < npts:
            target = int(self.downsample_to)
            syn = F.interpolate(
                syn.reshape(bsz * ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(bsz, ntr, target)
            obs = F.interpolate(
                obs.reshape(ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(ntr, target)
            npts_eff = target
        else:
            syn = syn.reshape(bsz, ntr, npts)
            obs = obs.reshape(ntr, npts)
            npts_eff = npts

        syn_raw_flat = syn.reshape(bsz * ntr, npts_eff)
        obs_raw_rep = obs.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * ntr, npts_eff)
        obs_raw = obs

        if self.normalize_mode == "zscore":
            syn_flat = self._zscore(syn_raw_flat)
            obs_rep = self._zscore(obs_raw_rep)
            obs_norm = self._zscore(obs_raw)
        elif self.normalize_mode == "demean":
            syn_flat = syn_raw_flat - syn_raw_flat.mean(dim=1, keepdim=True)
            obs_rep = obs_raw_rep - obs_raw_rep.mean(dim=1, keepdim=True)
            obs_norm = obs_raw - obs_raw.mean(dim=1, keepdim=True)
        else:
            syn_flat = syn_raw_flat
            obs_rep = obs_raw_rep
            obs_norm = obs_raw

        with torch.no_grad():
            s_xy = self._soft_dtw_dispatch(syn_flat, obs_rep)
            s_xx = self._soft_dtw_dispatch(syn_flat, syn_flat)
            s_yy = self._soft_dtw_dispatch(obs_norm, obs_norm)

            s_yy_rep = s_yy.unsqueeze(0).expand(bsz, -1).reshape(bsz * ntr)
            div = s_xy - 0.5 * s_xx - 0.5 * s_yy_rep
            div = torch.clamp(div, min=0.0)

            mse = ((syn_flat - obs_rep) ** 2).mean(dim=1)
            corr = (syn_flat * obs_rep).mean(dim=1)
            corr_cost = 1.0 - corr
            syn_rms = torch.sqrt((syn_raw_flat**2).mean(dim=1) + self.eps)
            obs_rms = torch.sqrt((obs_raw_rep**2).mean(dim=1) + self.eps)
            amp_cost = torch.abs(torch.log(syn_rms) - torch.log(obs_rms))

            trace_cost = (
                div
                + self.l2_weight * mse
                + self.corr_weight * corr_cost
                + self.amp_weight * amp_cost
            )

            per_particle = trace_cost.reshape(bsz, ntr)
            if self.aggregation == "median":
                cost = torch.median(per_particle, dim=1).values
            else:
                cost = per_particle.mean(dim=1)
            ll = -0.5 * cost / (self.sigma**2)

        return ll.cpu().numpy()


class GSOTLikelihood:
    """
    Graph-space OT likelihood using Sinkhorn divergence on per-trace features.
    """

    def __init__(
        self,
        sigma=0.05,
        epsilon=0.03,
        device="cpu",
        downsample_to=48,
        n_iters=40,
        time_weight=0.30,
        amp_weight=1.00,
        slope_weight=0.35,
        l2_weight=0.08,
        corr_weight=0.06,
        amp_penalty_weight=0.08,
        aggregation="median",
        normalize_mode="demean",
        mass_floor=1e-3,
        eps=1e-8,
    ):
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.eps = float(eps)
        self.downsample_to = downsample_to
        self.n_iters = int(n_iters)
        self.time_weight = float(time_weight)
        self.amp_weight = float(amp_weight)
        self.slope_weight = float(slope_weight)
        self.l2_weight = float(l2_weight)
        self.corr_weight = float(corr_weight)
        self.amp_penalty_weight = float(amp_penalty_weight)
        self.mass_floor = float(mass_floor)
        if aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        self.aggregation = aggregation
        if normalize_mode not in ("none", "demean", "zscore"):
            raise ValueError("normalize_mode must be 'none', 'demean', or 'zscore'")
        self.normalize_mode = normalize_mode
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

    def _zscore(self, x):
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / std

    def _signal_features_and_mass(self, x):
        bsz, npts = x.shape
        time = torch.linspace(-0.5, 0.5, npts, device=x.device, dtype=x.dtype)
        time = time[None, :].expand(bsz, -1)

        dx = torch.zeros_like(x)
        if npts > 1:
            dx[:, 1:] = x[:, 1:] - x[:, :-1]

        feat = torch.stack(
            (
                self.time_weight * time,
                self.amp_weight * x,
                self.slope_weight * dx,
            ),
            dim=2,
        )

        mass = torch.abs(x) + self.mass_floor
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return feat, mass

    def _sinkhorn_cost_batch(self, a, b, c):
        k = torch.exp(-c / self.epsilon).clamp_min(self.eps)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.n_iters):
            kv = torch.bmm(k, v.unsqueeze(2)).squeeze(2).clamp_min(self.eps)
            u = a / kv
            ktu = (
                torch.bmm(k.transpose(1, 2), u.unsqueeze(2))
                .squeeze(2)
                .clamp_min(self.eps)
            )
            v = b / ktu

        transport = u.unsqueeze(2) * k * v.unsqueeze(1)
        return (transport * c).sum(dim=(1, 2))

    def compute_per_trace_cost(self, synthetics, observation):
        if synthetics.ndim == 3:
            synthetics = synthetics[np.newaxis, ...]

        bsz, nsta, ncomp, npts = synthetics.shape
        ntr = nsta * ncomp

        syn = torch.from_numpy(synthetics).float().to(self.device)
        obs = torch.from_numpy(observation).float().to(self.device)

        if self.downsample_to is not None and self.downsample_to < npts:
            target = int(self.downsample_to)
            syn = F.interpolate(
                syn.reshape(bsz * ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(bsz, ntr, target)
            obs = F.interpolate(
                obs.reshape(ntr, 1, npts),
                size=target,
                mode="linear",
                align_corners=False,
            ).reshape(ntr, target)
            npts_eff = target
        else:
            syn = syn.reshape(bsz, ntr, npts)
            obs = obs.reshape(ntr, npts)
            npts_eff = npts

        syn_raw_flat = syn.reshape(bsz * ntr, npts_eff)
        obs_raw_rep = obs.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * ntr, npts_eff)
        obs_raw = obs

        if self.normalize_mode == "zscore":
            syn_flat = self._zscore(syn_raw_flat)
            obs_rep = self._zscore(obs_raw_rep)
            obs_norm = self._zscore(obs_raw)
        elif self.normalize_mode == "demean":
            syn_flat = syn_raw_flat - syn_raw_flat.mean(dim=1, keepdim=True)
            obs_rep = obs_raw_rep - obs_raw_rep.mean(dim=1, keepdim=True)
            obs_norm = obs_raw - obs_raw.mean(dim=1, keepdim=True)
        else:
            syn_flat = syn_raw_flat
            obs_rep = obs_raw_rep
            obs_norm = obs_raw

        with torch.no_grad():
            syn_feat, syn_mass = self._signal_features_and_mass(syn_flat)
            obs_feat_rep, obs_mass_rep = self._signal_features_and_mass(obs_rep)
            obs_feat, obs_mass = self._signal_features_and_mass(obs_norm)

            c_xy = torch.cdist(syn_feat, obs_feat_rep, p=2) ** 2
            c_xx = torch.cdist(syn_feat, syn_feat, p=2) ** 2
            c_yy = torch.cdist(obs_feat, obs_feat, p=2) ** 2

            ot_xy = self._sinkhorn_cost_batch(syn_mass, obs_mass_rep, c_xy)
            ot_xx = self._sinkhorn_cost_batch(syn_mass, syn_mass, c_xx)
            ot_yy = self._sinkhorn_cost_batch(obs_mass, obs_mass, c_yy)

            ot_yy_rep = ot_yy.unsqueeze(0).expand(bsz, -1).reshape(bsz * ntr)
            div = torch.clamp(ot_xy - 0.5 * ot_xx - 0.5 * ot_yy_rep, min=0.0)

            mse = ((syn_flat - obs_rep) ** 2).mean(dim=1)
            corr = (syn_flat * obs_rep).mean(dim=1)
            corr_cost = 1.0 - corr
            syn_rms = torch.sqrt((syn_raw_flat**2).mean(dim=1) + self.eps)
            obs_rms = torch.sqrt((obs_raw_rep**2).mean(dim=1) + self.eps)
            amp_cost = torch.abs(torch.log(syn_rms) - torch.log(obs_rms))

            trace_cost = (
                div
                + self.l2_weight * mse
                + self.corr_weight * corr_cost
                + self.amp_penalty_weight * amp_cost
            )

            per_particle = trace_cost.reshape(bsz, ntr)

        return per_particle.cpu().numpy()

    def compute_log_likelihood(self, synthetics, observation, trace_weights=None):
        per_particle = self.compute_per_trace_cost(synthetics, observation)

        if trace_weights is None:
            if self.aggregation == "median":
                cost = np.median(per_particle, axis=1)
            else:
                cost = np.mean(per_particle, axis=1)
        else:
            w = np.asarray(trace_weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != per_particle.shape[1]:
                raise ValueError(
                    "trace_weights length mismatch: "
                    f"got {w.shape[0]}, expected {per_particle.shape[1]}"
                )
            w = np.clip(w, 0.0, None)
            wsum = float(np.sum(w))
            if wsum <= self.eps:
                raise ValueError("trace_weights must contain positive values")
            cost = np.sum(per_particle * w[None, :], axis=1) / wsum

        ll = -0.5 * cost / (self.sigma**2)
        return np.asarray(ll, dtype=np.float64)
