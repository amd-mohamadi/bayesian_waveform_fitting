"""Green's function forward model: pyrocko/QSEIS engine + linear MT basis.

Displacement is linear in the 6 moment-tensor components, so for a fixed source
location we precompute 6 elementary "basis" seismograms per station-channel (one
per unit MT component, NED order [mnn, mee, mdd, mne, mnd, med]). Any moment
tensor's waveform is then the linear combination ``m6 @ basis`` -- verified exact
to ~2e-7 relative error against a direct full-MT synthetic.
"""
import numpy as np
from pyrocko import gf, orthodrome, trace as ptrace

MT_COMPONENTS = ("mnn", "mee", "mdd", "mne", "mnd", "med")  # NED, pyrocko convention


class GFForward:
    """Wraps a pyrocko LocalEngine on one GF store and produces the MT basis."""

    def __init__(self, store_superdirs, store_id, event, stations, channels=("Z", "R", "T"),
                 quantity="displacement"):
        self.engine = gf.LocalEngine(store_superdirs=list(store_superdirs))
        self.store_id = store_id
        self.store = self.engine.get_store(store_id)
        self.deltat = self.store.config.deltat
        self.event = event            # object with .lat .lon .depth .time
        self.stations = stations
        self.channels = channels
        self.quantity = quantity      # "displacement" | "velocity" | "acceleration"
        self.targets, self.meta = self._build_targets()

    def _build_targets(self):
        targets, meta = [], []
        for st in self.stations:
            for cha in self.channels:
                c = st.get_channel(cha)
                if c is None:
                    continue
                targets.append(gf.Target(
                    quantity=self.quantity,
                    codes=(st.network, st.station, st.location, cha),
                    lat=st.lat, lon=st.lon, depth=0.0,
                    store_id=self.store_id, interpolation="multilinear",
                    azimuth=c.azimuth, dip=c.dip,
                ))
                meta.append({"station": st, "channel": cha,
                             "nslc": (st.network, st.station, st.location, cha)})
        return targets, meta

    def _unit_sources(self):
        sources = []
        for cn in MT_COMPONENTS:
            kw = dict(lat=self.event.lat, lon=self.event.lon, depth=self.event.depth,
                      time=0.0, mnn=0., mee=0., mdd=0., mne=0., mnd=0., med=0.)
            kw[cn] = 1.0
            sources.append(gf.MTSource(**kw))
        return sources

    def arrivals(self):
        """P and S traveltimes (source-relative seconds) and epicentral distance per target."""
        out = []
        for m in self.meta:
            st = m["station"]
            dist = orthodrome.distance_accurate50m(
                self.event.lat, self.event.lon, st.lat, st.lon)
            args = (self.event.depth, dist)
            tp = self.store.t("any_P", args)
            ts = self.store.t("any_S", args)
            out.append({"distance": dist, "tp": tp, "ts": ts})
        return out

    def raw_basis(self):
        """Per target: dict(basis=(6, npts) array, tmin=source-relative, deltat).

        All 6 basis traces of a target share the same time axis.
        """
        resp = self.engine.process(self._unit_sources(), self.targets)
        rl = resp.results_list  # [isource][itarget]
        out = []
        for jt in range(len(self.targets)):
            arrs, tmin, npts = [], None, None
            for isrc in range(6):
                stc = rl[isrc][jt].trace
                y = np.asarray(stc.data, dtype=float)
                arrs.append(y)
                tmin = stc.tmin
                npts = len(y)
            B = np.zeros((6, npts))
            for isrc in range(6):
                B[isrc, :len(arrs[isrc])] = arrs[isrc]
            out.append({"basis": B, "tmin": tmin, "deltat": self.deltat})
        return out

    def basis_pyrocko_traces(self):
        """Per target: list of 6 pyrocko Trace objects (for processing/filtering)."""
        raw = self.raw_basis()
        out = []
        for jt, r in enumerate(raw):
            net, sta, loc, cha = self.meta[jt]["nslc"]
            trs = []
            for isrc in range(6):
                trs.append(ptrace.Trace(
                    network=net, station=sta, location=loc, channel=cha,
                    tmin=r["tmin"], deltat=r["deltat"], ydata=r["basis"][isrc].copy()))
            out.append(trs)
        return out


class NpzGFForward:
    """MT basis from an OpenSWPC reciprocity (green-mode) GF store npz.

    Same interface as GFForward where the dataset builders need it (`meta`,
    `deltat`, `event`, `stations`, `raw_basis()`, `basis_pyrocko_traces()`),
    but the basis comes from a packed 3-D store: gf (n_points, 6, n_sta, 3, N)
    in OpenSWPC conventions (mt order mxx,myy,mzz,myz,mxz,mxy; comp order
    x=E, y=N, z=down-positive; traces start at origin time).

    NED mapping (x=E, y=N, z=D): [mnn,mee,mdd,mne,mnd,med] =
    gf[[myy,mxx,mzz,mxy,myz,mxz]] = gf indices [1,0,2,5,3,4].
    Units: OpenSWPC green traces are nm/s per Nm (m_green.f90: UC_DERIV*1e9),
    hence scale=1e-9 -> m/s per Nm, matching the qseis basis. z_sign=+1
    verified empirically against qseis synthetics (P polarity on Z).
    """

    _NED_IDX = [1, 0, 2, 5, 3, 4]

    def __init__(self, npz_path, event, stations, source_xyz_km,
                 channels=("Z", "R", "T"), z_sign=1.0, scale=1e-9, all_points=False):
        d = np.load(npz_path)
        pts = d["points_xyz_km"]
        pid = int(np.argmin(np.sum((pts - np.asarray(source_xyz_km)) ** 2, axis=1)))
        self.pid = pid
        self.points_xyz_km = pts
        self.n_points = len(pts)
        self.point_xyz = pts[pid]
        self.point_offset_m = 1e3 * float(np.linalg.norm(pts[pid] - np.asarray(source_xyz_km)))
        self.deltat = float(d["dt"])
        self.event = event
        self.stations = stations
        self.channels = channels
        self._z_sign, self._scale = z_sign, scale

        sta_of_code = {str(c).split(".")[1]: i for i, c in enumerate(d["station_codes"])}
        self.meta, self._rot = [], []
        for st in stations:
            si = sta_of_code.get(st.station)
            if si is None:
                continue
            theta = np.deg2rad(orthodrome.azimuth(event, st))
            c, s = np.cos(theta), np.sin(theta)
            for cha in channels:
                self.meta.append({"station": st, "channel": cha,
                                  "nslc": (st.network, st.station, st.location, cha)})
                self._rot.append((si, cha, c, s))
        gf_all = d["gf"]                                    # (P, 6, n_sta, 3, N)
        self._gf_all = gf_all if all_points else None
        self._basis = self._point_basis(gf_all[pid])

    def _point_basis(self, gf):
        """Rotate one point's gf (6, n_sta, 3, N) to per-target NED bases (6, N)."""
        out = []
        for si, cha, c, s in self._rot:
            E, N = self._scale * gf[:, si, 0, :], self._scale * gf[:, si, 1, :]
            Z = self._z_sign * self._scale * gf[:, si, 2, :]  # comp order x,y,z
            by_cha = {"Z": Z, "R": N * c + E * s, "T": -N * s + E * c}
            out.append(np.asarray(by_cha[cha], dtype=float)[self._NED_IDX])
        return out

    def _basis_at(self, pid):
        if pid is None or pid == self.pid:
            return self._basis
        if self._gf_all is None:
            raise ValueError("pass all_points=True to access non-nearest green points")
        return self._point_basis(self._gf_all[pid])

    def grid(self):
        """Regular-grid view of the cloud: (axes [x, y, z] in km, pid_lut (nx, ny, nz))."""
        pts = np.round(self.points_xyz_km, 6)
        axes = [np.unique(pts[:, k]) for k in range(3)]
        lut = np.full([len(a) for a in axes], -1, dtype=int)
        idx = [np.searchsorted(axes[k], pts[:, k]) for k in range(3)]
        lut[idx[0], idx[1], idx[2]] = np.arange(len(pts))
        if (lut < 0).any():
            raise ValueError("green-point cloud is not a full regular grid")
        return axes, lut

    def raw_basis(self, pid=None):
        return [{"basis": B, "tmin": 0.0, "deltat": self.deltat}
                for B in self._basis_at(pid)]

    def basis_pyrocko_traces(self, pid=None):
        out = []
        for jt, B in enumerate(self._basis_at(pid)):
            net, sta, loc, cha = self.meta[jt]["nslc"]
            out.append([ptrace.Trace(network=net, station=sta, location=loc, channel=cha,
                                     tmin=0.0, deltat=self.deltat, ydata=B[i].copy())
                        for i in range(6)])
        return out


def basis_onset_anchors(forward, guess_anchors, search_s=0.35, pid=None):
    """Per-target synthetic-arrival anchors detected from the (raw) MT basis.

    CAP-style equivalent of a traveltime table for stores that have none (the
    3-D npz store): the mechanism-independent basis amplitude A(t) = ||B(:,t)||
    locates where the synthetic energy actually is. The search is confined to
    ``guess_anchors[jt] +/- search_s`` (the observed pick; the 3D model's
    timing error is well inside that), which keeps the detector away from the
    P precursor on horizontals and from any late-trace numerical tail.

    Anchor = first sample in the search window exceeding 20% of the window's
    own peak amplitude (leading edge of the phase).

    Returns a list of source-relative times aligned with ``forward.meta``.
    """
    raw = forward.raw_basis(pid) if pid is not None else forward.raw_basis()
    out = []
    for jt, r in enumerate(raw):
        A = np.linalg.norm(r["basis"], axis=0)
        dt, t0 = r["deltat"], r["tmin"]
        i0 = max(0, int(round((guess_anchors[jt] - search_s - t0) / dt)))
        i1 = min(len(A), int(round((guess_anchors[jt] + search_s - t0) / dt)) + 1)
        seg = A[i0:i1]
        k = i0 + int(np.argmax(seg > 0.2 * (seg.max() + 1e-30)))
        out.append(t0 + k * dt)
    return out
