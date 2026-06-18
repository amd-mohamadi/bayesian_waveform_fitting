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
