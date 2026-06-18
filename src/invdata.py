"""Load a CAPE ``invdata.pkl`` into the objects the waveform-fitting pipeline needs.

``invdata.pkl`` stores, per event: the source (origin time, lat/lon, depth, mw),
per-station ENZ velocity waveforms (with absolute start time + sampling rate), and
P/S phase picks as absolute UTC strings. This module converts that into a pyrocko
Event, pyrocko Stations (with Z/N/E channels), an {nslc: Trace} observed dict, and
source-relative P/S pick offsets -- so windows can be anchored on the picks rather
than on a velocity-model traveltime.
"""
import pickle
from datetime import datetime, timezone

import numpy as np
from pyrocko import model, trace as ptrace, orthodrome

# ENZ component convention (pyrocko Channel azimuth, dip; dip +down, Z up = -90)
CHANNELS = {"Z": (0.0, -90.0), "N": (0.0, 0.0), "E": (90.0, 0.0)}


def utc(s):
    """Parse an ISO-ish UTC timestamp ('...T..Z' or '... ..') to a unix epoch float."""
    s = s.strip().replace("Z", "").replace("T", " ")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def load_invdata(path, depth_m=None):
    """Return (event, stations, observed, picks_rel, raw).

    event      : pyrocko.model.Event (depth overridable via ``depth_m`` in metres)
    stations   : list[pyrocko.model.Station] with Z/N/E channels
    observed   : {(net, sta, loc, cha): pyrocko.trace.Trace}  (ENZ velocity)
    picks_rel  : {(net, sta, loc): {"P", "S", "p_score", "s_score"}} source-relative s
    raw        : the loaded invdata dict (for any extra fields)
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    src = d["source"]
    ev_time = utc(src["origin_time"])
    depth = float(depth_m) if depth_m is not None else float(src["depth_km"]) * 1000.0
    mag = src.get("mw", src.get("magnitude"))
    event = model.Event(lat=float(src["latitude"]), lon=float(src["longitude"]),
                        depth=depth, time=ev_time,
                        magnitude=(float(mag) if mag is not None else None),
                        name=str(d.get("event_id", "")))

    stations, observed, picks_rel = [], {}, {}
    for sid in d["station_ids"]:
        net, sta, loc = sid.split(".")[:3]
        meta = d["station_metadata"][sid]
        st = model.Station(network=net, station=sta, location=loc,
                           lat=float(meta["latitude"]), lon=float(meta["longitude"]),
                           depth=0.0)
        for cha, (az, dip) in CHANNELS.items():
            st.add_channel(model.Channel(cha, azimuth=az, dip=dip))
        stations.append(st)

        wf = d["waveforms"][sid]
        t0, dt = utc(wf["starttime"]), float(wf["delta"])
        for cha in ("Z", "N", "E"):
            y = np.asarray(wf["components"][cha], dtype=float)
            observed[(net, sta, loc, cha)] = ptrace.Trace(
                network=net, station=sta, location=loc, channel=cha,
                tmin=t0, deltat=dt, ydata=y)

        pk = d["picks"].get(sid, {})
        picks_rel[(net, sta, loc)] = {
            "P": (utc(pk["P"]) - ev_time) if pk.get("P") else None,
            "S": (utc(pk["S"]) - ev_time) if pk.get("S") else None,
            "p_score": pk.get("p_score"),
            "s_score": pk.get("s_score"),
        }
    return event, stations, observed, picks_rel, d


def to_zrt(event, stations, observed):
    """Rotate ENZ stations + observed traces to ZRT (vertical/radial/transverse).

    Radial points from the source toward the station (azimuth theta = the
    event->station azimuth, dip 0); transverse is theta + 90 (dip 0); vertical (Z)
    is unchanged. The station R/T channel orientations and the observed R/T traces
    are rotated with the SAME convention, so the pyrocko engine -- which synthesises
    each target by projecting ground motion onto its channel (azimuth, dip) -- yields
    synthetics directly comparable to the rotated observed:

        R =  N*cos(theta) + E*sin(theta)
        T = -N*sin(theta) + E*cos(theta)

    R/T channels are added to each station; the returned observed dict keeps only
    Z/R/T (N/E dropped). Stations are modified in place.
    """
    new_obs = {}
    for st in stations:
        theta = orthodrome.azimuth(event, st)          # event->station, deg CW from N
        th = np.deg2rad(theta)
        c, s = np.cos(th), np.sin(th)
        st.add_channel(model.Channel("R", azimuth=theta, dip=0.0))
        st.add_channel(model.Channel("T", azimuth=theta + 90.0, dip=0.0))
        key = (st.network, st.station, st.location)
        trZ = observed.get(key + ("Z",))
        trN = observed.get(key + ("N",))
        trE = observed.get(key + ("E",))
        if trZ is not None:
            new_obs[key + ("Z",)] = trZ
        if trN is not None and trE is not None:
            yN, yE = trN.get_ydata(), trE.get_ydata()
            n = min(len(yN), len(yE))
            yN, yE = yN[:n], yE[:n]
            for cha, y in (("R", yN * c + yE * s), ("T", -yN * s + yE * c)):
                new_obs[key + (cha,)] = ptrace.Trace(
                    network=st.network, station=st.station, location=st.location,
                    channel=cha, tmin=trN.tmin, deltat=trN.deltat, ydata=y)
    return stations, new_obs
