"""Load the Grond example fixture: event, stations, observed displacement traces."""
import os

import numpy as np
from pyrocko import model, io


def load_event(path):
    return model.load_events(path)[0]


def load_stations(path):
    return model.load_stations(path)


def load_observed(directory):
    """Load every trace file in ``directory`` into {nslc: pyrocko.Trace}."""
    obs = {}
    for fn in sorted(os.listdir(directory)):
        full = os.path.join(directory, fn)
        if not os.path.isfile(full):
            continue
        try:
            for tr in io.load(full):
                obs[(tr.network, tr.station, tr.location, tr.channel)] = tr
        except Exception:
            continue
    return obs


def event_m6_ned(event):
    """True moment tensor [mnn,mee,mdd,mne,mnd,med] in N*m, or None if absent."""
    mt = event.moment_tensor
    if mt is None:
        return None
    m = mt.m()  # 3x3 in NED
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])
