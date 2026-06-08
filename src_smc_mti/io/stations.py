from pathlib import Path
from typing import List, Tuple, Union
import glob

import numpy as np
import obspy
import pyproj


def _station_id_from_xml(xml_file: str, net_code: str, sta_code: str, cha) -> str:
    stem = Path(xml_file).stem
    parts = stem.split(".")
    if len(parts) >= 4 and parts[0] == net_code and parts[1] == sta_code:
        return ".".join(parts[:4])

    loc = getattr(cha, "location_code", "")
    chan = getattr(cha, "code", "")
    band = chan[:2] if len(chan) >= 2 else chan
    return f"{net_code}.{sta_code}.{loc}.{band}"


def load_stations_from_xml(
    station_xml_dir: Union[str, Path],
) -> Tuple[np.ndarray, List[str], float, float]:
    """
    Parse StationXML files and return station rows, station ids, and centroid.

    Station rows use Axitra/OpenSWPC convention: [idx, x_m, y_m, depth_m].
    """
    station_xml_dir = Path(station_xml_dir)
    easting_ref = 335697.57
    northing_ref = 4263390.77
    transformer = pyproj.Proj("EPSG:26912", preserve_units=False)

    xml_files = sorted(glob.glob(str(station_xml_dir / "*.xml")))
    stations = []
    codes = []
    processed = set()
    idx = 1
    for xml_file in xml_files:
        try:
            inv = obspy.read_inventory(xml_file)
        except Exception:
            continue
        for net in inv:
            for sta in net:
                if len(sta) == 0:
                    continue
                cha = sta[0]
                station_id = _station_id_from_xml(xml_file, net.code, sta.code, cha)
                if station_id in processed:
                    continue
                try:
                    easting, northing = transformer(cha.longitude, cha.latitude)
                except Exception:
                    continue
                stations.append(
                    [
                        idx,
                        float(easting - easting_ref),
                        float(northing - northing_ref),
                        float(cha.depth),
                    ]
                )
                codes.append(station_id)
                processed.add(station_id)
                idx += 1

    if not stations:
        raise ValueError(f"No stations found in {station_xml_dir}")

    arr = np.array(stations, dtype=float)
    cx, cy = float(arr[:, 1].mean()), float(arr[:, 2].mean())
    return arr, codes, cx, cy
