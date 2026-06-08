import h5py


def load_observation(h5_path, index):
    """Load ground-truth synthetic observation from HDF5."""
    with h5py.File(h5_path, "r") as f:
        gt_params = {}
        for key in f["params"].keys():
            gt_params[key] = f["params"][key][index]
        observation = f["waveforms"][index]
    return gt_params, observation
