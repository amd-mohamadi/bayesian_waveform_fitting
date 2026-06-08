import matplotlib.pyplot as plt
import numpy as np


def station_misfit_summary(observation: np.ndarray, synthetics: np.ndarray) -> dict:
    """
    Compute simple per-station/per-component RMS misfit in processed space.
    """
    resid = synthetics - observation
    rms_station = np.sqrt(np.mean(resid**2, axis=(1, 2)))
    rms_station_comp = np.sqrt(np.mean(resid**2, axis=2))
    rms_comp = np.sqrt(np.mean(resid**2, axis=(0, 2)))
    return {
        "rms_station": rms_station,
        "rms_station_comp": rms_station_comp,
        "rms_comp": rms_comp,
    }


def plot_waveform_comparison(
    observation, best_synthetic, stations, output_path="waveform_comparison.png"
):
    """Plot observation vs best-fit synthetic waveform."""
    n_stations, _, n_time = observation.shape
    fig, axes = plt.subplots(n_stations, 3, figsize=(15, 1.8 * n_stations))
    component_names = ["Z (P-wave)", "N (S-wave)", "E (S-wave)"]
    time_axis = np.linspace(0, 0.14, n_time)

    for i in range(n_stations):
        for c in range(3):
            ax = axes[i, c] if n_stations > 1 else axes[c]
            ax.plot(
                time_axis, observation[i, c], "b-", label="Observation", linewidth=1.5
            )
            ax.plot(
                time_axis, best_synthetic[i, c], "r--", label="Best Fit", linewidth=1.5
            )
            if i == 0:
                ax.set_title(component_names[c])
            if c == 0:
                ax.set_ylabel(f"Station {i + 1}")
            if i == n_stations - 1:
                ax.set_xlabel("Time (s)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved waveform comparison to {output_path}")
    plt.close()
