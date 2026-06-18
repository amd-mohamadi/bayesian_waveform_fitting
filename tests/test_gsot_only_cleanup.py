from pathlib import Path

import numpy as np

import src.waveform_likelihoods as waveform_likelihoods
from run_inversion import build_config, clip_particles
from run_grond_regional_smoke import build_grond_regional_smoke_config
from src.pick_likelihoods import RadiationPickLikelihood
from src.workflows.axitra_gsot import (
    build_gaussian_toeplitz_covariance,
    build_source_delay_profiles,
    make_phase_window_taper,
)
from src.waveform_likelihoods import (
    GSOTLikelihood,
    GaussianToeplitzWaveformLikelihood,
    GrondCCMaxNormWaveformLikelihood,
    GrondL1GroupedWaveformLikelihood,
    GrondL1WaveformLikelihood,
    GrondPhaseRatioLikelihood,
)


def test_run_inversion_uses_single_axitra_gsot_config():
    config = build_config()

    assert Path("run_cape_inversion.py").exists() is False
    assert config.likelihood == "grond_l1_grouped_cc"
    assert config.axitra_greens_dir is not None
    assert config.dc_only is True
    assert config.n_particles == 600
    assert config.n_stages == 16
    assert config.initial_design == "halton"
    assert config.synthetic_window_source == "picks"
    assert config.source_profile_enabled is False
    assert config.source_profile_east_offsets_m == (-500.0, 0.0, 500.0)
    assert config.source_profile_north_offsets_m == (-500.0, 0.0, 500.0)
    assert config.source_profile_depths_m == (3200.0,)
    assert config.components == "PZ,SR,ST"
    assert config.phase_window_p_len == 0.40
    assert config.phase_window_s_len == 0.40
    assert config.phase_window_p_pre == 0.15
    assert config.phase_window_s_pre == 0.20
    assert config.window_taper == "hanning"
    assert config.phase_window_p_tfade == 0.02
    assert config.phase_window_s_tfade == 0.03
    assert config.source_target_freq_hz == 10.0
    assert config.source_delay == 0.0
    assert config.source_delay_profile_enabled is True
    assert config.source_delay_profile_offsets_s == (-0.10, -0.05, 0.0, 0.05, 0.10)
    assert config.grond_polarity_weight == 0.5
    assert config.grond_target_balancing is False
    assert config.grond_target_balancing_niter == 100
    assert not hasattr(config, "grond_amplitude_scale_profile_enabled")
    assert not hasattr(config, "grond_amplitude_scale_mw_offsets")
    assert config.gaussian_noise_window_s == 120.0
    assert config.gaussian_noise_gap_before_p_s == 10.0
    assert config.pick_likelihood_weight == 0.0
    assert config.pick_likelihood_terms == (
        "PPolarity",
        "SHPolarity",
        "P/SHAmplitudeRatio",
        "P/SVAmplitudeRatio",
    )
    assert config.dc_initial_strike is None
    assert config.dc_initial_dip is None
    assert config.dc_initial_rake is None


def test_grond_regional_smoke_uses_gaussian_toeplitz_diagnostic_config():
    config = build_grond_regional_smoke_config()

    assert config.likelihood == "gaussian_toeplitz"
    assert config.grond_target_balancing is False
    assert config.synthetic_window_source == "velocity"
    assert config.source_delay_profile_enabled is True
    assert config.source_profile_enabled is True
    assert config.dc_only is False
    assert config.source_target_freq_hz == 0.30
    assert config.sigma == 1.0
    assert config.sigma_end == 1.0
    assert not hasattr(config, "grond_amplitude_scale_profile_enabled")


def test_run_inversion_no_large_cli_surface():
    text = Path("run_inversion.py").read_text(encoding="utf-8")

    assert "argparse" not in text
    assert "parser.add_argument" not in text
    assert "parse_args" not in text


def test_legacy_likelihood_classes_are_not_active_exports():
    assert hasattr(waveform_likelihoods, "GSOTLikelihood")
    assert hasattr(waveform_likelihoods, "GaussianToeplitzWaveformLikelihood")
    assert hasattr(waveform_likelihoods, "GrondL1WaveformLikelihood")
    assert hasattr(waveform_likelihoods, "GrondL1GroupedWaveformLikelihood")
    assert hasattr(waveform_likelihoods, "GrondCCMaxNormWaveformLikelihood")
    assert hasattr(waveform_likelihoods, "GrondPhaseRatioLikelihood")
    assert not hasattr(waveform_likelihoods, "SoftDTWLikelihood")
    assert not hasattr(waveform_likelihoods, "L2Likelihood")
    assert not hasattr(waveform_likelihoods, "ProfiledM0StudentTLikelihood")


def test_clip_particles_is_fixed_source_five_dimensional():
    particles = np.asarray([[-10.0, 10.0, 10.0, -1.0, 10.0]], dtype=float)

    clipped = clip_particles(particles)

    assert clipped.shape == (1, 5)
    assert -np.pi / 6 <= clipped[0, 0] <= np.pi / 6
    assert -np.pi / 2 <= clipped[0, 1] <= np.pi / 2
    assert 0.0 <= clipped[0, 2] < 2 * np.pi
    assert 0.0 <= clipped[0, 3] <= 1.0
    assert -np.pi / 2 <= clipped[0, 4] <= np.pi / 2


def test_source_delay_profiles_include_base_once():
    config = build_config()

    profiles = build_source_delay_profiles(config)

    labels = [label for label, _ in profiles]
    delays = [delay for _, delay in profiles]
    assert labels.count("base_delay") == 1
    assert delays[0] == 0.0
    assert sorted(delays) == [-0.1, -0.05, 0.0, 0.05, 0.1]


def test_grond_cos_taper_uses_phase_specific_short_fades():
    taper = make_phase_window_taper(
        100,
        0.40,
        0.40,
        taper="grond_cos",
        p_tfade=0.02,
        s_tfade=0.03,
    )

    assert taper.shape == (3, 100)
    assert np.allclose(taper[1], taper[2])
    assert np.all(taper >= 0.0)
    assert np.all(taper <= 1.0)
    assert np.allclose(taper[:, 50], 1.0)
    assert np.count_nonzero(taper[0] < 1.0) < np.count_nonzero(taper[1] < 1.0)


def test_gsot_trace_mask_excludes_missing_waveforms():
    obs_trace = np.sin(np.linspace(0.0, np.pi, 16, dtype=np.float32))
    syn_trace = obs_trace.copy()
    observation = np.stack([obs_trace, np.zeros_like(obs_trace)], axis=0)[None, :, :]

    synthetics_a = observation[None, :, :, :].copy()
    synthetics_b = synthetics_a.copy()
    synthetics_b[:, :, 1, :] = 100.0

    model = GSOTLikelihood(
        sigma=1.0,
        epsilon=0.05,
        downsample_to=None,
        n_iters=5,
        normalize_mode="zscore",
        aggregation="mean",
    )

    ll_a = model.compute_log_likelihood(
        synthetics_a, observation, trace_mask=np.asarray([[True, False]])
    )
    ll_b = model.compute_log_likelihood(
        synthetics_b, observation, trace_mask=np.asarray([[True, False]])
    )

    assert np.allclose(ll_a, ll_b)


def test_grond_l1_likelihood_is_sign_sensitive():
    obs_trace = np.sin(np.linspace(0.0, np.pi, 24, dtype=np.float32))
    observation = obs_trace[None, None, :]
    matching = observation[None, :, :, :].copy()
    reversed_trace = -matching

    model = GrondL1WaveformLikelihood(
        sigma=1.0,
        p_shift_samples=0,
        s_shift_samples=0,
        autoshift_penalty_max=0.10,
        norm_exponent=1,
        component_tokens=("PZ",),
        aggregation="mean",
    )

    ll_matching = model.compute_log_likelihood(matching, observation)
    ll_reversed = model.compute_log_likelihood(reversed_trace, observation)

    assert ll_matching[0] > ll_reversed[0]


def test_grond_grouped_l1_uses_family_summed_misfit_norms():
    observation = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    small_trace_error = observation[None, :, :, :].copy()
    small_trace_error[:, :, 0, 0] = 0.0
    large_trace_error = observation[None, :, :, :].copy()
    large_trace_error[:, :, 1, 0] = 5.0

    trace_normalized = GrondL1WaveformLikelihood(
        sigma=1.0,
        p_shift_samples=0,
        s_shift_samples=0,
        autoshift_penalty_max=0.10,
        norm_exponent=1,
        component_tokens=("SR", "ST"),
        aggregation="mean",
    )
    grouped = GrondL1GroupedWaveformLikelihood(
        sigma=1.0,
        p_shift_samples=0,
        s_shift_samples=0,
        autoshift_penalty_max=0.10,
        norm_exponent=1,
        component_tokens=("SR", "ST"),
        aggregation="mean",
    )

    ll_trace_small = trace_normalized.compute_log_likelihood(
        small_trace_error, observation
    )
    ll_trace_large = trace_normalized.compute_log_likelihood(
        large_trace_error, observation
    )
    ll_grouped_small = grouped.compute_log_likelihood(small_trace_error, observation)
    ll_grouped_large = grouped.compute_log_likelihood(large_trace_error, observation)

    assert ll_trace_large[0] > ll_trace_small[0]
    assert ll_grouped_small[0] > ll_grouped_large[0]


def test_grond_grouped_l1_applies_target_weights():
    observation = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    first_trace_error = observation[None, :, :, :].copy()
    first_trace_error[:, :, 0, 0] = 0.0
    second_trace_error = observation[None, :, :, :].copy()
    second_trace_error[:, :, 1, 0] = 0.0

    model = GrondL1GroupedWaveformLikelihood(
        sigma=1.0,
        p_shift_samples=0,
        s_shift_samples=0,
        autoshift_penalty_max=0.10,
        norm_exponent=1,
        component_tokens=("SR", "ST"),
        aggregation="mean",
    )
    model.set_target_weights(np.asarray([10.0, 1.0], dtype=float))

    ll_first = model.compute_log_likelihood(first_trace_error, observation)
    ll_second = model.compute_log_likelihood(second_trace_error, observation)

    assert ll_second[0] > ll_first[0]


def test_grond_grouped_l1_no_longer_accepts_amplitude_scale_profile():
    try:
        GrondL1GroupedWaveformLikelihood(
            sigma=1.0,
            p_shift_samples=0,
            s_shift_samples=0,
            norm_exponent=1,
            component_tokens=("PZ",),
            aggregation="mean",
            amplitude_scales=(1.0, 2.0),
        )
    except TypeError:
        pass
    else:
        raise AssertionError("amplitude_scales should not be accepted")


def test_gaussian_toeplitz_prefers_exact_over_amplitude_mismatch():
    observation = np.asarray([[[1.0, -0.5, 0.25, 0.0]]], dtype=np.float32)
    exact = observation[None, :, :, :].copy()
    mismatched = 2.0 * exact
    weights = np.eye(4, dtype=float)[None, :, :]
    model = GaussianToeplitzWaveformLikelihood(
        sigma=1.0,
        chol_inv_weights=weights,
        logdets=np.asarray([0.0], dtype=float),
    )

    ll_exact = model.compute_log_likelihood(exact, observation)
    ll_mismatched = model.compute_log_likelihood(mismatched, observation)

    assert ll_exact[0] > ll_mismatched[0]


def test_gaussian_toeplitz_covariance_and_sigma_scale_residual_penalty():
    observation = np.zeros((1, 1, 4), dtype=np.float32)
    mismatched = np.ones((1, 1, 1, 4), dtype=np.float32)

    identity = GaussianToeplitzWaveformLikelihood(
        sigma=1.0,
        chol_inv_weights=np.eye(4, dtype=float)[None, :, :],
        logdets=np.asarray([0.0], dtype=float),
    )
    larger_cov = GaussianToeplitzWaveformLikelihood(
        sigma=1.0,
        chol_inv_weights=(0.5 * np.eye(4, dtype=float))[None, :, :],
        logdets=np.asarray([4 * np.log(4.0)], dtype=float),
    )
    larger_sigma = GaussianToeplitzWaveformLikelihood(
        sigma=2.0,
        chol_inv_weights=np.eye(4, dtype=float)[None, :, :],
        logdets=np.asarray([0.0], dtype=float),
    )

    exact_penalty_identity = (
        identity.compute_log_likelihood(mismatched, observation)[0]
        - identity.compute_log_likelihood(observation[None, :, :, :], observation)[0]
    )
    exact_penalty_large_cov = (
        larger_cov.compute_log_likelihood(mismatched, observation)[0]
        - larger_cov.compute_log_likelihood(observation[None, :, :, :], observation)[0]
    )
    exact_penalty_large_sigma = (
        larger_sigma.compute_log_likelihood(mismatched, observation)[0]
        - larger_sigma.compute_log_likelihood(observation[None, :, :, :], observation)[0]
    )

    assert exact_penalty_large_cov > exact_penalty_identity
    assert exact_penalty_large_sigma > exact_penalty_identity


def test_gaussian_toeplitz_respects_trace_mask():
    observation = np.zeros((1, 2, 4), dtype=np.float32)
    synthetics_a = observation[None, :, :, :].copy()
    synthetics_b = synthetics_a.copy()
    synthetics_b[:, :, 1, :] = 100.0
    weights = np.stack([np.eye(4, dtype=float), np.eye(4, dtype=float)], axis=0)
    model = GaussianToeplitzWaveformLikelihood(
        sigma=1.0,
        chol_inv_weights=weights,
        logdets=np.zeros(2, dtype=float),
    )

    ll_a = model.compute_log_likelihood(
        synthetics_a, observation, trace_mask=np.asarray([[True, False]])
    )
    ll_b = model.compute_log_likelihood(
        synthetics_b, observation, trace_mask=np.asarray([[True, False]])
    )

    assert np.allclose(ll_a, ll_b)


def test_gaussian_toeplitz_covariance_builder_positive_definite_and_masked():
    config = build_config()
    config.time_steps = 16
    config.gaussian_noise_window_s = 32.0
    config.gaussian_noise_gap_before_p_s = 2.0
    config.bp_p_low = None
    config.bp_p_high = None
    config.bp_s_low = None
    config.bp_s_high = None
    config.detrend = True
    config.demean = True

    rng = np.random.default_rng(123)
    npts = 96
    invdata = {
        "waveforms": {
            "XX.AAA.00.HH": {
                "starttime": "2020-01-01T00:00:00",
                "sampling_rate": 1.0,
                "delta": 1.0,
                "components": {
                    "Z": rng.normal(0.0, 1.0, npts).astype(np.float32),
                    "N": rng.normal(0.0, 1.0, npts).astype(np.float32),
                    "E": rng.normal(0.0, 1.0, npts).astype(np.float32),
                },
            }
        },
        "picks": {"XX.AAA.00.HH": {"P": "2020-01-01T00:00:50"}},
    }
    cov = build_gaussian_toeplitz_covariance(
        invdata=invdata,
        station_ids=["XX.AAA.00.HH"],
        component_tokens=("PZ", "SR", "ST"),
        stations=np.asarray([[1.0, 10.0, 0.0, 0.0]], dtype=float),
        source_loc=(0.0, 0.0, 5000.0),
        trace_mask=np.asarray([[True, False, True]]),
        args=config,
    )

    assert cov["chol_inv_weights"].shape == (3, 16, 16)
    assert cov["logdets"].shape == (3,)
    assert np.isfinite(cov["logdets"][[0, 2]]).all()
    assert np.allclose(cov["covariance_matrices"][1], np.eye(16))
    for idx in (0, 2):
        eigvals = np.linalg.eigvalsh(cov["covariance_matrices"][idx])
        assert np.min(eigvals) > 0.0


def test_grond_cc_likelihood_prefers_correlated_trace():
    obs_trace = np.sin(np.linspace(0.0, np.pi, 24, dtype=np.float32))
    observation = obs_trace[None, None, :]
    matching = observation[None, :, :, :].copy()
    reversed_trace = -matching

    model = GrondCCMaxNormWaveformLikelihood(
        sigma=1.0,
        p_shift_samples=0,
        s_shift_samples=0,
        component_tokens=("PZ",),
        aggregation="mean",
    )

    ll_matching = model.compute_log_likelihood(matching, observation)
    ll_reversed = model.compute_log_likelihood(reversed_trace, observation)

    assert ll_matching[0] > ll_reversed[0]


def test_grond_phase_ratio_likelihood_uses_peak_ratios():
    observation = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    matching = observation[None, :, :, :].copy()
    mismatched = observation[None, :, :, :].copy()
    mismatched[:, :, 0, :] *= 3.0

    model = GrondPhaseRatioLikelihood(
        sigma=1.0,
        component_tokens=("PZ", "SR", "ST"),
        pairs=(("PZ", "SR"),),
        fit_log_ratio=True,
        waterlevel=0.01,
        aggregation="mean",
    )

    ll_matching = model.compute_log_likelihood(matching, observation)
    ll_mismatched = model.compute_log_likelihood(mismatched, observation)

    assert ll_matching[0] > ll_mismatched[0]


def test_radiation_pick_likelihood_prefers_matching_polarity():
    model = RadiationPickLikelihood(
        polarity_a=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        polarity_sigma=np.asarray([0.1], dtype=float),
    )
    matching = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    reversed_mt = -matching

    ll_matching = model.compute_log_likelihood(matching)
    ll_reversed = model.compute_log_likelihood(reversed_mt)

    assert ll_matching[0] > ll_reversed[0]


def test_radiation_pick_likelihood_prefers_matching_ratio():
    model = RadiationPickLikelihood(
        ratio_num_a=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        ratio_den_a=np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        ratio_observed=np.asarray([2.0], dtype=float),
        ratio_sigma_log=np.asarray([0.2], dtype=float),
    )
    matching = np.asarray([[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    mismatched = np.asarray([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)

    ll_matching = model.compute_log_likelihood(matching)
    ll_mismatched = model.compute_log_likelihood(mismatched)

    assert ll_matching[0] > ll_mismatched[0]
