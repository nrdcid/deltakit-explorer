from unittest.mock import Mock

import numpy as np
import pytest

from deltakit_explorer.analysis.error_budget import (
    BoundSearchParameters,
    DiscretisationStrategy,
    FittingParameters,
    SamplingParameters,
    find_error_budget_bounds,
)


@pytest.fixture
def config():
    return BoundSearchParameters(parameter_domains=[(0, 1)])


def test_finder_requires_keyword_only_configuration():
    with pytest.raises(TypeError, match="search_parameters"):
        find_error_budget_bounds(Mock(), [0.02], {3: [3]})
    with pytest.raises(TypeError, match="positional"):
        find_error_budget_bounds(Mock(), [0.02], {3: [3]}, object())


@pytest.mark.parametrize("scale", [0.5, 1.0, 0.75])
def test_scaled_center_and_configured_width(scale):
    parameters = np.array([0.02, 0.04])
    original = parameters.copy()
    result = find_error_budget_bounds(
        noise_model=Mock(),
        noise_parameters=parameters,
        num_rounds_by_distances={3: [3], 5: [3]},
        search_parameters=BoundSearchParameters(
            parameter_domains=[(0, 1)] * 2,
            initial_relative_half_width=0.2,
        ),
        gradient_evaluation_scale=scale,
    )
    center = parameters * scale
    np.testing.assert_allclose(
        result.bounds, np.column_stack((center * 0.8, center * 1.2))
    )
    np.testing.assert_array_equal(parameters, original)


def test_default_center_and_width(config):
    result = find_error_budget_bounds(
        Mock(), [0.02], {3: [3]}, search_parameters=config
    )
    np.testing.assert_allclose(result.bounds, [(0.0075, 0.0125)])


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        *[
            ("gradient_evaluation_scale", value, "gradient_evaluation_scale")
            for value in [0, -1, np.nan, np.inf, True, [0.5, 1]]
        ],
        *[
            ("shots_per_trial", value, "shots_per_trial")
            for value in [0, -1, 1.5, True, np.inf]
        ],
        *[
            ("noise_parameters", value, "noise_parameters")
            for value in [[], 0.02, [[0.02]], [np.nan], [np.inf]]
        ],
    ],
)
def test_invalid_inputs(name, value, error, config):
    kwargs = {
        "noise_model": Mock(),
        "noise_parameters": [0.02],
        "num_rounds_by_distances": {3: [3]},
        "search_parameters": config,
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match=error):
        find_error_budget_bounds(**kwargs)


@pytest.mark.parametrize("name", ["batch_size", "max_workers"])
def test_invalid_sampling_execution_settings(name, config):
    with pytest.raises(ValueError, match=name):
        find_error_budget_bounds(
            Mock(),
            [0.02],
            {3: [3]},
            search_parameters=config,
            sampling_parameters=SamplingParameters(**{name: 0}),
        )


def test_new_options_do_not_sample_or_change_fitting(config):
    model, generator = Mock(), Mock()
    fitting = FittingParameters(
        discretisation_strategy=DiscretisationStrategy.LINEAR,
        fitting_degree=1,
        num_points_per_parameters=3,
    )
    result = find_error_budget_bounds(
        model,
        [0.02],
        {3: [3]},
        search_parameters=config,
        shots_per_trial=123,
        fitting_parameters=fitting,
        sampling_parameters=SamplingParameters(
            max_shots=0,
            lep_target_rse=-1,
            lep_computation_min_fails=0,
            batch_size=7,
            max_workers=2,
        ),
        memory_generator=generator,
        enable_correlations=True,
        seed=42,
    )
    assert result.shots_used == ()
    assert fitting.fitting_degree == 1
    assert fitting.num_points_per_parameters == 3
    model.assert_not_called()
    generator.assert_not_called()


def test_initial_interval_is_clipped_to_domain():
    result = find_error_budget_bounds(
        Mock(),
        [0.02],
        {3: [3]},
        search_parameters=BoundSearchParameters(parameter_domains=[(0.009, 0.011)]),
    )
    np.testing.assert_allclose(result.bounds, [(0.009, 0.011)])


@pytest.mark.parametrize("parameter", [0, 2, 3])
def test_center_must_be_inside_domain(parameter, config):
    with pytest.raises(ValueError, match="evaluation point.*domain"):
        find_error_budget_bounds(
            Mock(), [parameter], {3: [3]}, search_parameters=config
        )


def test_zero_center_requires_explicit_bounds():
    with pytest.raises(ValueError, match="zero.*explicit bounds"):
        find_error_budget_bounds(
            Mock(),
            [0],
            {3: [3]},
            search_parameters=BoundSearchParameters(parameter_domains=[(-1, 1)]),
        )


def test_logarithmic_fit_requires_positive_center():
    with pytest.raises(ValueError, match="logarithmic"):
        find_error_budget_bounds(
            Mock(),
            [-0.02],
            {3: [3]},
            search_parameters=BoundSearchParameters(parameter_domains=[(-1, 1)]),
        )


def test_logarithmic_lower_endpoint_stays_positive():
    result = find_error_budget_bounds(
        Mock(),
        [0.02],
        {3: [3]},
        search_parameters=BoundSearchParameters(
            parameter_domains=[(0, 1)], initial_relative_half_width=2
        ),
    )
    lower, upper = result.bounds[0]
    assert 0 < lower < 0.01 < upper <= 1


def test_linear_fit_accepts_negative_center():
    result = find_error_budget_bounds(
        Mock(),
        [-0.02],
        {3: [3]},
        search_parameters=BoundSearchParameters(parameter_domains=[(-1, 1)]),
        fitting_parameters=FittingParameters(
            discretisation_strategy=DiscretisationStrategy.LINEAR
        ),
    )
    np.testing.assert_allclose(result.bounds, [(-0.0125, -0.0075)])


@pytest.mark.parametrize(
    "name",
    [
        "num_rounds_per_distance",
        "gradient_evaluation_point",
        "parameter_indices",
        "initial_relative_width",
        "sensitivity_threshold",
        "logical_error_rate_min",
        "logical_error_rate_max",
        "max_iterations",
        "max_relative_width",
    ],
)
def test_removed_arguments_are_rejected(name, config):
    with pytest.raises(TypeError, match=name):
        find_error_budget_bounds(
            Mock(), [0.02], {3: [3]}, search_parameters=config, **{name: None}
        )
