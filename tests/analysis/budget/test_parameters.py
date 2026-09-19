from dataclasses import fields
from unittest.mock import Mock

import numpy as np
import pytest

from deltakit_explorer.analysis.error_budget import (
    BoundSearchParameters,
    find_error_budget_bounds,
    get_error_budget,
)


def test_bound_search_defaults_and_required_domains():
    with pytest.raises(TypeError, match="parameter_domains"):
        BoundSearchParameters()

    parameters = BoundSearchParameters(parameter_domains=[(0, 1)])
    assert parameters.initial_relative_half_width == 0.25
    assert parameters.expansion_factor == 2.0
    assert parameters.sensitivity_z_score == 3.0
    assert parameters.min_logical_failures == 10
    assert parameters.max_lep == 0.45
    assert parameters.max_trials_per_parameter == 16
    assert {field.name for field in fields(parameters)} == {
        "parameter_domains",
        "initial_relative_half_width",
        "expansion_factor",
        "sensitivity_z_score",
        "min_logical_failures",
        "max_lep",
        "max_trials_per_parameter",
    }


@pytest.mark.parametrize(
    "domains",
    [
        [],
        [(0,)],
        [(0, 1, 2)],
        [(0, 1), (0,)],
        [(1, 1)],
        [(2, 1)],
        [(np.nan, 1)],
        [(0, np.inf)],
        [(-np.inf, 1)],
    ],
)
def test_bound_search_rejects_invalid_domains(domains):
    with pytest.raises(ValueError, match="parameter_domains"):
        BoundSearchParameters(parameter_domains=domains)


def test_bound_search_preserves_arbitrary_domains_without_mutable_aliases():
    domains = [[-2, -1], [10, 20]]
    parameters = BoundSearchParameters(parameter_domains=domains)
    domains[0][0] = np.nan
    assert parameters.parameter_domains == ((-2.0, -1.0), (10.0, 20.0))


def test_bound_search_accepts_valid_custom_values():
    parameters = BoundSearchParameters(
        parameter_domains=np.array([[-10.0, 10.0]]),
        initial_relative_half_width=2.0,
        expansion_factor=1.01,
        sensitivity_z_score=0.1,
        min_logical_failures=np.int64(1),
        max_lep=0.499,
        max_trials_per_parameter=1,
    )
    assert parameters.parameter_domains == ((-10.0, 10.0),)
    assert parameters.max_trials_per_parameter == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (name, value)
        for name, values in {
            "initial_relative_half_width": [0, -1, np.nan, np.inf, -np.inf],
            "expansion_factor": [0, 1, np.nan, np.inf, -np.inf],
            "sensitivity_z_score": [0, -1, np.nan, np.inf, -np.inf],
            "max_lep": [0, -1, 0.5, 1, np.nan, np.inf, -np.inf],
            "min_logical_failures": [0, -1, 1.5, True, np.nan, np.inf],
            "max_trials_per_parameter": [0, -1, 1.5, True, np.nan, np.inf],
        }.items()
        for value in values
    ],
)
def test_bound_search_rejects_invalid_field_ranges(name, value):
    with pytest.raises(ValueError, match=name):
        BoundSearchParameters(parameter_domains=[(0, 1)], **{name: value})


@pytest.mark.parametrize("entrypoint", ["finder", "budget"])
@pytest.mark.parametrize("domain_count", [1, 3])
def test_public_entrypoints_reject_domain_count_before_work(entrypoint, domain_count):
    config = BoundSearchParameters(parameter_domains=[(0, 1)] * domain_count)
    noise_model = Mock(side_effect=AssertionError("must not sample"))
    memory_generator = Mock(side_effect=AssertionError("must not generate circuits"))
    kwargs = {
        "noise_model": noise_model,
        "noise_parameters": [0.02, 0.04],
        "memory_generator": memory_generator,
    }
    if entrypoint == "finder":
        function = find_error_budget_bounds
        kwargs.update(num_rounds_per_distance={3: [3]}, search_parameters=config)
    else:
        function = get_error_budget
        kwargs.update(
            num_rounds_by_distances={3: [3]},
            noise_parameters_exploration_bounds=[(0.001, 0.1)] * 2,
            bound_search_parameters=config,
        )
    with pytest.raises(ValueError, match="parameter_domains.*2"):
        function(**kwargs)
    noise_model.assert_not_called()
    memory_generator.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["finder", "budget"])
def test_public_entrypoints_accept_matching_domain_count(entrypoint, monkeypatch):
    config = BoundSearchParameters(parameter_domains=[(0, 1)] * 2)
    gradient = Mock(return_value=(np.ones(2), np.ones(2)))
    monkeypatch.setattr(
        "deltakit_explorer.analysis.error_budget._budget.inverse_lambda_gradient_at",
        gradient,
    )
    if entrypoint == "finder":
        result = find_error_budget_bounds(
            Mock(), [0.02, 0.04], {3: [3]}, search_parameters=config
        )
        assert len(result.bounds) == 2
    else:
        result = get_error_budget(
            Mock(),
            [0.02, 0.04],
            {3: [3]},
            [(0.001, 0.1)] * 2,
            bound_search_parameters=config,
        )
        assert result.contributions == (0.02, 0.04)
        gradient.assert_called_once()
