"""
Verification of the lattice optimization module: the analytical (adjoint) gradients of the objectives and the
kriging gradient of the relative density constraint are compared with finite differences, for every
parameterization, then a short optimization is run.

The domain decomposition solver with the RBF surrogate of the Schur complements is used (datasets of
data/outputs/schur_complement and data/outputs/relative_densities).
These tests require the simulation dependencies (FEniCSx) and are skipped otherwise.
"""
import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("sklearn")

from pyLatticeOpti.lattice_opti import LatticeOpti

pytestmark = pytest.mark.simulation


@pytest.fixture
def make_opti(tmp_path, monkeypatch):
    """Build a LatticeOpti on a 3x1x1 cantilever (fixed on Xmin, loaded along Z on Xmax)."""
    def _make(objective_type="compliance", parameterization="unit_cell", hybrid=False, objective_function="min",
              density_gradient=True, density_target=0.05, max_iterations=3, normalization=True):
        geom_types = ["BCC", "Hybrid1", "Hybrid4"] if hybrid else ["BCC"]
        optimization_parameters = {"type": parameterization, "hybrid": hybrid}
        if parameterization == "linear":
            optimization_parameters["direction"] = ["x"]
        config = {
            "geometry": {
                "cell_size": {"x": 1, "y": 1, "z": 1},
                "number_of_cells": {"x": 3, "y": 1, "z": 1},
                "radii": [0.05] * len(geom_types),
                "geom_types": geom_types,
            },
            "simulation_parameters": {
                "enable": True, "material": "VeroClear", "periodicity": True,
                "DDM": {"enable_preconditioner": False, "max_iterations": 2000,
                        "schur_complement_computation": {"type": "RBF", "precision_greedy": 1e-6}},
            },
            "boundary_conditions": {
                "Displacement": {"Fixed": {"Surface": ["Xmin"], "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
                                           "Value": [0, 0, 0, 0, 0, 0]}},
                "Force": {"Load": {"Surface": ["Xmax"], "DOF": ["Z"], "Value": [-0.01]}},
            },
            "optimization_informations": {
                "objective_function": objective_function,
                "objective_type": objective_type,
                "objective_data": {"Surface": ["Xmax"], "DOF": ["Z"]},
                "max_iterations": max_iterations,
                "optimization_parameters": optimization_parameters,
                "constraints": {"relative_density": {"value": density_target, "compute_gradient": density_gradient}},
                "enable_parameter_normalization": normalization,
                "simulation_type": "DDM",
                "enable_gradient_computing": True,
            },
        }
        path = tmp_path / "optimization.json"
        path.write_text(json.dumps(config))
        lattice = LatticeOpti(str(path))
        lattice.redefine_optim_parameters(disp=False)
        lattice._initialize_optimization_solver()
        lattice._add_constraint_density()
        return lattice
    return _make


def _relative_error(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def _parameters(lattice) -> list[float]:
    """Point of evaluation inside the bounds, with distinct values for each parameter."""
    if lattice.optimization_parameters["type"] == "linear":
        return [0.2, 0.4]
    return list(np.linspace(0.3, 0.6, lattice.number_parameters))


# =============================================================================
# Objective gradients
# =============================================================================

@pytest.mark.parametrize("objective_type, objective_function", [
    ("compliance", "min"),
    ("displacement", "max"),
    ("displacement_ratio", "max"),
])
@pytest.mark.parametrize("parameterization, hybrid", [
    ("unit_cell", False),
    ("constant", False),
    ("linear", False),
    ("unit_cell", True),
    ("constant", True),
])
def test_objective_gradient_matches_finite_differences(make_opti, objective_type, objective_function,
                                                       parameterization, hybrid):
    lattice = make_opti(objective_type=objective_type, objective_function=objective_function,
                        parameterization=parameterization, hybrid=hybrid)
    r = _parameters(lattice)
    lattice.objective(r)  # sets the normalization scale
    gradient = lattice.gradient(r)
    fd = lattice.finite_difference_gradient(r, eps=1e-4)
    assert gradient.shape == (lattice.number_parameters,)
    # The tolerance is set by the conjugate gradient tolerance of the DDM solver
    assert _relative_error(gradient, fd) < 1e-3


def test_compliance_decreases_with_radius(make_opti):
    lattice = make_opti(parameterization="constant")
    lattice.objective([0.4])  # the optimizer always evaluates the objective (normalization scale) first
    assert lattice.gradient([0.4])[0] < 0.0
    assert lattice.objective([0.6]) < lattice.objective([0.4])


def test_gradient_requires_normalization_scale(make_opti):
    lattice = make_opti(parameterization="constant")
    with pytest.raises(RuntimeError):
        lattice.gradient([0.4])


def test_gradient_sign_follows_objective_function(make_opti):
    r = [0.4]
    g_min = make_opti(objective_type="displacement", parameterization="constant", objective_function="min")
    g_max = make_opti(objective_type="displacement", parameterization="constant", objective_function="max")
    for lattice in (g_min, g_max):
        lattice.objective(r)
    assert g_min.objective(r) == pytest.approx(-g_max.objective(r))
    np.testing.assert_allclose(g_min.gradient(r), -g_max.gradient(r), rtol=1e-10)


def test_gradient_without_normalization(make_opti):
    lattice = make_opti(parameterization="constant", normalization=False)
    r = [0.06]
    lattice.objective(r)
    gradient = lattice.gradient(r)
    fd = lattice.finite_difference_gradient(r, eps=1e-5)
    assert _relative_error(gradient, fd) < 1e-3


# =============================================================================
# Relative density constraint
# =============================================================================

@pytest.mark.parametrize("parameterization, hybrid", [
    ("unit_cell", False),
    ("constant", False),
    ("unit_cell", True),
    ("constant", True),
])
def test_density_gradient_kriging_matches_finite_differences(make_opti, parameterization, hybrid):
    lattice = make_opti(parameterization=parameterization, hybrid=hybrid)
    r = _parameters(lattice)
    gradient = lattice.density_constraint_gradient(r)
    fd = lattice.finite_difference_density_gradient(r, eps=1e-4)
    assert _relative_error(gradient, fd) < 1e-4
    # A larger radius always increases the density
    assert np.all(gradient > 0.0)


def test_density_constraint_value(make_opti):
    lattice = make_opti(parameterization="constant", density_target=0.05)
    r = [0.4]
    lattice.set_optimization_parameters(r)
    assert lattice.density_constraint(r) == pytest.approx(lattice.get_relative_density() - 0.05)


def test_kriging_density_close_to_geometric_density(make_opti):
    """The surrogate is trained on the exact volume (overlaps removed): below the sum of the beam volumes."""
    lattice = make_opti(parameterization="constant", hybrid=True)
    lattice.set_optimization_parameters([0.4, 0.3, 0.2])
    kriging = lattice.get_relative_density()
    beams = np.mean([cell.relative_density for cell in lattice.cells])
    assert 0.8 * beams < kriging < beams


@pytest.mark.parametrize("mode, sign", [("upper", 1.0), ("lower", -1.0)])
def test_density_constraint_modes(make_opti, mode, sign):
    lattice = make_opti(parameterization="constant")
    lattice.constraints_dict["relative_density"]["mode"] = mode
    lattice.constraints = []
    lattice._add_constraint_density()
    constraint = lattice.constraints[-1]
    r = [0.4]
    assert constraint.fun(r) == pytest.approx(sign * lattice.density_constraint(r))
    np.testing.assert_allclose(constraint.jac(r), sign * lattice.density_constraint_gradient(r))


def test_density_constraint_band_requires_tolerance(make_opti):
    lattice = make_opti(parameterization="constant")
    lattice.constraints_dict["relative_density"]["mode"] = "band"
    with pytest.raises(ValueError):
        lattice._add_constraint_density()


# =============================================================================
# Parameterization and normalization
# =============================================================================

def test_normalization_round_trip(make_opti):
    lattice = make_opti(parameterization="unit_cell")
    radii = [lattice.min_radius, 0.05, lattice.max_radius]
    normalized = lattice.normalize_optimization_parameters(radii)
    np.testing.assert_allclose(normalized, [0.0, (0.05 - 0.01) / 0.09, 1.0])
    np.testing.assert_allclose(lattice.denormalize_optimization_parameters(normalized), radii)
    with pytest.raises(ValueError):
        lattice.normalize_optimization_parameters([0.5])


def test_set_optimization_parameters_unit_cell(make_opti):
    lattice = make_opti(parameterization="unit_cell")
    assert lattice.number_parameters == 3
    lattice.set_optimization_parameters([0.0, 0.5, 1.0])
    np.testing.assert_allclose([cell.radii[0] for cell in sorted(lattice.cells, key=lambda c: c.index)],
                               [0.01, 0.055, 0.1])
    with pytest.raises(ValueError):
        lattice.set_optimization_parameters([0.5])


def test_set_optimization_parameters_linear(make_opti):
    lattice = make_opti(parameterization="linear")
    assert lattice.number_parameters == 2
    slope, intercept = 0.3, 0.2
    lattice.set_optimization_parameters([slope, intercept])
    for cell in lattice.cells:
        x_normalized = cell.center_point[0] / lattice.size_x
        expected = 0.01 + 0.09 * intercept + 0.09 * slope * x_normalized
        assert cell.radii[0] == pytest.approx(expected)


def test_set_optimization_parameters_hybrid_constant(make_opti):
    lattice = make_opti(parameterization="constant", hybrid=True)
    assert lattice.number_parameters == 3
    lattice.set_optimization_parameters([0.0, 0.5, 1.0])
    for cell in lattice.cells:
        np.testing.assert_allclose(cell.radii, [0.01, 0.055, 0.1])


def test_objective_normalized_by_initial_value(make_opti):
    lattice = make_opti(parameterization="constant")
    assert lattice.objective([0.4]) == pytest.approx(1.0)
    assert lattice.denorm_objective == pytest.approx(lattice.initial_value_objective)
    assert lattice.compute_compliance() > 0.0


# =============================================================================
# Optimization run
# =============================================================================

def test_optimization_improves_objective_and_respects_density(make_opti, tmp_path):
    lattice = make_opti(parameterization="unit_cell", density_target=0.05, max_iterations=15)
    initial_parameters = list(lattice.initial_parameters)
    initial_density = lattice.density_constraint(initial_parameters) + 0.05
    lattice.optimize_lattice()
    history = lattice._history
    assert len(history["iteration"]) >= 2

    # The initial design is feasible (density below the target): the optimum uses the available material
    assert initial_density < 0.05
    assert lattice.get_relative_density() == pytest.approx(0.05, abs=1e-4)
    assert lattice.actual_objective < 1.0  # normalized by the initial compliance
    # More material near the clamped end of the cantilever
    radii = [cell.radii[0] for cell in sorted(lattice.cells, key=lambda c: c.index)]
    assert radii[0] > radii[-1]

    lattice.save_optimization_json(str(tmp_path / "summary.json"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["solution"]["final_parameters"] == pytest.approx(list(lattice.solution.x))
    assert len(summary["history"]["objective"]) == len(history["objective"])
