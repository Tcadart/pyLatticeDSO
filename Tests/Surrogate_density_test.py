"""
Tests of the relative density surrogate (kriging) used by the optimization: exact gradient of the GPR mean,
dataset handling, and consistency of the committed models with the exact volume computed with Gmsh.

These tests require scikit-learn and SciPy (simulation dependencies) and are skipped otherwise.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyLatticeDesign.lattice import Lattice
from pyLatticeOpti.surrogate_model_relative_densities import (_valid_combinations, gp_mean_gradient_rbf_pipeline,
                                                              load_dataset, remove_large_volume_variations_dict,
                                                              save_dataset)

pytestmark = pytest.mark.simulation

project_root = Path(__file__).resolve().parents[1]
MODELS_DIR = project_root / "data" / "outputs" / "relative_densities" / "surrogate_model"


def _training_data(n: int = 40, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.01, 0.1, size=(n, d))
    y = np.pi * np.sum(x ** 2, axis=1) + 0.3 * x[:, 0] * x[:, -1]
    return x, y


def _finite_difference(predict, x0: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return np.array([(predict(x0 + h * e) - predict(x0 - h * e)) / (2 * h) for e in np.eye(x0.size)])


@pytest.mark.parametrize("kernel", [
    RBF(length_scale=[1.0, 1.0]),
    ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]),
    ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6),
])
@pytest.mark.parametrize("with_scaler, normalize_y", [(True, True), (False, False), (False, True)])
def test_gp_mean_gradient_matches_finite_differences(kernel, with_scaler, normalize_y):
    x, y = _training_data()
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, random_state=0)
    model = Pipeline([("x_scaler", StandardScaler()), ("gpr", gpr)]) if with_scaler else gpr
    model.fit(x, y)

    x0 = np.array([0.043, 0.067])
    gradient = gp_mean_gradient_rbf_pipeline(model, x0)
    # Quasi-interpolating GPR: small steps are dominated by round-off, the length scales are ~1 (scaled) or ~0.1
    fd = _finite_difference(lambda p: float(model.predict(p[None, :])[0]), x0, h=1e-4)
    np.testing.assert_allclose(gradient, fd, rtol=1e-4, atol=1e-8)

    # Batched evaluation
    batch = gp_mean_gradient_rbf_pipeline(model, np.vstack([x0, x0 + 0.01]))
    assert batch.shape == (2, 2)
    np.testing.assert_allclose(batch[0], gradient, rtol=1e-12)


def test_gp_mean_gradient_rejects_unsupported_models():
    from sklearn.gaussian_process.kernels import Matern
    x, y = _training_data()
    model = GaussianProcessRegressor(kernel=Matern()).fit(x, y)
    with pytest.raises(ValueError):
        gp_mean_gradient_rbf_pipeline(model, x[0])
    with pytest.raises(ValueError):
        gp_mean_gradient_rbf_pipeline(object(), x[0])


def test_committed_model_gradient_matches_finite_differences():
    model = joblib.load(MODELS_DIR / "kriging_model_BCC_Hybrid1_Hybrid4")["model"]
    x0 = np.array([0.045, 0.025, 0.035])
    gradient = gp_mean_gradient_rbf_pipeline(model, x0)
    fd = _finite_difference(lambda p: float(np.ravel(model.predict(p[None, :]))[0]), x0, h=1e-5)
    np.testing.assert_allclose(gradient, fd, rtol=1e-4, atol=1e-8)


# =============================================================================
# Dataset handling
# =============================================================================

def test_valid_combinations():
    combinations = _valid_combinations(0.01, (0.0, 0.03), 2)
    values = {0.0, 0.01, 0.02, 0.03, 0.001}
    assert all(sum(c) > 0.001 for c in combinations)
    assert all(math.isclose(v, min(values, key=lambda w: abs(w - v))) for c in combinations for v in c)
    # 5 values per radius, minus (0, 0), (0, 0.001) and (0.001, 0)
    assert len(combinations) == 5 ** 2 - 3


def test_remove_large_volume_variations():
    dataset = {(round(r, 2),): math.pi * r ** 2 for r in np.arange(0.01, 0.11, 0.01)}
    dataset[(0.05,)] = 0.9  # outlier
    filtered = remove_large_volume_variations_dict(dataset, distance_threshold=0.015, variation_threshold=0.1)
    assert (0.05,) not in filtered
    # Its neighbours are removed with it, the rest of the dataset is kept
    assert (0.04,) not in filtered and (0.06,) not in filtered
    assert len(filtered) == len(dataset) - 3


def test_save_and_load_dataset(tmp_path):
    dataset = {(0.01,): 0.002, (0.02,): 0.008, (0.03,): 0.7}
    save_dataset(tmp_path, "RelativeDensities_test", dataset)
    assert load_dataset(tmp_path, "RelativeDensities_test", min_vol=None, max_vol=None,
                        apply_variation_filter=False) == dataset
    # Default filter: volumes above 0.6 are removed
    assert (0.03,) not in load_dataset(tmp_path, "RelativeDensities_test", apply_variation_filter=False)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path, "missing")


# =============================================================================
# Consistency of the committed models with the exact volume
# =============================================================================

@pytest.mark.parametrize("geometries", [
    ["BCC"],
    ["BCC", "Hybrid1"],
    ["BCC", "Hybrid4"],
    ["Hybrid1"],
    ["Hybrid4"],
    ["Hybrid1", "Hybrid4"],
    ["BCC", "Hybrid1", "Hybrid4"],
])
def test_kriging_model_matches_gmsh_volume(tmp_path, geometries):
    pytest.importorskip("gmsh")
    radii = [0.05, 0.03, 0.04][:len(geometries)]
    config = {"geometry": {"cell_size": {"x": 1, "y": 1, "z": 1}, "number_of_cells": {"x": 1, "y": 1, "z": 1},
                           "radii": radii, "geom_types": geometries}}
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(config))
    cell = Lattice(str(path))
    exact = cell.generate_mesh_lattice_Gmsh(cut_mesh_at_boundary=True, save_STL=False, volume_computation=True,
                                            only_relative_density=True)

    model = joblib.load(MODELS_DIR / ("kriging_model_" + "_".join(geometries)))["model"]
    predicted = float(np.ravel(model.predict([radii]))[0])
    assert predicted == pytest.approx(exact, rel=5e-2)
