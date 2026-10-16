"""
Tests of the numerical building blocks of the simulation module that only depend on NumPy:
the thin-plate-spline RBF interpolator (surrogate of the reduced basis coefficients) and the
conjugate gradient solver (domain decomposition solver).
"""
import numpy as np
import pytest

from pyLatticeSim.conjugate_gradient_solver import conjugate_gradient_solver
from pyLatticeSim.utils_rbf import ThinPlateSplineRBF


@pytest.fixture
def training_set():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 0.1, size=(25, 2))
    y = np.column_stack([np.sin(30 * x[:, 0]) * x[:, 1], x[:, 0] ** 2 + 3 * x[:, 1]])
    return x, y


def test_rbf_interpolates_training_points(training_set):
    x, y = training_set
    rbf = ThinPlateSplineRBF(x, y)
    np.testing.assert_allclose(rbf.evaluate(x), y, rtol=1e-8, atol=1e-10)


def test_rbf_reproduces_linear_functions():
    """The linear polynomial tail makes the interpolant exact for affine data."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(12, 3))
    coefficients = np.array([2.0, -1.0, 0.5])
    y = x @ coefficients + 0.3
    rbf = ThinPlateSplineRBF(x, y)

    x_query = rng.uniform(0.0, 1.0, size=(5, 3))
    np.testing.assert_allclose(rbf.evaluate(x_query).ravel(), x_query @ coefficients + 0.3, atol=1e-10)
    np.testing.assert_allclose(rbf.gradient(x_query[0]).ravel(), coefficients, atol=1e-8)


def test_rbf_output_shapes(training_set):
    x, y = training_set
    rbf = ThinPlateSplineRBF(x, y)
    assert rbf.evaluate(x[0]).shape == (2,)
    assert rbf.evaluate(x[:4]).shape == (4, 2)
    assert rbf.gradient(x[0]).shape == (2, 2)
    assert rbf.gradient(x[:4]).shape == (4, 2, 2)

    rbf_scalar = ThinPlateSplineRBF(x, y[:, 0])
    assert rbf_scalar.evaluate(x[:3]).shape == (3, 1)


def test_rbf_gradient_matches_finite_differences(training_set):
    x, y = training_set
    rbf = ThinPlateSplineRBF(x, y)
    x0 = np.array([0.043, 0.071])
    h = 1e-6
    fd = np.array([(rbf.evaluate(x0 + h * e) - rbf.evaluate(x0 - h * e)) / (2 * h) for e in np.eye(2)])
    np.testing.assert_allclose(rbf.gradient(x0), fd, rtol=1e-5, atol=1e-7)

    # Batched gradient is consistent with the single-point one
    batch = rbf.gradient(np.vstack([x0, x0 + 0.01]))
    np.testing.assert_allclose(batch[0], rbf.gradient(x0), rtol=1e-12)


def test_rbf_gradient_at_training_point_is_finite(training_set):
    """r = 0 at a center: the kernel derivative must be taken as zero, not NaN."""
    x, y = training_set
    rbf = ThinPlateSplineRBF(x, y)
    assert np.all(np.isfinite(rbf.gradient(x[3])))


def test_rbf_regularization_smooths_data(training_set):
    x, y = training_set
    rbf = ThinPlateSplineRBF(x, y, reg=1e-2)
    residual = np.linalg.norm(rbf.evaluate(x) - y)
    assert residual > 1e-6  # no longer interpolating
    assert residual < np.linalg.norm(y)


def _spd_matrix(n: int, condition: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return q @ np.diag(np.geomspace(1.0, condition, n)) @ q.T


def test_conjugate_gradient_solves_spd_system():
    A = _spd_matrix(40, 1e3)
    b = np.random.default_rng(2).normal(size=40)
    x, info = conjugate_gradient_solver(A, b, maxiter=500, tol=1e-10, mintol=1e-14)
    assert info == 0
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-7, atol=1e-9)


def test_conjugate_gradient_with_preconditioner():
    A = _spd_matrix(40, 1e4, seed=3)
    b = np.random.default_rng(4).normal(size=40)
    iterations = {}
    for name, M in [("none", None), ("exact", np.linalg.inv(A))]:
        count = []
        x, info = conjugate_gradient_solver(A, b, M=M, maxiter=500, tol=1e-10, mintol=1e-14,
                                            callback=lambda xk: count.append(1))
        assert info == 0
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-6, atol=1e-8)
        iterations[name] = len(count)
    # With the inverse as preconditioner, CG converges in one iteration
    assert iterations["exact"] == 1
    assert iterations["none"] > iterations["exact"]


def test_conjugate_gradient_reports_non_convergence():
    A = _spd_matrix(30, 1e6, seed=5)
    b = np.ones(30)
    _, info = conjugate_gradient_solver(A, b, maxiter=2, tol=1e-12, mintol=1e-16)
    assert info != 0


def test_conjugate_gradient_accepts_linear_operator():
    scipy_sparse_linalg = pytest.importorskip("scipy.sparse.linalg")
    A = _spd_matrix(20, 1e2, seed=6)
    b = np.arange(20, dtype=float)
    operator = scipy_sparse_linalg.LinearOperator(A.shape, matvec=lambda v: A @ v)
    x, info = conjugate_gradient_solver(operator, b, maxiter=200, tol=1e-12, mintol=1e-16)
    assert info == 0
    np.testing.assert_allclose(A @ x, b, atol=1e-8)
