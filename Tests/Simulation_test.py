"""
Verification of the beam simulations on small lattices: full-scale FEM, domain decomposition (DDM) with exact and
surrogate Schur complements, and properties of the Schur complements and of the reduced basis.

These tests require the simulation dependencies (FEniCSx) and are skipped otherwise.
"""
import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("scipy")

from pyLatticeSim.greedy_algorithm import project_to_reduced_basis, reduce_basis_greedy
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_schur import get_schur_complement, load_schur_complement_dataset
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

pytestmark = pytest.mark.simulation

LOAD = -0.01  # total force on the loaded face


@pytest.fixture
def lattice_file(tmp_path):
    """Write the configuration of a BCC cantilever (fixed on Xmin, loaded along Z on Xmax)."""
    def _write(number_of_cells=(3, 1, 1), radius=0.05, load=LOAD, periodicity=False, schur=None,
               geom_types=("BCC",)):
        config = {
            "geometry": {
                "cell_size": {"x": 1, "y": 1, "z": 1},
                "number_of_cells": dict(zip("xyz", number_of_cells)),
                "radii": [radius] * len(geom_types),
                "geom_types": list(geom_types),
            },
            "simulation_parameters": {"enable": True, "material": "VeroClear", "periodicity": periodicity},
            "boundary_conditions": {
                "Displacement": {"Fixed": {"Surface": ["Xmin"], "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
                                           "Value": [0, 0, 0, 0, 0, 0]}},
                "Force": {"Load": {"Surface": ["Xmax"], "DOF": ["Z"], "Value": [load]}},
            },
        }
        if schur is not None:
            config["simulation_parameters"]["DDM"] = {
                "enable_preconditioner": False,
                "max_iterations": 5000,
                "schur_complement_computation": {"type": schur, "precision_greedy": 1e-6},
            }
        path = tmp_path / f"lattice_{schur}_{load}_{periodicity}_{len(geom_types)}.json"
        path.write_text(json.dumps(config))
        return str(path)
    return _write


def _boundary_displacements(lattice) -> dict:
    """Displacement vector of each node on the boundary of the cells, keyed by its coordinates."""
    return {(round(n.x, 6), round(n.y, 6), round(n.z, 6)): np.array(n.displacement_vector, dtype=float)
            for n in lattice.nodes if n.index_boundary is not None}


def _relative_difference(displacements_a: dict, displacements_b: dict) -> float:
    assert displacements_a.keys() == displacements_b.keys()
    keys = sorted(displacements_a)
    a = np.array([displacements_a[k] for k in keys])
    b = np.array([displacements_b[k] for k in keys])
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _tip_deflection(lattice, x_tip: float) -> float:
    return float(np.mean([n.displacement_vector[2] for n in lattice.nodes if np.isclose(n.x, x_tip)]))


@pytest.fixture
def fem_cantilever(lattice_file):
    lattice = LatticeSim(lattice_file())
    solve_FEM_FenicsX(lattice)
    return lattice


# =============================================================================
# Full-scale FEM
# =============================================================================

def test_fem_boundary_conditions_and_equilibrium(fem_cantilever):
    lattice = fem_cantilever
    fixed = [n for n in lattice.nodes if np.isclose(n.x, 0.0)]
    loaded = [n for n in lattice.nodes if np.isclose(n.x, 3.0)]
    assert len(fixed) == 4 and len(loaded) == 4

    # Imposed displacements are respected
    assert max(np.abs(n.displacement_vector).max() for n in fixed) == 0.0
    # The applied load is distributed over the loaded face
    assert sum(n.applied_force[2] for n in lattice.nodes) == pytest.approx(LOAD)
    # Global equilibrium: the reactions balance the applied load
    reactions = np.sum([n.reaction_force_vector[:3] for n in fixed], axis=0)
    np.testing.assert_allclose(reactions, [0.0, 0.0, -LOAD], atol=1e-8)
    # The tip moves in the direction of the load
    assert _tip_deflection(lattice, 3.0) < 0.0


def test_fem_is_linear_in_the_load(fem_cantilever, lattice_file):
    lattice = LatticeSim(lattice_file(load=2 * LOAD))
    solve_FEM_FenicsX(lattice)
    assert _tip_deflection(lattice, 3.0) == pytest.approx(2 * _tip_deflection(fem_cantilever, 3.0), rel=1e-8)


def test_fem_stiffness_increases_with_radius(fem_cantilever, lattice_file):
    lattice = LatticeSim(lattice_file(radius=0.07))
    solve_FEM_FenicsX(lattice)
    assert abs(_tip_deflection(lattice, 3.0)) < abs(_tip_deflection(fem_cantilever, 3.0))


def test_fem_compliance_is_positive(fem_cantilever):
    """The external work f.u of the applied loads is twice the strain energy: strictly positive."""
    compliance = sum(n.applied_force[k] * n.displacement_vector[k] for n in fem_cantilever.nodes for k in range(6))
    assert compliance > 0.0


# =============================================================================
# Domain decomposition
# =============================================================================

def test_ddm_exact_matches_fem(fem_cantilever, lattice_file):
    """
    Static condensation is exact: DDM with exact Schur complements reproduces the full-scale FEM.
    Without periodicity, boundary cells differ from interior cells (penalized zones at the nodes), so each
    cell must get its own Schur complement.
    """
    lattice = LatticeSim(lattice_file(schur="exact"), enable_domain_decomposition_solver=True)
    _, info, _, _ = lattice.solve_DDM()
    assert info == 0
    assert _relative_difference(_boundary_displacements(lattice), _boundary_displacements(fem_cantilever)) < 1e-6


def test_ddm_equilibrium(lattice_file):
    lattice = LatticeSim(lattice_file(schur="RBF", periodicity=True), enable_domain_decomposition_solver=True)
    lattice.solve_DDM()
    fixed = [n for n in lattice.nodes if np.isclose(n.x, 0.0)]
    reactions = np.sum([n.reaction_force_vector[:3] for n in fixed], axis=0)
    np.testing.assert_allclose(reactions, [0.0, 0.0, -LOAD], atol=1e-6 * abs(LOAD) * 1e3)
    assert max(np.abs(n.displacement_vector).max() for n in fixed) == 0.0


@pytest.mark.parametrize("surrogate", ["RBF", "linear", "nearest_neighbor"])
def test_ddm_surrogate_matches_exact_on_training_radius(lattice_file, surrogate):
    """With a radius of the training set, the reduced basis reproduces the exact Schur complement."""
    exact = LatticeSim(lattice_file(schur="exact", periodicity=True), enable_domain_decomposition_solver=True)
    exact.solve_DDM()
    approx = LatticeSim(lattice_file(schur=surrogate, periodicity=True), enable_domain_decomposition_solver=True)
    approx.solve_DDM()
    assert _relative_difference(_boundary_displacements(approx), _boundary_displacements(exact)) < 1e-5


def test_ddm_rbf_between_training_radii(lattice_file):
    exact = LatticeSim(lattice_file(schur="exact", periodicity=True, radius=0.055),
                       enable_domain_decomposition_solver=True)
    exact.solve_DDM()
    approx = LatticeSim(lattice_file(schur="RBF", periodicity=True, radius=0.055),
                        enable_domain_decomposition_solver=True)
    approx.solve_DDM()
    assert _relative_difference(_boundary_displacements(approx), _boundary_displacements(exact)) < 1e-2


def test_ddm_cells_share_schur_complement_only_when_identical(lattice_file):
    periodic = LatticeSim(lattice_file(schur="exact", periodicity=True), enable_domain_decomposition_solver=True)
    assert all(cell.schur_complement is periodic.cells[0].schur_complement for cell in periodic.cells)

    lattice = LatticeSim(lattice_file(schur="exact"), enable_domain_decomposition_solver=True)
    for i, cell in enumerate(lattice.cells):
        np.testing.assert_allclose(cell.schur_complement, get_schur_complement(lattice, i), rtol=1e-10, atol=1e-12)
    # The two end cells are mirror images of each other but differ from the middle cell
    assert not np.allclose(lattice.cells[0].schur_complement, lattice.cells[1].schur_complement)


# =============================================================================
# Schur complement and reduced basis
# =============================================================================

@pytest.fixture
def single_cell(lattice_file):
    return LatticeSim(lattice_file(number_of_cells=(1, 1, 1), schur="exact", periodicity=True),
                      enable_domain_decomposition_solver=True)


def test_schur_complement_properties(single_cell):
    S = get_schur_complement(single_cell)
    assert S.shape == (48, 48)  # 8 boundary nodes x 6 DOFs for a BCC cell
    np.testing.assert_allclose(S, S.T, atol=1e-12 * np.abs(S).max())
    eigenvalues = np.linalg.eigvalsh(0.5 * (S + S.T))
    scale = eigenvalues[-1]
    # Positive semi-definite with exactly the 6 rigid body modes in its kernel
    assert eigenvalues[0] > -1e-10 * scale
    assert np.sum(eigenvalues < 1e-10 * scale) == 6
    assert eigenvalues[6] > 1e-6 * scale


def test_schur_complement_rigid_body_modes(single_cell):
    cell = single_cell.cells[0]
    S = get_schur_complement(single_cell)
    nodes = cell.node_in_order_simulation
    center = np.mean([[n.x, n.y, n.z] for n in nodes], axis=0)
    for k in range(3):
        translation = np.zeros(6 * len(nodes))
        translation[k::6] = 1.0
        np.testing.assert_allclose(S @ translation, 0.0, atol=1e-10 * np.abs(S).max())
        # Infinitesimal rotation around axis k: u = omega x (x - c), theta = omega
        omega = np.eye(3)[k]
        rotation = np.zeros(6 * len(nodes))
        for i, n in enumerate(nodes):
            rotation[6 * i:6 * i + 3] = np.cross(omega, np.array([n.x, n.y, n.z]) - center)
            rotation[6 * i + 3:6 * i + 6] = omega
        np.testing.assert_allclose(S @ rotation, 0.0, atol=1e-9 * np.abs(S).max())


def test_schur_complement_dataset_matches_exact(single_cell):
    dataset = load_schur_complement_dataset(single_cell)
    assert (0.05,) in dataset
    S = get_schur_complement(single_cell)
    np.testing.assert_allclose(dataset[(0.05,)], S, rtol=1e-8, atol=1e-10 * np.abs(S).max())


def test_reduced_basis_greedy_reconstruction(single_cell):
    dataset = load_schur_complement_dataset(single_cell)
    residuals = []
    _, _, _, basis, alpha, _, _ = reduce_basis_greedy(dataset, 1e-6, verbose=0, residual_history=residuals)
    assert basis.shape[0] == 48 * 48
    assert basis.shape[1] <= len(dataset)
    # Orthonormal basis and decreasing residual
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)
    assert residuals[-1] < 1e-6 and np.all(np.diff(residuals) <= 1e-12)
    # Every snapshot is reconstructed from its coefficients
    for i, S in enumerate(dataset.values()):
        reconstruction = (basis @ alpha[:, i]).reshape(S.shape, order="F")
        assert np.linalg.norm(reconstruction - S) / np.linalg.norm(S) < 1e-5

    # Projection of new matrices in the basis
    alphas = project_to_reduced_basis({key: S.T for key, S in dataset.items()}, basis)
    for key, S in dataset.items():
        reconstruction = (basis @ alphas[key]).reshape(S.shape, order="F")
        assert np.linalg.norm(reconstruction - S) / np.linalg.norm(S) < 1e-5


def test_reduced_basis_tolerance_controls_size(single_cell):
    dataset = load_schur_complement_dataset(single_cell)
    sizes = [reduce_basis_greedy(dataset, tol, verbose=0)[3].shape[1] for tol in (1e-1, 1e-3, 1e-6)]
    assert sizes[0] <= sizes[1] <= sizes[2]
    assert sizes[0] < sizes[2]


def test_schur_gradient_rbf_matches_finite_differences(lattice_file):
    lattice = LatticeSim(lattice_file(number_of_cells=(1, 1, 1), schur="RBF", periodicity=True),
                         enable_domain_decomposition_solver=True)
    radius = [0.055]
    gradient = lattice._compute_schur_gradients_RBF_batch([radius])[0][0]
    h = 1e-6
    fd = (lattice.get_schur_complement_from_reduced_basis([radius[0] + h])
          - lattice.get_schur_complement_from_reduced_basis([radius[0] - h])) / (2 * h)
    assert np.linalg.norm(gradient - fd) / np.linalg.norm(fd) < 1e-5

    # Reference: derivative of the exact Schur complement, each cell built from scratch so that the penalized
    # zones at the nodes follow the radius
    def exact_schur(r):
        cell = LatticeSim(lattice_file(number_of_cells=(1, 1, 1), radius=r, schur="exact", periodicity=True),
                          enable_domain_decomposition_solver=True)
        return get_schur_complement(cell)

    h = 1e-5
    exact_gradient = (exact_schur(radius[0] + h) - exact_schur(radius[0] - h)) / (2 * h)
    assert np.linalg.norm(gradient - exact_gradient) / np.linalg.norm(exact_gradient) < 1e-3


@pytest.mark.xfail(strict=True, reason=(
    "In 'exact' mode, _schur_for_params changes the radius with cell.change_beam_radius, which keeps the "
    "penalized zones of the initial radius: dS/dr is wrong by 3-9%"))
def test_schur_gradient_exact_mode_follows_penalization(lattice_file):
    def build(r):
        return LatticeSim(lattice_file(number_of_cells=(1, 1, 1), radius=r, schur="exact", periodicity=True),
                          enable_domain_decomposition_solver=True)

    r, h = 0.055, 1e-5
    reference = (get_schur_complement(build(r + h)) - get_schur_complement(build(r - h))) / (2 * h)
    lattice = build(r)
    gradient = lattice._compute_schur_gradients(lattice.cells[0], [r])[0]
    assert np.linalg.norm(gradient - reference) / np.linalg.norm(reference) < 1e-3


def test_hybrid_cell_schur_from_reduced_basis(lattice_file):
    lattice = LatticeSim(lattice_file(number_of_cells=(1, 1, 1), schur="RBF", periodicity=True,
                                      geom_types=("BCC", "Hybrid1", "Hybrid4")),
                         enable_domain_decomposition_solver=True)
    S = lattice.get_schur_complement_from_reduced_basis([0.05, 0.03, 0.04])
    lattice.cells[0].define_node_order_to_simulate()
    n_dofs = 6 * len(lattice.cells[0].node_in_order_simulation)
    assert S.shape == (n_dofs, n_dofs)
    np.testing.assert_allclose(S, S.T, atol=1e-8 * np.abs(S).max())
    # Batched and single evaluations agree
    batch = lattice.get_schur_complement_from_reduced_basis_batch([[0.05, 0.03, 0.04], [0.02, 0.02, 0.02]])
    np.testing.assert_allclose(batch[0], S, rtol=1e-12, atol=1e-14)
