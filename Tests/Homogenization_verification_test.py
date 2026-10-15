"""
Verification of the beam homogenization module against closed-form solutions.

For a rigid-jointed periodic frame whose nodes are centers of symmetry of the cell (Cubic and BCC cells), the nodes
follow the macroscopic strain and do not rotate. Each beam is then a clamped-clamped Timoshenko beam with a relative
end displacement delta = l * eps . n and the homogenized stiffness has a closed form (see
examples/comparison_convergence/compare_homogenization_analytical.py). The beam model recovers it with a relative
error below 3e-4 for radii up to 0.2 (cell size 1).

These tests require the simulation dependencies (FEniCSx) and are skipped otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx_mpc")

from pyLatticeDesign.utils import open_lattice_parameters
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import get_homogenized_properties

pytestmark = pytest.mark.simulation

project_root = Path(__file__).resolve().parents[1]
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]  # order of the homogenization module
KAPPA = 0.9  # shear area coefficient of the beam model


def analytical_stiffness(geom_type: str, radius: float, cell_size: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Closed-form stiffness (engineering Voigt notation) of a periodic frame with affine, rotation-free nodes."""
    with open(project_root / "src" / "pyLatticeDesign" / "geometries" / f"{geom_type}.json", "r") as file:
        beams = np.array(json.load(file)["beams"], dtype=float)
    G = E / (2 * (1 + nu))
    A = np.pi * radius ** 2
    I = np.pi * radius ** 4 / 4
    volume = np.prod(cell_size)

    C = np.zeros((6, 6))
    for J, (k, l) in enumerate(VOIGT):
        eps = np.zeros((3, 3))
        eps[k, l] = eps[l, k] = 1.0 if k == l else 0.5
        sigma = np.zeros((3, 3))
        for beam in beams:
            branch = (beam[3:] - beam[:3]) * cell_size
            length = np.linalg.norm(branch)
            n = branch / length
            # Share of the beam inside the periodic cell: 1/2 on a face, 1/4 on an edge
            middle = 0.5 * (beam[:3] + beam[3:])
            weight = 0.5 ** np.sum(np.isclose(middle, 0.0) | np.isclose(middle, 1.0))
            delta = eps @ branch
            delta_axial = n @ delta
            phi = 12 * E * I / (KAPPA * G * A * length ** 2)
            k_transverse = 12 * E * I / (length ** 3 * (1 + phi))
            force = E * A / length * delta_axial * n + k_transverse * (delta - delta_axial * n)
            sigma += weight * np.outer(force, branch) / volume
        sigma = 0.5 * (sigma + sigma.T)
        C[:, J] = [sigma[i, j] for i, j in VOIGT]
    return 0.5 * (C + C.T)


@pytest.mark.parametrize("cell", ["Cubic", "BCC"])
def test_beam_homogenization_matches_closed_form(cell):
    radius = 0.1
    name_file = f"simulation/homogenization_{cell}"
    parameters = open_lattice_parameters(name_file)
    geometry = parameters["geometry"]
    cell_size = np.array([geometry["cell_size"][axis] for axis in "xyz"], dtype=float)
    material = parameters["simulation_parameters"]["material"]
    with open(project_root / "src" / "pyLatticeDesign" / "materials" / f"{material}.json", "r") as file:
        material_data = json.load(file)

    lattice = LatticeSim(name_file)
    lattice.reset_cell_with_new_radii([radius])
    lattice.reset_penalized_beams()
    _, analysis = get_homogenized_properties(lattice)

    C_beam = np.array(analysis.homogenizeMatrix, dtype=float)
    C_beam[:, 3:] /= 2.0  # tensor shear strains -> engineering Voigt notation
    C_beam = 0.5 * (C_beam + C_beam.T)
    C_ref = analytical_stiffness(geometry["geom_types"][0], radius, cell_size,
                                 material_data["Young_modulus"], material_data["Poisson_ratio"])

    error = np.linalg.norm(C_beam - C_ref, "fro") / np.linalg.norm(C_ref, "fro")
    assert error < 1e-3
