"""
Tests of the lattice design features: geometric transformations, symmetry, surface selection, radius updates,
beam deletion, save/load and volume computation with Gmsh.
"""
import json
import math

import numpy as np
import pytest

from pyLatticeDesign import design_transformation
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.utils import (PENALIZATION_STRATEGIES, function_penalization_Lzone,
                                   get_boundary_condition_color, penalization_length, save_lattice_object)


@pytest.fixture
def make_lattice(tmp_path):
    """Build a lattice from a minimal JSON configuration written in a temporary directory."""
    def _make(number_of_cells=(2, 2, 2), radii=(0.05,), geom_types=("BCC",), supplementary=None):
        config = {
            "geometry": {
                "cell_size": {"x": 1, "y": 1, "z": 1},
                "number_of_cells": dict(zip("xyz", number_of_cells)),
                "radii": list(radii),
                "geom_types": list(geom_types),
            }
        }
        if supplementary is not None:
            config["supplementary"] = supplementary
        path = tmp_path / "lattice.json"
        path.write_text(json.dumps(config))
        return Lattice(str(path))
    return _make


def _bcc_relative_density(radius: float) -> float:
    """Relative density of a BCC cell of size 1 without overlap correction: 8 half diagonals."""
    return 8 * math.pi * radius ** 2 * round(math.sqrt(3) / 2, 4)


# =============================================================================
# Radius updates and relative density
# =============================================================================

def test_relative_density_bcc(make_lattice):
    lattice = make_lattice()
    assert lattice.get_relative_density() == pytest.approx(_bcc_relative_density(0.05), rel=1e-12)


def test_relative_density_follows_lattice_radius_change(make_lattice):
    lattice = make_lattice()
    lattice.change_beam_radius([0.08])
    assert lattice.get_beam_radius_min_max() == (0.08, 0.08)
    assert lattice.get_relative_density() == pytest.approx(_bcc_relative_density(0.08), rel=1e-12)


def test_relative_density_follows_cell_radius_change(make_lattice):
    lattice = make_lattice(number_of_cells=(2, 1, 1))
    cell = lattice.cells[0]
    cell.change_beam_radius([0.02])
    assert cell.relative_density == pytest.approx(_bcc_relative_density(0.02), rel=1e-12)
    assert cell.volume_each_geom[0] == pytest.approx(_bcc_relative_density(0.02), rel=1e-12)
    assert lattice.get_relative_density() == pytest.approx(
        0.5 * (_bcc_relative_density(0.02) + _bcc_relative_density(0.05)), rel=1e-12)


def test_delete_beams_under_radius_threshold(make_lattice):
    lattice = make_lattice(number_of_cells=(2, 1, 1))
    n_beams = lattice.get_number_beams()
    lattice.cells[0].change_beam_radius([0.005])
    lattice.delete_beams_under_radius_threshold(0.01)
    assert lattice.get_number_beams() == n_beams // 2
    assert all(beam.radius > 0.01 for beam in lattice.beams)


# =============================================================================
# Topology queries
# =============================================================================

def test_find_point_on_lattice_surface(make_lattice):
    lattice = make_lattice()
    for surface in ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"]:
        points = lattice.find_point_on_lattice_surface([surface])
        assert len(points) == 9  # 3 x 3 corner nodes on each face of a 2x2x2 BCC lattice
        axis = "xyz".index(surface[0].lower())
        value = 0.0 if surface.endswith("min") else 2.0
        assert all(math.isclose([p.x, p.y, p.z][axis], value) for p in points)
    # Several surfaces: intersection (edge of the lattice)
    edge = lattice.find_point_on_lattice_surface(["Xmin", "Zmax"])
    assert len(edge) == 3
    assert all(math.isclose(p.x, 0.0) and math.isclose(p.z, 2.0) for p in edge)


def test_cells_queries(make_lattice):
    lattice = make_lattice(number_of_cells=(3, 2, 1))
    assert len(lattice.get_cells_on_surfaces(["Xmin"])) == 2
    assert len(lattice.get_cells_at_index("x", 0)) == 2
    assert len(lattice.get_cells_at_index("y", 1)) == 3
    assert np.shape(lattice.get_cell_occupancy_matrix()) == (3, 2, 1)
    assert lattice.find_minimum_beam_length() == pytest.approx(math.sqrt(3) / 2, abs=1e-4)
    assert lattice.get_name_lattice() == "BCC"


def test_hybrid_cell_beams(make_lattice):
    lattice = make_lattice(number_of_cells=(1, 1, 1), radii=(0.05, 0.03), geom_types=("BCC", "Octet"))
    types = {beam.type_beam for beam in lattice.beams}
    assert types == {0, 1}
    radii = {beam.type_beam: beam.radius for beam in lattice.beams}
    assert radii == {0: 0.05, 1: 0.03}


# =============================================================================
# Symmetry
# =============================================================================

@pytest.mark.parametrize("plane, reference, expected_box, expected_nodes", [
    ("YZ", {"x": 0.0}, [-2.0, 2.0, 0.0, 1.0, 0.0, 1.0], 24),
    ("X", {"x": 2.0}, [0.0, 4.0, 0.0, 1.0, 0.0, 1.0], 24),
    ("XY", {"z": 0.0}, [0.0, 2.0, 0.0, 1.0, -1.0, 1.0], 22),
])
def test_symmetry(make_lattice, plane, reference, expected_box, expected_nodes):
    lattice = make_lattice(number_of_cells=(2, 1, 1),
                           supplementary={"symmetries": {"plane": plane, "reference_point": reference}})
    assert lattice.get_number_cells() == 4
    assert lattice.get_number_beams() == 32
    # Nodes on the symmetry plane are shared by the original and the mirrored cells
    assert lattice.get_number_nodes() == expected_nodes
    np.testing.assert_allclose(lattice.get_lattice_boundary_box(), expected_box, atol=1e-12)
    assert lattice.get_relative_density() == pytest.approx(_bcc_relative_density(0.05), rel=1e-12)


@pytest.mark.parametrize("plane, reference, axis", [("YZ", {"x": 0.0}, 0), ("X", {"x": 2.0}, 0),
                                                ("XZ", {"y": 0.0}, 1), ("XY", {"z": 0.0}, 2)])
def test_symmetry_grid_positions(make_lattice, plane, reference, axis):
    lattice = make_lattice(number_of_cells=(2, 1, 1),
                           supplementary={"symmetries": {"plane": plane, "reference_point": reference}})
    positions = [tuple(cell.pos) for cell in lattice.cells]
    assert len(set(positions)) == len(positions)  # one cell per grid position
    expected = [2, 1, 1]
    expected[axis] *= 2
    assert [lattice.num_cells_x, lattice.num_cells_y, lattice.num_cells_z] == expected
    assert [lattice.size_x, lattice.size_y, lattice.size_z] == pytest.approx(expected)
    # Grid positions follow the coordinates
    for cell in lattice.cells:
        assert cell.pos[axis] == round(cell.coordinate[axis] - min(c.coordinate[axis] for c in lattice.cells))
    occupancy = lattice.get_cell_occupancy_matrix()
    assert all(occupancy[tuple(cell.pos)] is cell for cell in lattice.cells)
    # Neighbours across the symmetry plane
    n_neighbours = sorted(len(cell.get_all_cell_neighbours()) for cell in lattice.cells)
    assert n_neighbours == ([1, 1, 2, 2] if axis == 0 else [2, 2, 2, 2])


def test_symmetry_mirrors_radius_gradient(tmp_path):
    config = {
        "geometry": {"cell_size": {"x": 1, "y": 1, "z": 1}, "number_of_cells": {"x": 3, "y": 1, "z": 1},
                     "radii": [0.05], "geom_types": ["BCC"]},
        "gradient": {"radii": {"rule": "linear", "direction_x": True, "direction_y": False, "direction_z": False,
                               "parameter_x": 0.5, "parameter_y": 0.0, "parameter_z": 0.0}},
        "supplementary": {"symmetries": {"plane": "YZ", "reference_point": {"x": 0.0}}},
    }
    path = tmp_path / "graded.json"
    path.write_text(json.dumps(config))
    lattice = Lattice(str(path))
    radius_at = {round(cell.coordinate[0], 9): cell.beams_cell and next(iter(cell.beams_cell)).radius
                 for cell in lattice.cells}
    assert len(radius_at) == 6
    # The radius gradient is mirrored: same radius at the same distance from the symmetry plane
    for x in (0.0, 1.0, 2.0):
        assert radius_at[x] == pytest.approx(radius_at[-x - 1.0])
    assert len({round(r, 9) for r in radius_at.values()}) == 3


def test_neighbours_with_non_unit_cell_size(tmp_path):
    config = {"geometry": {"cell_size": {"x": 2, "y": 2, "z": 2}, "number_of_cells": {"x": 4, "y": 1, "z": 1},
                           "radii": [0.05], "geom_types": ["BCC"]}}
    path = tmp_path / "big_cells.json"
    path.write_text(json.dumps(config))
    lattice = Lattice(str(path))
    cells = sorted(lattice.cells, key=lambda c: c.pos)
    assert [len(c.get_all_cell_neighbours()) for c in cells] == [1, 2, 2, 1]
    assert cells[1].get_all_cell_neighbours() and set(cells[1].get_all_cell_neighbours()) == {cells[0], cells[2]}


def test_symmetry_invalid_plane(make_lattice):
    lattice = make_lattice(number_of_cells=(1, 1, 1))
    with pytest.raises(ValueError):
        lattice.apply_symmetry("AB", (0, 0, 0))
    with pytest.raises(ValueError):
        lattice.apply_symmetry("XY", None)


# =============================================================================
# Geometric transformations
# =============================================================================

def test_attractor_lattice(make_lattice):
    lattice = make_lattice()
    attractor = np.array([1.0, 1.0, 1.0])
    positions = {id(n): np.array([n.x, n.y, n.z]) for n in lattice.nodes}
    design_transformation.attractor_lattice(lattice, list(attractor), alpha=0.1)
    # Each node (even shared by several beams and cells) is moved once by alpha * |d| * d, d = attractor - node
    for node in lattice.nodes:
        before = positions[id(node)]
        d = attractor - before
        np.testing.assert_allclose([node.x, node.y, node.z], before + 0.1 * np.linalg.norm(d) * d, atol=1e-12)
    # Extreme x: node (0, 1, 1), |d| = 1, moved by alpha
    np.testing.assert_allclose(lattice.get_lattice_boundary_box(), [0.1, 1.9, 0.1, 1.9, 0.1, 1.9], atol=1e-12)


def test_fit_to_surface(make_lattice):
    lattice = make_lattice()
    positions = {id(n): (n.x, n.y, n.z) for n in lattice.nodes}
    design_transformation.fitToSurface(lattice, lambda x, y, slope: slope * x, mode="z", params={"slope": 0.1})
    for node in lattice.nodes:
        x, y, z = positions[id(node)]
        assert (node.x, node.y) == (x, y)
        assert node.z == pytest.approx(z + 0.1 * x)

    lattice = make_lattice()
    design_transformation.fitToSurface(lattice, lambda x, y: 0.5, mode="z_plan")
    assert all(node.z == pytest.approx(0.5) for node in lattice.nodes)

    with pytest.raises(ValueError):
        design_transformation.fitToSurface(make_lattice(), lambda x, y: 0.0, mode="unknown")


def test_cylindrical_transform(make_lattice):
    lattice = make_lattice(number_of_cells=(1, 4, 2))
    n_nodes = lattice.get_number_nodes()
    design_transformation.cylindrical_transform(lattice, 5.0)
    radii = [math.hypot(n.x, n.y) for n in lattice.nodes]
    np.testing.assert_allclose(radii, 5.0, rtol=1e-12)
    # Ymin and Ymax nodes are merged when the lattice closes on itself
    assert lattice.get_number_nodes() < n_nodes

    lattice = make_lattice(number_of_cells=(1, 4, 1))
    design_transformation.cylindrical_transform(lattice, 5.0, wall_thickness=True)
    radii = sorted({round(math.hypot(n.x, n.y), 9) for n in lattice.nodes})
    assert radii[0] == pytest.approx(5.0) and radii[-1] == pytest.approx(6.0)


def test_curve_lattice(make_lattice):
    lattice = make_lattice(number_of_cells=(2, 1, 1))
    positions = {id(n): (n.x, n.y, n.z) for n in lattice.nodes}
    design_transformation.curveLattice(lattice, 1.0, 0.5, 0.5, curvature_strength=0.1)
    # z -= strength * squared distance to the centre of curvature, x and y unchanged
    for node in lattice.nodes:
        x, y, z = positions[id(node)]
        assert (node.x, node.y) == (x, y)
        assert node.z == pytest.approx(z - 0.1 * ((x - 1.0) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2))
    assert lattice.z_min == pytest.approx(0.0 - 0.1 * (1 + 0.25 + 0.25))


def test_move_to_cylinder_form(make_lattice):
    lattice = make_lattice(number_of_cells=(2, 1, 1))
    positions = {id(n): (n.x, n.y, n.z) for n in lattice.nodes}
    design_transformation.moveToCylinderForm(lattice, 5.0)
    # Nodes shared by the two cells are moved only once
    for node in lattice.nodes:
        x, y, z = positions[id(node)]
        assert node.z == pytest.approx(z - (5.0 - math.sqrt(25.0 - (x - 1.0) ** 2)))
    assert lattice.z_min == pytest.approx(-(5.0 - math.sqrt(25.0 - 1.0)))
    assert lattice.z_max == pytest.approx(1.0)
    with pytest.raises(ValueError):
        design_transformation.moveToCylinderForm(make_lattice(number_of_cells=(2, 1, 1)), 0.5)


def test_transformation_updates_beam_lengths(make_lattice):
    lattice = make_lattice(number_of_cells=(1, 1, 1))
    design_transformation.fitToSurface(lattice, lambda x, y: x, mode="z")
    for beam in lattice.beams:
        p1, p2 = beam.point1, beam.point2
        assert beam.length == pytest.approx(math.dist((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z)), abs=1e-4)
    # Membership tests still work after the coordinates (and so the hashes) changed
    assert all(node in lattice.nodes for node in lattice.cells[0].points_cell)


# =============================================================================
# Penalization and utilities
# =============================================================================

def test_penalization_length_strategies():
    radius_neighbor, angle, beam_radius, beam_length = 0.05, 90.0, 0.04, 0.8
    assert penalization_length("angle", radius_neighbor, angle, beam_radius, beam_length) == pytest.approx(
        function_penalization_Lzone(radius_neighbor, angle))
    assert penalization_length("length", radius_neighbor, angle, beam_radius, beam_length) == pytest.approx(0.08)
    assert penalization_length("radius", radius_neighbor, angle, beam_radius, beam_length) == pytest.approx(0.04)
    assert penalization_length("sqrt3_radius", radius_neighbor, angle, beam_radius, beam_length) == pytest.approx(
        math.sqrt(3) * 0.04)
    # Free end or quasi-aligned beams: no penalized zone whatever the strategy
    for strategy in PENALIZATION_STRATEGIES:
        assert penalization_length(strategy, radius_neighbor, 0.0, beam_radius, beam_length) == 0.0
        assert penalization_length(strategy, radius_neighbor, 175.0, beam_radius, beam_length) < 1e-6
    with pytest.raises(ValueError):
        penalization_length("unknown", radius_neighbor, angle, beam_radius, beam_length)


def test_boundary_condition_color():
    color_fixed = get_boundary_condition_color([1, 1, 1, 1, 1, 1])
    assert color_fixed == get_boundary_condition_color([True] * 6)
    assert color_fixed != get_boundary_condition_color([1, 0, 0, 0, 0, 0])
    assert isinstance(color_fixed, str)


def test_save_and_open_pickle_lattice(make_lattice, tmp_path):
    lattice = make_lattice(number_of_cells=(2, 1, 1))
    lattice.change_beam_radius([0.07])
    path = tmp_path / "saved_lattice.pkl"
    save_lattice_object(lattice, str(path))
    assert path.exists()

    loaded = Lattice.open_pickle_lattice(str(path))
    assert loaded.get_number_cells() == lattice.get_number_cells()
    assert loaded.get_number_beams() == lattice.get_number_beams()
    assert loaded.get_number_nodes() == lattice.get_number_nodes()
    assert loaded.get_relative_density() == pytest.approx(lattice.get_relative_density())
    assert loaded.get_lattice_boundary_box() == lattice.get_lattice_boundary_box()

    with pytest.raises(FileNotFoundError):
        Lattice.open_pickle_lattice(str(tmp_path / "missing.pkl"))


# =============================================================================
# Volume with Gmsh (exact volume of the union of the beams)
# =============================================================================

def test_relative_density_gmsh(make_lattice):
    pytest.importorskip("gmsh")
    lattice = make_lattice(number_of_cells=(1, 1, 1))
    relative_density = lattice.generate_mesh_lattice_Gmsh(cut_mesh_at_boundary=True, save_STL=False,
                                                          volume_computation=True, only_relative_density=True)
    analytical = _bcc_relative_density(0.05)
    # The union of the beams removes the overlaps at the nodes: slightly below the sum of the beam volumes
    assert 0.8 * analytical < relative_density < analytical
