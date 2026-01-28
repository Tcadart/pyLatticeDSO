"""Test module for Point class functionality."""
import pytest
import math
from pyLatticeDesign.point import Point
from pyLatticeDesign.cell import Cell


@pytest.fixture
def test_cell():
    """Create a test cell for point testing."""
    return Cell([0, 0, 0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ['BCC'], [0.05], None, None, None)


def test_point_creation(test_cell):
    """Test basic point creation."""
    point = Point(1.0, 2.0, 3.0, [test_cell])
    
    # Note: coordinates might have small random perturbations due to uncertainty
    assert abs(point.x - 1.0) < 0.1  # Allow for default uncertainty
    assert abs(point.y - 2.0) < 0.1
    assert abs(point.z - 3.0) < 0.1
    assert point.cell_belongings == [test_cell]


def test_point_with_no_uncertainty(test_cell):
    """Test point creation with zero uncertainty."""
    point = Point(1.0, 2.0, 3.0, [test_cell], node_uncertainty_SD=0.0)
    
    assert point.x == 1.0
    assert point.y == 2.0
    assert point.z == 3.0


def test_point_with_uncertainty(test_cell):
    """Test point creation with uncertainty."""
    # Set a small but non-zero uncertainty
    uncertainty = 0.01
    point = Point(1.0, 2.0, 3.0, [test_cell], node_uncertainty_SD=uncertainty)
    
    # Coordinates should be close to original but potentially slightly different
    assert abs(point.x - 1.0) <= 5 * uncertainty  # 5-sigma confidence
    assert abs(point.y - 2.0) <= 5 * uncertainty
    assert abs(point.z - 3.0) <= 5 * uncertainty


def test_point_input_validation(test_cell):
    """Test point input validation."""
    # Test invalid coordinate types
    with pytest.raises(ValueError, match="Coordinates must be numeric"):
        Point("not_a_number", 0, 0, [test_cell])
    
    with pytest.raises(ValueError, match="Coordinates must be numeric"):
        Point(0, "not_a_number", 0, [test_cell])
    
    with pytest.raises(ValueError, match="Coordinates must be numeric"):
        Point(0, 0, "not_a_number", [test_cell])
    
    # Test invalid uncertainty
    with pytest.raises(ValueError, match="Node uncertainty standard deviation must be numeric"):
        Point(0, 0, 0, [test_cell], node_uncertainty_SD="invalid")
    
    with pytest.raises(ValueError, match="Node uncertainty standard deviation cannot be negative"):
        Point(0, 0, 0, [test_cell], node_uncertainty_SD=-0.1)


def test_point_default_attributes(test_cell):
    """Test point default attributes."""
    point = Point(0, 0, 0, [test_cell])
    
    assert point.index is None
    assert point.tag is None
    assert point.cell_local_tag == {}
    assert point.index_boundary is None
    assert point.displacement_vector == [0.0] * 6
    assert point.reaction_force_vector == [0.0] * 6
    assert point.applied_force == [0.0] * 6
    assert point.fixed_DOF == [False] * 6


def test_point_distance_calculation(test_cell):
    """Test distance calculation between points."""
    point1 = Point(0, 0, 0, [test_cell], node_uncertainty_SD=0.0)
    point2 = Point(3, 4, 0, [test_cell], node_uncertainty_SD=0.0)
    
    # Calculate distance manually
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z
    expected_distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    assert expected_distance == 5.0  # 3-4-5 triangle


def test_point_coordinate_properties(test_cell):
    """Test point coordinate properties."""
    x, y, z = 1.5, 2.5, 3.5
    point = Point(x, y, z, [test_cell], node_uncertainty_SD=0.0)
    
    assert isinstance(point.x, float)
    assert isinstance(point.y, float)
    assert isinstance(point.z, float)
    assert point.x == x
    assert point.y == y
    assert point.z == z


def test_point_with_multiple_cells():
    """Test point belonging to multiple cells."""
    cell1 = Cell([0, 0, 0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ['BCC'], [0.05], None, None, None)
    cell2 = Cell([1, 0, 0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], ['BCC'], [0.05], None, None, None)
    
    point = Point(0.5, 0.5, 0.5, [cell1, cell2], node_uncertainty_SD=0.0)
    
    assert len(point.cell_belongings) == 2
    assert cell1 in point.cell_belongings
    assert cell2 in point.cell_belongings


def test_point_dof_modification(test_cell):
    """Test modification of degrees of freedom."""
    point = Point(0, 0, 0, [test_cell])
    
    # Modify displacement vector
    point.displacement_vector[0] = 1.5
    assert point.displacement_vector[0] == 1.5
    assert point.displacement_vector[1] == 0.0  # Others should remain unchanged
    
    # Modify fixed DOF
    point.fixed_DOF[0] = True
    point.fixed_DOF[3] = True
    assert point.fixed_DOF[0] is True
    assert point.fixed_DOF[3] is True
    assert point.fixed_DOF[1] is False  # Others should remain False


def test_point_force_vectors(test_cell):
    """Test force vector functionality."""
    point = Point(0, 0, 0, [test_cell])
    
    # Test applied force modification
    point.applied_force[2] = 100.0  # Force in Z direction
    assert point.applied_force[2] == 100.0
    assert all(f == 0.0 for i, f in enumerate(point.applied_force) if i != 2)
    
    # Test reaction force modification
    point.reaction_force_vector[5] = -50.0  # Moment about Z
    assert point.reaction_force_vector[5] == -50.0
    assert all(f == 0.0 for i, f in enumerate(point.reaction_force_vector) if i != 5)