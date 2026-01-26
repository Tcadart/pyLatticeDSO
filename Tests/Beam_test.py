"""Test module for Beam class functionality."""
import pytest
from src.pyLatticeDesign.point import Point
from src.pyLatticeDesign.beam import Beam
from src.pyLatticeDesign.cell import Cell


@pytest.fixture
def test_cell():
    """Create a test cell for beam testing."""
    return Cell([0, 0, 0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ['BCC'], [0.05], None, None, None)


@pytest.fixture
def test_points(test_cell):
    """Create test points for beam testing."""
    point1 = Point(0, 0, 0, [test_cell])
    point2 = Point(1, 0, 0, [test_cell])
    return point1, point2


def test_beam_creation(test_points, test_cell):
    """Test basic beam creation."""
    point1, point2 = test_points
    beam = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    assert beam.point1 == point1
    assert beam.point2 == point2
    assert beam.radius == 0.05
    assert beam.material == 1
    assert beam.type_beam == 0
    assert test_cell in beam.cell_belongings


def test_beam_length(test_points, test_cell):
    """Test beam length calculation."""
    point1, point2 = test_points
    beam = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    assert beam.length == 1.0
    assert isinstance(beam.length, float)


def test_beam_volume(test_points, test_cell):
    """Test beam volume calculation."""
    point1, point2 = test_points
    beam = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    expected_volume = 3.141592653589793 * (0.05 ** 2) * 1.0  # π * r² * length
    assert abs(beam.volume - expected_volume) < 1e-10
    assert isinstance(beam.volume, float)


def test_beam_equality(test_points, test_cell):
    """Test beam equality comparison."""
    point1, point2 = test_points
    beam1 = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    # Create identical points and beam
    point3 = Point(0, 0, 0, [test_cell])
    point4 = Point(1, 0, 0, [test_cell])
    beam2 = Beam(point3, point4, 0.05, 1, 0, test_cell)
    
    # Test beam comparison with tolerance
    tolerance = 1e-10  # Single float value, not list
    assert beam1.is_identical_to(beam2, tolerance)


def test_beam_representation(test_points, test_cell):
    """Test beam string representation."""
    point1, point2 = test_points
    beam = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    repr_str = repr(beam)
    assert isinstance(repr_str, str)
    assert "Beam" in repr_str
    assert "0.05" in repr_str  # radius should be mentioned


def test_beam_properties(test_points, test_cell):
    """Test beam properties and attributes."""
    point1, point2 = test_points
    beam = Beam(point1, point2, 0.05, 1, 0, test_cell)
    
    # Test default values
    assert beam.index is None
    assert beam.beam_mod is False
    assert beam.penalization_coefficient == 1.5
    assert beam.initial_radius is None
    
    # Test angle properties
    assert "radius" in beam.angle_point_1
    assert "angle" in beam.angle_point_1
    assert "L_zone" in beam.angle_point_1


def test_beam_with_different_points(test_cell):
    """Test beam with different point configurations."""
    # Test vertical beam
    point1 = Point(0, 0, 0, [test_cell])
    point2 = Point(0, 0, 2, [test_cell])
    beam = Beam(point1, point2, 0.03, 2, 1, test_cell)
    
    assert beam.length == 2.0
    assert beam.radius == 0.03
    assert beam.material == 2
    assert beam.type_beam == 1
    
    # Test diagonal beam
    point3 = Point(0, 0, 0, [test_cell])
    point4 = Point(1, 1, 1, [test_cell])
    beam2 = Beam(point3, point4, 0.04, 0, 2, test_cell)
    
    expected_length = (3 ** 0.5)  # sqrt(1² + 1² + 1²)
    assert abs(beam2.length - expected_length) < 1e-4  # More reasonable tolerance

