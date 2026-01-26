"""Test module for Lattice class functionality."""
import os
import tempfile
import json
import pytest
from src.pyLatticeDesign.lattice import Lattice


@pytest.fixture
def simple_lattice_config():
    """Create a simple lattice configuration for testing."""
    return {
        "geometry": {
            "cell_size": {"x": 1, "y": 1, "z": 1},
            "number_of_cells": {"x": 2, "y": 2, "z": 2},
            "radii": [0.05],
            "geom_types": ["BCC"]
        }
    }


@pytest.fixture
def small_lattice_config():
    """Create a minimal lattice configuration for testing."""
    return {
        "geometry": {
            "cell_size": {"x": 1, "y": 1, "z": 1},
            "number_of_cells": {"x": 1, "y": 1, "z": 1},
            "radii": [0.05],
            "geom_types": ["BCC"]
        }
    }


@pytest.fixture
def temp_lattice_file(simple_lattice_config):
    """Create a temporary JSON file with lattice configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(simple_lattice_config, f)
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_small_lattice_file(small_lattice_config):
    """Create a temporary JSON file with minimal lattice configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(small_lattice_config, f)
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


def test_lattice_creation(temp_lattice_file):
    """Test basic lattice creation from JSON file."""
    lattice = Lattice(temp_lattice_file)
    assert lattice is not None
    assert len(lattice.cells) == 8  # 2x2x2 = 8 cells
    assert lattice.num_cells_x == 2
    assert lattice.num_cells_y == 2
    assert lattice.num_cells_z == 2


def test_lattice_dimensions(temp_lattice_file):
    """Test lattice dimension calculations."""
    lattice = Lattice(temp_lattice_file)
    assert lattice.size_x == 2.0  # 2 cells * 1 unit each
    assert lattice.size_y == 2.0
    assert lattice.size_z == 2.0
    assert lattice.x_min == 0.0
    assert lattice.x_max == 2.0
    assert lattice.y_min == 0.0
    assert lattice.y_max == 2.0
    assert lattice.z_min == 0.0
    assert lattice.z_max == 2.0


def test_beam_and_node_counts(temp_small_lattice_file):
    """Test beam and node counting functionality."""
    lattice = Lattice(temp_small_lattice_file)
    num_beams = lattice.get_number_beams()
    num_nodes = lattice.get_number_nodes()
    assert num_beams > 0
    assert num_nodes > 0
    assert isinstance(num_beams, int)
    assert isinstance(num_nodes, int)


def test_relative_density(temp_small_lattice_file):
    """Test relative density calculation."""
    lattice = Lattice(temp_small_lattice_file)
    rd = lattice.get_relative_density()
    assert 0 < rd < 1
    assert isinstance(rd, float)


def test_lattice_statistics(temp_small_lattice_file):
    """Test lattice statistical methods."""
    lattice = Lattice(temp_small_lattice_file)
    
    # Test that we can get basic statistics without errors
    num_beams = lattice.get_number_beams()
    num_nodes = lattice.get_number_nodes()
    relative_density = lattice.get_relative_density()
    
    assert all(isinstance(x, (int, float)) for x in [num_beams, num_nodes, relative_density])
    assert num_beams > 0
    assert num_nodes > 0
    assert 0 < relative_density < 1


def test_lattice_representation(temp_small_lattice_file):
    """Test lattice string representation."""
    lattice = Lattice(temp_small_lattice_file)
    repr_str = repr(lattice)
    assert isinstance(repr_str, str)
    assert "Lattice" in repr_str
    assert "1.0" in repr_str  # Size should be mentioned
