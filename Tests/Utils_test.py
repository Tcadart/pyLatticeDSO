"""Test module for utility functions."""
import pytest
import tempfile
import json
import os
from pathlib import Path
from src.pyLattice.utils import (
    open_lattice_parameters, 
    _validate_inputs_lattice,
    _validate_inputs_cell,
    function_penalization_Lzone
)


def test_open_lattice_parameters():
    """Test opening lattice parameters from JSON file."""
    # Test with existing file
    params = open_lattice_parameters("design/BCC_cell.json")
    assert isinstance(params, dict)
    assert "geometry" in params
    assert "cell_size" in params["geometry"]
    assert "number_of_cells" in params["geometry"]
    assert "radii" in params["geometry"]
    assert "geom_types" in params["geometry"]


def test_open_lattice_parameters_file_not_found():
    """Test opening non-existent lattice parameters file."""
    with pytest.raises(FileNotFoundError):
        open_lattice_parameters("nonexistent_file.json")


def test_validate_inputs_lattice_valid():
    """Test lattice input validation with valid inputs."""
    # Should not raise any exceptions
    _validate_inputs_lattice(
        cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
        num_cells_x=2, num_cells_y=2, num_cells_z=2,
        geom_types=["BCC"], radii=[0.05],
        grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
        uncertainty_node=0.0, eraser_blocks=None
    )


def test_validate_inputs_lattice_invalid_cell_size():
    """Test lattice input validation with invalid cell sizes."""
    # Test negative cell size
    with pytest.raises(AssertionError, match="cell_size_x must be a positive number"):
        _validate_inputs_lattice(
            cell_size_x=-1.0, cell_size_y=1.0, cell_size_z=1.0,
            num_cells_x=2, num_cells_y=2, num_cells_z=2,
            geom_types=["BCC"], radii=[0.05],
            grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
            uncertainty_node=0.0, eraser_blocks=None
        )
    
    # Test zero cell size
    with pytest.raises(AssertionError, match="cell_size_y must be a positive number"):
        _validate_inputs_lattice(
            cell_size_x=1.0, cell_size_y=0.0, cell_size_z=1.0,
            num_cells_x=2, num_cells_y=2, num_cells_z=2,
            geom_types=["BCC"], radii=[0.05],
            grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
            uncertainty_node=0.0, eraser_blocks=None
        )


def test_validate_inputs_lattice_invalid_num_cells():
    """Test lattice input validation with invalid number of cells."""
    # Test non-integer number of cells
    with pytest.raises(AssertionError, match="num_cells_x must be a positive integer"):
        _validate_inputs_lattice(
            cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
            num_cells_x=2.5, num_cells_y=2, num_cells_z=2,
            geom_types=["BCC"], radii=[0.05],
            grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
            uncertainty_node=0.0, eraser_blocks=None
        )
    
    # Test negative number of cells
    with pytest.raises(AssertionError, match="num_cells_y must be a positive integer"):
        _validate_inputs_lattice(
            cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
            num_cells_x=2, num_cells_y=-1, num_cells_z=2,
            geom_types=["BCC"], radii=[0.05],
            grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
            uncertainty_node=0.0, eraser_blocks=None
        )


def test_validate_inputs_cell_valid():
    """Test cell input validation with valid inputs."""
    # Should not raise any exceptions
    _validate_inputs_cell(
        pos=[0, 0, 0],
        initial_size=[1.0, 1.0, 1.0],
        coordinate=[0.0, 0.0, 0.0],
        geom_types=["BCC"],
        radii=[0.05],
        grad_radius=None,
        grad_dim=None,
        grad_mat=None,
        uncertainty_node=0.0,
        _verbose=0
    )


def test_validate_inputs_cell_invalid_pos():
    """Test cell input validation with invalid position."""
    with pytest.raises(TypeError):
        _validate_inputs_cell(
            pos=[0, 0],  # Should have 3 elements
            initial_size=[1.0, 1.0, 1.0],
            coordinate=[0.0, 0.0, 0.0],
            geom_types=["BCC"],
            radii=[0.05],
            grad_radius=None,
            grad_dim=None,
            grad_mat=None,
            uncertainty_node=0.0,
            _verbose=0
        )


def test_validate_inputs_cell_invalid_radii():
    """Test cell input validation with invalid radii."""
    with pytest.raises(TypeError):
        _validate_inputs_cell(
            pos=[0, 0, 0],
            initial_size=[1.0, 1.0, 1.0],
            coordinate=[0.0, 0.0, 0.0],
            geom_types=["BCC"],
            radii="invalid",  # Should be list
            grad_radius=None,
            grad_dim=None,
            grad_mat=None,
            uncertainty_node=0.0,
            _verbose=0
        )


def test_function_penalization_lzone():
    """Test penalization L-zone function."""
    # Test with different inputs
    result1 = function_penalization_Lzone(0.05, 0.0)
    result2 = function_penalization_Lzone(0.05, 0.5)
    result3 = function_penalization_Lzone(0.05, 1.0)
    
    # All results should be numeric
    assert isinstance(result1, (int, float))
    assert isinstance(result2, (int, float))
    assert isinstance(result3, (int, float))
    
    # Results should be non-negative for valid inputs
    assert result1 >= 0
    assert result2 >= 0
    assert result3 >= 0


def test_function_penalization_lzone_edge_cases():
    """Test penalization L-zone function with edge cases."""
    # Test with very small positive number
    result = function_penalization_Lzone(1e-10, 0.1)
    assert isinstance(result, (int, float))
    assert result >= 0
    
    # Test with larger number
    result = function_penalization_Lzone(0.1, 2.0)
    assert isinstance(result, (int, float))


def test_lattice_validation_geometry_types():
    """Test validation of geometry types."""
    # Valid geometry type
    _validate_inputs_lattice(
        cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
        num_cells_x=1, num_cells_y=1, num_cells_z=1,
        geom_types=["BCC"], radii=[0.05],
        grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
        uncertainty_node=0.0, eraser_blocks=None
    )
    
    # Valid multiple geometry types
    _validate_inputs_lattice(
        cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
        num_cells_x=1, num_cells_y=1, num_cells_z=1,
        geom_types=["BCC", "FCC"], radii=[0.05, 0.08],
        grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
        uncertainty_node=0.0, eraser_blocks=None
    )


def test_lattice_validation_radii_geometry_mismatch():
    """Test validation when radii and geometry types don't match in length."""
    with pytest.raises(AssertionError):
        _validate_inputs_lattice(
            cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0,
            num_cells_x=1, num_cells_y=1, num_cells_z=1,
            geom_types=["BCC", "FCC"], radii=[0.05],  # Mismatch: 2 geom types, 1 radius
            grad_radius_property=None, grad_dim_property=None, grad_mat_property=None,
            uncertainty_node=0.0, eraser_blocks=None
        )