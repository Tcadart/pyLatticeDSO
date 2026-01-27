# JSON Input Parameters Documentation

This document provides a comprehensive reference for all possible JSON input parameters used in pyLattice, pyLatticeSim, and pyLatticeOpti.

## Table of Contents
1. [Lattice (Design) Parameters](#lattice-design-parameters)
2. [LatticeSim (Simulation) Parameters](#latticesim-simulation-parameters)
3. [LatticeOpti (Optimization) Parameters](#latticeOpti-optimization-parameters)

---

## LatticeDesign Parameters

These parameters are used for creating basic lattice structures.

| Parameter Name | Architecture/Path | Data Type | Description |
|----------------|-------------------|-----------|-------------|
| `cell_size.x` | `geometry.cell_size.x` | float | Cell size in X direction |
| `cell_size.y` | `geometry.cell_size.y` | float | Cell size in Y direction |
| `cell_size.z` | `geometry.cell_size.z` | float | Cell size in Z direction |
| `number_of_cells.x` | `geometry.number_of_cells.x` | int | Number of cells in X direction |
| `number_of_cells.y` | `geometry.number_of_cells.y` | int | Number of cells in Y direction |
| `number_of_cells.z` | `geometry.number_of_cells.z` | int | Number of cells in Z direction |
| `radii` | `geometry.radii` | list[float] | List of radii values for beams |
| `geom_types` | `geometry.geom_types` | list[str] | List of geometry types (e.g., "BCC", "Octahedron", "Hybrid1", "Hybrid4") |
| `enable_randomness` | `geometry.enable_randomness` | bool | Enable randomness in radius selection (optional, default: false) |
| `range_radius` | `geometry.range_radius` | list[float] | Range for random radius [min, max] (optional, default: [0.01, 0.1]) |
| `randomness_hybrid` | `geometry.randomness_hybrid` | bool | Enable randomness in hybrid cell types (optional, default: false) |
| `radii.rule` | `gradient.radii.rule` | str | Gradient rule for radii ("constant", "linear") (optional, default: "constant") |
| `radii.direction_x` | `gradient.radii.direction_x` | bool | Apply gradient in X direction (optional, default: false) |
| `radii.direction_y` | `gradient.radii.direction_y` | bool | Apply gradient in Y direction (optional, default: false) |
| `radii.direction_z` | `gradient.radii.direction_z` | bool | Apply gradient in Z direction (optional, default: false) |
| `radii.parameter_x` | `gradient.radii.parameter_x` | float | Gradient parameter for X direction (optional, default: 0.0) |
| `radii.parameter_y` | `gradient.radii.parameter_y` | float | Gradient parameter for Y direction (optional, default: 0.0) |
| `radii.parameter_z` | `gradient.radii.parameter_z` | float | Gradient parameter for Z direction (optional, default: 0.0) |
| `cell_dimension.rule` | `gradient.cell_dimension.rule` | str | Gradient rule for cell dimensions (optional, default: "constant") |
| `cell_dimension.direction_x` | `gradient.cell_dimension.direction_x` | bool | Apply gradient in X direction (optional, default: false) |
| `cell_dimension.direction_y` | `gradient.cell_dimension.direction_y` | bool | Apply gradient in Y direction (optional, default: false) |
| `cell_dimension.direction_z` | `gradient.cell_dimension.direction_z` | bool | Apply gradient in Z direction (optional, default: false) |
| `cell_dimension.parameter_x` | `gradient.cell_dimension.parameter_x` | float | Gradient parameter for X direction (optional, default: 0.0) |
| `cell_dimension.parameter_y` | `gradient.cell_dimension.parameter_y` | float | Gradient parameter for Y direction (optional, default: 0.0) |
| `cell_dimension.parameter_z` | `gradient.cell_dimension.parameter_z` | float | Gradient parameter for Z direction (optional, default: 0.0) |
| `material.type_beam` | `gradient.material.type_beam` | int | Material beam type (optional, default: 0) |
| `material.direction` | `gradient.material.direction` | int | Material gradient direction (optional, default: 0) |
| `node_uncertainty` | `suplementary.node_uncertainty` | float | Node position uncertainty (optional, default: 0.0) |
| `erased_blocks` | `suplementary.erased_blocks` | dict | Dictionary of blocks to erase from the lattice (optional) |
| `erased_blocks.block_N.start_point.x` | `suplementary.erased_blocks.block_N.start_point.x` | float | X coordinate of block start point |
| `erased_blocks.block_N.start_point.y` | `suplementary.erased_blocks.block_N.start_point.y` | float | Y coordinate of block start point |
| `erased_blocks.block_N.start_point.z` | `suplementary.erased_blocks.block_N.start_point.z` | float | Z coordinate of block start point |
| `erased_blocks.block_N.dimensions_block.x` | `suplementary.erased_blocks.block_N.dimensions_block.x` | float | Block dimension in X direction |
| `erased_blocks.block_N.dimensions_block.y` | `suplementary.erased_blocks.block_N.dimensions_block.y` | float | Block dimension in Y direction |
| `erased_blocks.block_N.dimensions_block.z` | `suplementary.erased_blocks.block_N.dimensions_block.z` | float | Block dimension in Z direction |
| `symmetries.plane` | `suplementary.symmetries.plane` | str | Symmetry plane ("xy", "xz", "yz") (optional) |
| `symmetries.reference_point.x` | `suplementary.symmetries.reference_point.x` | float | X coordinate of symmetry reference point (optional, default: 0.0) |
| `symmetries.reference_point.y` | `suplementary.symmetries.reference_point.y` | float | Y coordinate of symmetry reference point (optional, default: 0.0) |
| `symmetries.reference_point.z` | `suplementary.symmetries.reference_point.z` | float | Z coordinate of symmetry reference point (optional, default: 0.0) |

---

## LatticeSim (Simulation) Parameters

These parameters extend the Lattice parameters with simulation-specific settings. **All Lattice parameters above are also available.**

| Parameter Name | Architecture/Path | Data Type | Description |
|----------------|-------------------|-----------|-------------|
| `enable` | `simulation_parameters.enable` | bool | Enable simulation properties |
| `material` | `simulation_parameters.material` | str | Material name (e.g., "VeroClear") (optional, default: "VeroClear") |
| `periodicity` | `simulation_parameters.periodicity` | bool | Enable periodic boundary conditions (optional, default: false) |
| `DDM.enable_preconditioner` | `simulation_parameters.DDM.enable_preconditioner` | bool | Enable preconditioner for DDM solver (optional) |
| `DDM.preconditioner_type` | `simulation_parameters.DDM.preconditioner_type` | str | Type of preconditioner ("mean", "nearest_reference") (optional) |
| `DDM.max_iterations` | `simulation_parameters.DDM.max_iterations` | int | Maximum iterations for DDM solver (optional, default: 1000) |
| `DDM.schur_complement_computation.type` | `simulation_parameters.DDM.schur_complement_computation.type` | str | Schur complement computation type ("exact", "FE2", "nearest_neighbor", "linear", "RBF") (required if DDM enabled) |
| `DDM.schur_complement_computation.precision_greedy` | `simulation_parameters.DDM.schur_complement_computation.precision_greedy` | float | Precision for greedy algorithm (required for non-exact types) |
| `boundary_conditions.Displacement` | `boundary_conditions.Displacement` | dict | Displacement boundary conditions (optional) |
| `boundary_conditions.Displacement.<Name>.Surface` | `boundary_conditions.Displacement.<Name>.Surface` | list[str] | Surfaces to apply displacement (e.g., ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"]) |
| `boundary_conditions.Displacement.<Name>.DOF` | `boundary_conditions.Displacement.<Name>.DOF` | list[str] | Degrees of freedom ("X", "Y", "Z", "RX", "RY", "RZ") |
| `boundary_conditions.Displacement.<Name>.Value` | `boundary_conditions.Displacement.<Name>.Value` | list[float] | Displacement values for each DOF |
| `boundary_conditions.Displacement.<Name>.SurfaceCells` | `boundary_conditions.Displacement.<Name>.SurfaceCells` | list[str] | Surface cells for constraint (optional) |
| `boundary_conditions.Force` | `boundary_conditions.Force` | dict | Force boundary conditions (optional) |
| `boundary_conditions.Force.<Name>.Surface` | `boundary_conditions.Force.<Name>.Surface` | list[str] | Surfaces to apply force |
| `boundary_conditions.Force.<Name>.DOF` | `boundary_conditions.Force.<Name>.DOF` | list[str] | Degrees of freedom for force application |
| `boundary_conditions.Force.<Name>.Value` | `boundary_conditions.Force.<Name>.Value` | list[float] | Force values for each DOF |

---

## LatticeOpti (Optimization) Parameters

These parameters extend the LatticeSim parameters with optimization-specific settings. **All Lattice and LatticeSim parameters above are also available.**

| Parameter Name | Architecture/Path | Data Type | Description |
|----------------|-------------------|-----------|-------------|
| `objective_function` | `optimization_informations.objective_function` | str | Optimization objective ("min" or "max") (required) |
| `objective_type` | `optimization_informations.objective_type` | str | Type of objective function (e.g., "compliance") (required) |
| `objective_data` | `optimization_informations.objective_data` | any | Additional objective data (optional) |
| `max_iterations` | `optimization_informations.max_iterations` | int | Maximum optimization iterations (optional, default: 20) |
| `optimization_parameters.type` | `optimization_informations.optimization_parameters.type` | str | Type of optimization parameters ("constant", "unit_cell", etc.) (required) |
| `optimization_parameters.direction` | `optimization_informations.optimization_parameters.direction` | str | Optimization direction ("x", "y", "z") (optional) |
| `optimization_parameters.hybrid` | `optimization_informations.optimization_parameters.hybrid` | bool | Enable hybrid optimization (required) |
| `constraints.relative_density` | `optimization_informations.constraints.relative_density` | dict | Relative density constraint (optional) |
| `constraints.relative_density.value` | `optimization_informations.constraints.relative_density.value` | float | Target relative density value |
| `constraints.relative_density.compute_gradient` | `optimization_informations.constraints.relative_density.compute_gradient` | bool | Compute gradient for constraint (optional) |
| `enable_parameter_normalization` | `optimization_informations.enable_parameter_normalization` | bool | Enable parameter normalization (optional, default: false) |
| `simulation_type` | `optimization_informations.simulation_type` | str | Simulation type for optimization ("FEM" or "DDM") (required) |
| `enable_gradient_computing` | `optimization_informations.enable_gradient_computing` | bool | Enable gradient computing (optional, default: false) |

---

## Examples

### LatticeDesign Example
```json
{
  "geometry": {
    "cell_size": {"x": 1, "y": 1, "z": 1},
    "number_of_cells": {"x": 3, "y": 3, "z": 4},
    "radii": [0.1],
    "geom_types": ["Octahedron"]
  },
  "gradient": {
    "radii": {
      "rule": "linear",
      "direction_y": true,
      "parameter_y": 1.01
    }
  }
}
```

### LatticeSim (Simulation) Example
```json
{
  "geometry": {
    "cell_size": {"x": 1, "y": 1, "z": 1},
    "number_of_cells": {"x": 1, "y": 1, "z": 1},
    "radii": [0.05],
    "geom_types": ["BCC"]
  },
  "simulation_parameters": {
    "enable": true,
    "material": "VeroClear",
    "periodicity": true
  }
}
```

### LatticeOpti (Optimization) Example
```json
{
  "geometry": {
    "cell_size": {"x": 1, "y": 1, "z": 1},
    "number_of_cells": {"x": 6, "y": 3, "z": 3},
    "radii": [0.1],
    "geom_types": ["BCC"]
  },
  "simulation_parameters": {
    "enable": true,
    "material": "VeroClear",
    "periodicity": false
  },
  "boundary_conditions": {
    "Displacement": {
      "Fixed": {
        "Surface": ["Xmin"],
        "DOF": ["X", "Y", "Z", "RX", "RY", "RZ"],
        "Value": [0, 0, 0, 0, 0, 0]
      }
    }
  },
  "optimization_informations": {
    "objective_function": "min",
    "objective_type": "compliance",
    "optimization_parameters": {
      "type": "constant",
      "hybrid": false
    },
    "constraints": {
      "relative_density": {"value": 0.03}
    },
    "enable_parameter_normalization": true,
    "simulation_type": "FEM"
  }
}
```

---

## Notes

- Parameters marked as "(optional)" have default values that will be used if not specified
- Parameters marked as "(required)" must be provided for the respective functionality
- The architecture path shows the nested structure in the JSON file
- All numeric values should be appropriate for the physical context (e.g., positive values for dimensions)
- String values for surfaces: "Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"
- String values for DOF: "X", "Y", "Z", "RX", "RY", "RZ"
