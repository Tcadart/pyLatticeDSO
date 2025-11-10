# 📚 Examples

This page provides practical examples of using pyLattice for various lattice design and simulation tasks.

---

## 🔵 Basic Lattice Generation

### Simple BCC Lattice
Create and visualize a basic Body-Centered Cubic (BCC) lattice structure:

```python
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

# Load a simple BCC lattice configuration
name_file = "design/simple_BCC"
lattice_object = Lattice(name_file)

# Create visualizer and plot the lattice
visualizer = LatticePlotting()
visualizer.visualize_lattice(lattice_object, beam_color_type="radii")
```

### All Design Parameters
Example showcasing all available design parameters:

```python
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

name_file = "design/all_design_parameters"
lattice_object = Lattice(name_file, verbose=1)

# Visualize with different color schemes
visualizer = LatticePlotting()
visualizer.visualize_lattice(lattice_object, beam_color_type="materials")
```

---

## 🔧 Mesh Operations

### Lattice Trimming with MeshTrimmer
Use MeshTrimmer to cut lattice structures to fit complex geometries:

```python
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting
from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer

# Load a mesh (e.g., bone structure)
name_mesh = "CutedBone"
mesh_trimmer = MeshTrimmer(name_mesh)
mesh_trimmer.scale_mesh(1.5)

# Create lattice with mesh trimming
name_lattice = "Bone_cuted_hybrid"
lattice_object = Lattice.from_json(name_lattice, mesh_trimmer)

# Visualize the trimmed lattice
visualizer = LatticePlotting()
visualizer.visualize_lattice(lattice_object)
```

---

## 🧮 Finite Element Simulation

### Simple Beam Flexion Simulation
Perform a finite element simulation using pyLatticeSim:

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults

# Load simulation configuration
name_file = "simulation/beam_flexion"
lattice_sim = LatticeSim(name_file, verbose=1)

# Run FE simulation
result = solve_FEM_FenicsX(lattice_sim)

# Export results
exportSimulationResults(lattice_sim, result)
```

### Homogenization Analysis
Compute effective material properties of a lattice unit cell:

```python
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils import create_homogenization_figure
from pyLatticeSim.utils_simulation import get_homogenized_properties
from pyLatticeSim.export_simulation_results import exportSimulationResults

name_file = "simulation/hybrid_cell_homogenization"
lattice_sim = LatticeSim(name_file, verbose=1)

# Compute homogenized properties
properties = get_homogenized_properties(lattice_sim)
print(f"Effective Young's modulus: {properties['E_eff']:.2f} MPa")
```

---

## 🚀 Optimization

### Topology Optimization
Optimize lattice topology for minimum compliance:

```python
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti

name_file = "optimization/optimization_beam_flexion"
lattice_object = LatticeOpti(name_file, verbose=1, convergence_plotting=True)

# Run optimization
lattice_object.run_optimization()

# Visualize optimized result
visualizer = LatticePlotting()
visualizer.visualize_lattice(lattice_object.get_optimized_lattice())
```

### Surrogate Model Generation
Create surrogate models for predicting lattice properties:

```python
from pyLattice.lattice import Lattice
from pyLatticeOpti.surrogate_model_relative_densities import (
    compute_relative_densities_dataset, 
    plot_3D_scatter,
    evaluate_kriging_from_pickle
)

# Generate dataset
dataset = compute_relative_densities_dataset("hybrid_cell_parametric")

# Create and visualize surrogate model
plot_3D_scatter(dataset)
model = evaluate_kriging_from_pickle("relative_density_model")
```

---

## 💾 Data Management

### Saving and Loading Lattices
Save computed lattices for later use:

```python
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

# Create and save a lattice
lattice = Lattice("design/complex_lattice")
lattice.save_pickle_lattice("my_lattice")

# Load the saved lattice
loaded_lattice = Lattice.open_pickle_lattice("my_lattice")

# Visualize
visualizer = LatticePlotting()
visualizer.visualize_lattice(loaded_lattice)
```

---

## 📐 Custom Geometries

### Adding New Unit Cell Geometries
Define custom lattice unit cells using JSON geometry files:

```json
{
  "name": "MyCustomCell",
  "description": "A custom parametric lattice unit cell",
  "parameters": {
    "height": 0.35,
    "angle": 20,
    "offset": "height - tan(angle * pi / 180) / 2"
  },
  "beams": [
    [0.0, 0.0, 0.0, 0.5, 0.5, "height"],
    [0.5, 0.5, "height", 1.0, 1.0, 1.0],
    [0.0, 1.0, "offset", 1.0, 0.0, "1 - offset"]
  ]
}
```

Use the custom geometry:
```python
from pyLattice.lattice import Lattice

# The geometry file should be placed in src/pyLattice/geometries/
lattice = Lattice("design/config_with_custom_cell")
```

---

## 🛠️ Advanced Usage

### Domain Decomposition
For large-scale simulations using domain decomposition methods:

```python
from pyLatticeSim.lattice_sim import LatticeSim

name_file = "simulation/3PointBending"
solver_DDM = LatticeSim(name_file, verbose=1, enable_domain_decomposition_solver=True)

# Run simulation with domain decomposition
result = solver_DDM.solve_with_DDM()
```

### Schur Complement Methods
Use Schur complement methods for efficient substructuring:

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import get_schur_complement

name_file = "simulation/hybrid_cell_simulation"
lattice_sim = LatticeSim(name_file, verbose=1)

# Compute Schur complement
schur_matrix = get_schur_complement(lattice_sim)
```

---

For more examples, check the `examples/` directory in the repository, which contains:
- **design/**: Basic lattice generation examples
- **simulation/**: Finite element analysis examples  
- **optimization/**: Topology and parameter optimization examples

Each example includes a corresponding JSON parameter file and detailed comments explaining the setup and usage.