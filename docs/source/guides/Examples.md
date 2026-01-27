# 📚 Examples

This page provides practical examples of using pyLatticeDSO for various lattice design, simulation tasks, and optimization scenarios. Each example includes code snippets and explanations to help you get started quickly.

---

## 🔵 Basic Lattice Generation

### Simple BCC Lattice
Create and visualize a basic Body-Centered Cubic (BCC) lattice structure:

```python
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "design/"
name_lattice = "simple_BCC"

lattice_object = Lattice(name_file + name_lattice, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_system_coordinates=False)

```

### All Design Parameters
Example showcasing all available design parameters:

```python
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "design/all_design_parameters"

lattice_object = Lattice(name_file, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object)
```

---

## 🔧 Mesh Operations

### Lattice Trimming with MeshTrimmer
Use MeshTrimmer to cut lattice structures to fit complex geometries:

```python
from src.pyLatticeDesign.plotting_lattice import LatticePlotting
from src.pyLatticeDesign.lattice import Lattice

from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer

name_mesh = "CutedBone"  # get from https://anatomytool.org/content/thunthu-3d-model-bones-lower-limb
mesh_trimmer = MeshTrimmer(name_mesh)
mesh_trimmer.plot_mesh(zoom = 3)

name_lattice = "design/BCC_trimmed_example"
lattice_object = Lattice(name_lattice, mesh_trimmer)
lattice_object.cut_beam_with_mesh_trimmer()

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")

```
See the [MeshTrimmer class documentation](MeshTrimmer_class.md) for more details.

---

## 🧮 Finite Element Simulation

### Simple Beam Flexion Simulation
Perform a finite element simulation using pyLatticeSim:

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/simulation_beam_flexion"

lattice_Sim_object = LatticeSim(name_file)

sol, simulation_lattice = solve_FEM_FenicsX(lattice_Sim_object)

# Visualization with matplotlib
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii", deformed_form=True,
                             enable_boundary_conditions=True)

# Export the results to Paraview
export_results = exportSimulationResults(simulation_lattice, name_file)
export_results.export_displacement_rotation()
export_results.export_finalize()
```

### Homogenization Analysis
Compute effective material properties of a lattice unit cell:

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils import create_homogenization_figure
from pyLatticeSim.utils_simulation import get_homogenized_properties
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_file = "simulation/hybrid_cell_simulation"

lattice_object = LatticeSim(name_file)

mat_S_orthotropic, homogenization_analysis = get_homogenized_properties(lattice_object)

create_homogenization_figure(mat_S_orthotropic, save=True, name_file=name_file, plot= False)

# Export simulations to Paraview
exportData = exportSimulationResults(homogenization_analysis, name_file)
exportData.export_data_homogenization()
```

---

## 🚀 Optimization

### Optimization of Lattice Structure
Optimize lattice structure for minimum compliance:

```python
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti

name_file = "optimization/optimization_beam_flexion"

lattice_object = LatticeOpti(name_file, verbose=1, convergence_plotting = True)

lattice_object.optimize_lattice()

# Visualization optimized lattice
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True)
```

---

## 💾 Data Management

### Saving and Loading Lattices
Save computed lattices for later use:

```python
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeDesign.utils import save_lattice_object


path = "design/"
name_file = "L_logo"

lattice_object = Lattice(path + name_file, verbose=1)

save_lattice_object(lattice_object, name_file + "_saved")

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
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

Save the above JSON content in a file named `MyCustomCell.json` inside the `src/pyLatticeDesign/geometries/` directory.

---

## 🛠️ Advanced Usage

### Domain Decomposition
For large-scale simulations using domain decomposition methods:

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "simulation/Three_point_bending"

solver_DDM = LatticeSim(name_file, verbose=1, enable_domain_decomposition_solver=True)

solver_DDM.solve_DDM()

# Visualization with matplotlib
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(solver_DDM, beam_color_type="radii", deformed_form=True, enable_boundary_conditions=True,
                             domain_decomposition_simulation_plotting=True)
```


---

For more examples, check the `examples/` directory in the repository, which contains:
- **design/**: Lattice structure generation examples
- **simulation/**: Finite element analysis examples and domain decomposition
- **optimization/**: Parameter optimization examples

Each example includes a corresponding JSON parameter file and detailed comments explaining the setup and usage.