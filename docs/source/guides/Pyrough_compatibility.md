# pyLattice ↔︎ Pyrough compatibility
Combining the design capabilities of **pyLatticeDSO** with the surface roughness control provided by **Pyrough** 
enables the generation of truss lattice structures with rough beam surfaces.
This integration allows users to investigate the influence of surface roughness on the mechanical behavior of lattice structures, which is of major interest for applications in additive manufacturing and materials science.

---
## Installation
To use **pyLatticeDSO** with **Pyrough**, install pyLatticeDSO as a submodule inside your Pyrough repository:

```bash
git submodule add https://github.com/jamodeo12/Pyrough.git Pyrough/
```
Additional dependencies are required to handle the increased mesh complexity of lattice structures.
From the Pyrough root directory, run:
```bash
pip install trimesh manifold3d
```

## Usage
To generate a rough lattice structure, two input files are required:
1. A lattice geometry definition file created for **pyLatticeDSO**
2. A roughness parameter file processed by **Pyrough**

Sample files are provided in:
- `examples/BCC_cell.json` (lattice geometry)
- `examples/lattice_wire.json` (roughness configuration)
From the Pyrough root directory, run:
```bash
python Pyrough.py .\examples\lattice_wire.json
```
This command generates a rough lattice structure using the lattice geometry defined in `BCC_cell.json` and applies the surface roughness parameters specified in `lattice_wire.json`.
The image below illustrates a BCC lattice structure with rough beam surfaces generated using this approach:
![Rough Lattice Structure](../images/BCC_rough.png)

See the [Pyrough documentation](https://jamodeo12.github.io/Pyrough/) for more details on configuring roughness parameters and additional functionalities.

---
## Influence of the roughness on the mechanical response
The example `examples/Pyrough/BCC_rough_same_density.py` quantifies the effect of the roughness on a BCC cell at
constant relative density (3D solid model, linear elasticity):

1. Pyrough generates rough cells with random roughness parameters (RMS, Hurst exponent, number of Fourier modes) with
   `Lattice.generate_mesh_lattice_rough` (the `seed` argument makes the random surface reproducible);
2. as Pyrough adds the roughness outward, the strut radius of each rough cell is decreased until its relative density
   matches the one of the smooth cell;
3. the surface mesh is cut by the flat bounding box, cleaned and meshed with tetrahedra
   (`SolidModel(..., surface_mesh_file=...)`);
4. the apparent moduli in compression and in shear (bonded rigid platens) are compared with the smooth cell.

Requirements: the `pyLattice_comp_V2` branch of Pyrough, whose root folder is given by the environment variable
`PYROUGH_PATH`, and `pip install manifold3d pymeshlab trimesh networkx wulffpack ase meshio`.
```bash
PYROUGH_PATH=/path/to/Pyrough python examples/Pyrough/BCC_rough_same_density.py
```
