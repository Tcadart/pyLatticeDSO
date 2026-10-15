![pyLattice_logo](/docs/source/images/pyLattice_logo.png)

# pyLatticeDSO: Design, Simulation, and Optimization for truss lattice structures

[![Tests](https://github.com/Tcadart/pyLatticeDSO/actions/workflows/tests.yml/badge.svg)](https://github.com/Tcadart/pyLatticeDSO/actions/workflows/tests.yml)
[![Documentation](https://github.com/Tcadart/pyLatticeDSO/actions/workflows/deploy-docs.yml/badge.svg)](https://tcadart.github.io/pyLatticeDSO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENCE.txt)

**pyLatticeDSO** is a Python toolkit to design, simulate and optimize truss lattice structures. A lattice is described
by a JSON input file (cell geometries, cell sizes, radii, boundary conditions, ...) and goes through three modules:

| Module | Content | Main dependencies |
|---|---|---|
| `pyLatticeDesign` | Lattice generation (BCC, Octet, Kelvin, hybrid cells, custom cells), graded radii and cell sizes, trimming with a mesh, plotting, export (Gmsh mesh, Grasshopper) | NumPy, Matplotlib, Gmsh |
| `pyLatticeSim` | Finite element simulation with a beam model, 3D solid model, homogenization, domain decomposition method (Schur complements, reduced basis, surrogate models) | FEniCSx 0.9, `dolfinx_mpc`, PETSc |
| `pyLatticeOpti` | Optimization of the beam radii, with the full beam model or the domain decomposition method | SciPy, scikit-learn |

📖 **Documentation:** <https://tcadart.github.io/pyLatticeDSO/>

## Installation

The simulation module relies on FEniCSx, which is only distributed through conda-forge and requires Linux or macOS
(on Windows, use WSL2). The tested environment is pinned in [`environment.yml`](environment.yml):

```bash
git clone https://github.com/Tcadart/pyLatticeDSO.git
cd pyLatticeDSO
conda env create -f environment.yml
conda activate pyLatticeDSO
pip install -e .
```

The design module alone only needs `pip install -e .` (Python 3.10 to 3.12, any OS).
See the [installation guide](docs/source/guides/Installation_tutorial.md) for the details, including Windows (WSL2)
and Docker.

## Quick start

```python
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting

# Input file: data/inputs/preset_lattice/design/simple_BCC.json
lattice = Lattice("design/simple_BCC")
LatticePlotting().visualize_lattice(lattice, beam_color_type="radii")
```

Simulation of a BCC lattice in flexion (requires the simulation module):

```python
from pyLatticeSim.lattice_sim import LatticeSim
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

lattice = LatticeSim("simulation/simulation_beam_flexion")
solution, simulation = solve_FEM_FenicsX(lattice)
```

More examples are in [`examples/`](examples/):

- `design/`: lattice generation, all design parameters, meshing, trimming with a mesh
- `simulation/`: beam and solid simulations, homogenization, domain decomposition, Schur complement datasets
- `optimization/`: radius optimization with the full beam model and with the domain decomposition method
- `comparison_convergence/`: comparisons between models and convergence studies

## Tests

```bash
pytest -v                 # all tests
pytest -v -m simulation   # simulation verification tests only
```

The test suite contains unit tests of the design module and verification tests of the simulation module against
closed-form solutions. The simulation tests are skipped when FEniCSx is not installed. The tests run on every push
through GitHub Actions. See [Reproducibility](docs/source/guides/Reproducibility.md) for the tested versions and the
computational requirements.

## Repository structure

```
src/
  pyLatticeDesign/   lattice generation, cell geometries (geometries/*.json), materials (materials/*.json)
  pyLatticeSim/      finite element simulation, homogenization, domain decomposition
  pyLatticeOpti/     optimization
data/inputs/         JSON input files (preset_lattice/) and meshes
data/outputs/        simulation results, Schur complement datasets, figures
examples/            example scripts
Tests/               test suite (pytest)
docs/                documentation (MkDocs)
```

## Contributing and contact

Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
Contact: Thomas Cadart, [thomas.cadart19@gmail.com](mailto:thomas.cadart19@gmail.com)

## License

pyLatticeDSO is distributed under the MIT license, see [LICENCE.txt](LICENCE.txt).
