# 🔁 Tested environment, verification and reproducibility

## 1. Tested software versions

The results of the article were produced with the environment below, which is pinned in `environment.yml` at the root
of the repository.

| Package | Tested version | Channel | Used by |
|---|---|---|---|
| Python | 3.12.14 | conda-forge | all |
| NumPy | 1.26.4 | conda-forge | all |
| Matplotlib | 3.11.2 | PyPI | all |
| Gmsh | 4.15.2 | PyPI | design (meshing) |
| SymPy | 1.14.0 | PyPI | design |
| joblib | 1.6.0 | conda-forge | design, optimization |
| FEniCSx (`fenics-dolfinx`, UFL, Basix) | 0.9.0 (UFL 2024.2.0) | conda-forge | simulation |
| `dolfinx_mpc` | 0.9.1 | conda-forge | simulation (periodic conditions) |
| PETSc / petsc4py | 3.23.0 | conda-forge | simulation |
| mpi4py / Open MPI | 4.1.2 / 5.0.8 | conda-forge | simulation |
| SciPy | 1.17.1 | conda-forge | simulation, optimization |
| scikit-learn | 1.9.1 | conda-forge | optimization (surrogate models) |
| trimesh / Rtree / PyEmbree | 5.1.1 / 1.4.1 / 0.1.6 | conda-forge | mesh trimming (optional) |
| pytest | 9.1.1 | PyPI | tests |

Tested operating systems: Ubuntu under WSL2 (Windows 11, Linux kernel 6.18) and GitHub Actions `ubuntu-latest`.
The design module (`pyLatticeDesign`) also runs on Windows and macOS with Python 3.10 to 3.12; the simulation module
requires Linux or macOS (see the [installation guide](Installation_tutorial.md)).

## 2. Installation

```bash
git clone https://github.com/Tcadart/pyLatticeDSO.git
cd pyLatticeDSO
conda env create -f environment.yml
conda activate pyLatticeDSO
pip install -e .
```

The design module alone only needs `pip install -e .`.

## 3. Computational requirements

Reference machine: laptop with an Intel Core i7-9750H (6 cores, 12 threads, 2.6 GHz) and 8 GB of RAM, Ubuntu under
WSL2. All simulations of the article run on this machine in serial. 8 GB of RAM is the limiting factor only for the
3D solid reference meshes (e.g. the Octet cell at r = 0.025 does not fit in memory); the beam and domain decomposition
models need much less memory.

## 4. Automated tests

The test suite runs with `pytest` from the root of the repository:

```bash
pytest -v                 # all tests
pytest -v -m simulation   # simulation verification tests only
```

| Test file | Content |
|---|---|
| `Point_test.py`, `Beam_test.py`, `Cell_test.py`, `Lattice_test.py` | Unit tests of the geometric entities and of the lattice generation |
| `Gradient_test.py` | Graded radii and cell sizes |
| `Materials_test.py`, `Utils_test.py`, `Timing_test.py` | Material library, input files and utilities |
| `Homogenization_verification_test.py` | Verification of the beam homogenization against the closed-form stiffness of the Cubic and BCC cells (relative error below 1e-3) |

The simulation tests are skipped automatically when FEniCSx is not installed. The GitHub Actions workflow
`.github/workflows/tests.yml` runs the tests on every push and pull request: the design tests with pip on Python 3.10,
3.11 and 3.12, and the full suite in the environment of `environment.yml`.
