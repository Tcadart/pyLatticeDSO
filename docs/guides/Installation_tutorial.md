# 📘 Installation Guide

## Overview
**pyLattice** is a Python toolkit for the design, analysis, and finite element (FE) simulation of lattice structures.  
The package supports:
- **Lattice generation and visualization**  
- **Mesh trimming and geometry operations**  
- **Finite Element simulations with FEniCSx**  

This guide explains how to set up a Python environment with **conda** and install all dependencies required for both core functionality and optional modules.

---

## 1. Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).  
- Linux is recommended for full compatibility with FEniCSx and Gmsh.  

---

## 2. Create a Conda Environment
We recommend Python **3.12** for compatibility with FEniCSx and PyEmbree.  

```bash
conda create -n pyLatticeDesign python=3.12
conda activate pyLatticeDesign
```

---

## 3. Install Core Dependencies
Install the core package and its main dependencies using `pip`
```bash
pip install -e .
```
This will install the core dependencies:
- `numpy`
- `matplotlib`
- `colorama`
- `joblib`
- `pytest`
- `gmsh`
- `sympy`

---

## 4. Install Optional Dependencies
### 4.1. Simulation (FEniCSx-based)
The simulation backend relies on [FEniCSx](https://fenicsproject.org/).
These packages are not available on PyPI and must be installed via conda-forge:
```bash
conda install -c conda-forge fenics-dolfinx dolfinx_mpc
```
This will install:
- `dolfinx`
- `ufl`
- `basix`
- `petsc4py`
- `dolfinx_mpc`

⚠️ Do not attempt to install these packages with `pip`, as they require compiled binaries only distributed via `conda-forge`.

### 4.2. Mesh Operation Dependencies
For **mesh trimming and ray intersection operations**, additional geometry libraries are required.
It is recommended to install them via conda-forge for proper native support:
```bash
conda install -c conda-forge trimesh rtree pyembree libspatialindex
```
This will install:
- `trimesh` - geometry and ray operations
- `rtree` - spatial indexing (requires `libspatialindex`)
- `pyembree` - high-performance ray tracing backend

---

## 5. Optional: Verify Installation
After installation, test your setup:
```bash
python -c "import pyLattice; print('pyLattice installed successfully')"
```

To check optional modules:
```bash
# Simulation check
python -c "import dolfinx; print('FEniCSx OK')"

# Mesh operation check
python -c "import trimesh, rtree; print('Trimesh & Rtree OK')"
```

---

## 6. Example Usage
Run the provided examples from the `examples/` directory:
```bash
python examples/simple_BCC_plot.py
```

