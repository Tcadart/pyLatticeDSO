# 📘 Installation Guide

## Overview
This guide explains how to set up a Python environment with **conda** and install all dependencies required for both core functionality and optional modules.

---

## 1. Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).  
- Linux is recommended for full compatibility with FEniCSx and Gmsh.  
- ⚠️ **The simulation module (`pyLatticeSim`) requires Linux (e.g. Ubuntu) or macOS.** Windows users must go through WSL2, see [Windows users](#412-windows-users-wsl2).  

---

## 2. Create a Conda Environment
We recommend Python **3.12** for compatibility with FEniCSx and PyEmbree.  

```bash
conda create -n pyLatticeDSO python=3.12
conda activate pyLatticeDSO
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
conda install -c conda-forge fenics-dolfinx=0.9.0 dolfinx_mpc
```
This will install:
- `dolfinx`
- `ufl`
- `basix`
- `petsc4py`
- `dolfinx_mpc`

⚠️ Do not attempt to install these packages with `pip`, as they require compiled binaries only distributed via `conda-forge`.

#### 4.1.1. Linux / Ubuntu only
`dolfinx_mpc` depends on PETSc, which is not available for Windows on conda-forge. The simulation part of the code must therefore be run on **Linux (e.g. Ubuntu)** or macOS. The design and mesh parts (`pyLatticeDesign`) work on Windows.

#### 4.1.2. Windows users (WSL2)
On Windows, install Ubuntu through WSL2 (Windows Subsystem for Linux) and run the simulation inside it:

1. Install WSL2 with Ubuntu. Open **PowerShell as administrator**, run the command below, then restart the computer:
   ```powershell
   wsl --install -d Ubuntu
   ```
   On first launch, Ubuntu asks you to create a Linux username and password.

2. In the Ubuntu terminal, install Miniforge (a conda distribution that uses conda-forge by default):
   ```bash
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh
   source ~/.bashrc
   ```

3. Get the code. You can either clone the repository inside the Linux file system (faster), or reach your Windows files through `/mnt/c/...`:
   ```bash
   git clone <repository-url> ~/pyLatticeDSO
   cd ~/pyLatticeDSO
   # or: cd "/mnt/c/Users/<user>/path/to/pyLatticeDSO"
   ```

4. Create the environment and install everything:
   ```bash
   conda create -n pyLatticeDSO -c conda-forge python=3.12 fenics-dolfinx=0.9.0 dolfinx_mpc=0.9 mpich
   conda activate pyLatticeDSO
   pip install -e .
   ```

5. (Optional) To edit and run the code from VS Code, install the **WSL** extension, then run `code .` from the Ubuntu terminal.

Alternatively, a ready-to-use Docker image with FEniCSx and `dolfinx_mpc` is available:
```bash
docker run -it -v "$(pwd):/root/shared" ghcr.io/jorgensd/dolfinx_mpc:v0.9.0
```

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
python -c "import pyLatticeDesign, pyLatticeSim, pyLatticeOpti; print('pyLattice installed successfully')"
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

