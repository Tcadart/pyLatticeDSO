# Example Tests

This directory contains tests to validate all example scripts in the `examples/` directory.

## What is Tested

The test suite performs the following checks on all example scripts:

1. **Syntax Validation**: Ensures all example files have valid Python syntax
2. **Import Validation**: Ensures all imports in example files can be resolved
3. **Documentation**: Ensures all examples have module-level docstrings

## Test Categories

Tests are organized by example category:

- **Design Examples** (`examples/design/`): Basic lattice design examples requiring only core dependencies
- **Optimization Examples** (`examples/optimization/`): Examples using optimization features requiring scikit-learn
- **Simulation Examples** (`examples/simulation/`): Examples using FEM simulation requiring FEniCSx (dolfinx)

## Running Tests Locally

### Run All Tests
```bash
cd /home/runner/work/pyLattice/pyLattice
MPLBACKEND=Agg python -m pytest tests/test_examples.py -v
```

### Run Tests for Specific Category
```bash
# Design examples only
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_design_examples_syntax -v
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_design_examples_imports -v

# Optimization examples only
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_optimization_examples_syntax -v
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_optimization_examples_imports -v

# Simulation examples only
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_simulation_examples_syntax -v
MPLBACKEND=Agg python -m pytest tests/test_examples.py::test_simulation_examples_imports -v
```

### Prerequisites

Before running tests, ensure dependencies are installed:

```bash
# Core dependencies
pip install -e .

# For optimization examples
pip install -e ".[optimization]"

# For simulation examples (requires conda)
conda install -c conda-forge fenics-dolfinx=0.9.0
pip install -e ".[simulation]"
```

## CI/CD Integration

The tests run automatically via GitHub Actions on every push to the `master` branch. See `.github/workflows/test-examples.yml` for the workflow configuration.

Each category (design, optimization, simulation) runs in a separate job with appropriate dependencies installed.

## Notes

- Tests use `MPLBACKEND=Agg` to run in headless mode without requiring a display
- The matplotlib backend is now configurable via the `MPLBACKEND` environment variable
- Import tests validate that modules can be imported but do not execute the examples
- Simulation examples require system dependencies (MPI, GMSH with OpenGL) that are installed in CI
