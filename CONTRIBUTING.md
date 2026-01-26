# Contributing to pyLattice

We welcome contributions to pyLattice! This document provides guidelines for contributing to the project.

## Contact

For questions, suggestions, or collaboration opportunities, please contact:
📧 [thomas.cadart19@gmail.com](mailto:thomas.cadart19@gmail.com)

## Development Setup

1. **Fork and Clone**: Fork the repository and clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/pyLattice.git
   cd pyLatticeDesign
   ```

2. **Environment Setup**: Create a development environment:
   ```bash
   conda create -n pyLatticeDesign-dev python=3.12
   conda activate pyLatticeDesign-dev
   pip install -e .[dev]
   ```

3. **Install Optional Dependencies** (if needed):
   ```bash
   # For simulation features
   conda install -c conda-forge fenics-dolfinx dolfinx_mpc
   
   # For mesh operations
   conda install -c conda-forge trimesh rtree pyembree libspatialindex
   ```

## Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include a clear description and reproduction steps
- Provide system information (OS, Python version, dependency versions)

### 💡 Feature Requests  
- Describe the proposed feature and its use case
- Consider starting with a discussion before implementing

### 🔧 Code Contributions
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 📚 Documentation
- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify explanations

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to new functions and classes
- Keep functions focused and modular

### Testing
- Run existing tests: `python -m pytest Tests/`
- Add tests for new features
- Test both core functionality and edge cases

### Documentation
- Update relevant documentation for any changes
- Build docs locally to check formatting:
  ```bash
  cd docs
  make html
  ```

### Commit Guidelines
- Use clear, descriptive commit messages
- Keep commits focused on a single change
- Reference issue numbers when applicable

## Pull Request Process

1. **Create a Branch**: Create a feature branch from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your changes following the guidelines above

3. **Test**: Ensure all tests pass and add new tests as needed

4. **Documentation**: Update documentation and examples if applicable

5. **Submit PR**: Create a pull request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes (if any)

6. **Review Process**: Respond to feedback and make requested changes

## Project Structure

Understanding the project structure helps with contributions:

```
pyLattice/
├── src/pyLattice/          # Core lattice generation
├── src/pyLatticeSim/       # Simulation backend  
├── src/pyLatticeOpti/      # Optimization tools
├── examples/               # Example scripts
├── Tests/                  # Unit tests
├── docs/                   # Documentation source
├── data/                   # Input data and configurations
└── pyproject.toml         # Project configuration
```

## Adding New Features

### New Lattice Geometries
1. Create a JSON geometry file in `src/pyLattice/geometries/`
2. Follow the existing format (see `cell_geometries.md`)
3. Add an example showing how to use it
4. Update documentation

### New Simulation Features
1. Add functionality to `src/pyLatticeSim/`
2. Ensure compatibility with FEniCSx
3. Add comprehensive tests
4. Document the new capability

### New Optimization Methods
1. Implement in `src/pyLatticeOpti/`
2. Follow the existing optimization interface
3. Provide examples and validation cases

## Questions?

If you have questions about contributing, please:
1. Check existing issues and documentation
2. Start a discussion on GitHub
3. Contact the maintainer via email

Thank you for contributing to pyLattice! 🙏

