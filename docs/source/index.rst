.. pyLattice documentation master file, created by
   sphinx-quickstart on Fri Jul 25 15:46:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyLatticeDSO's documentation!
============================================

**pyLatticeDSO** is a comprehensive Python toolkit for designing, analyzing, and simulating lattice structures.
The package provides capabilities for:

- **Lattice Generation**: Create various lattice geometries (BCC, Octet, Kelvin, etc.)
- **Visualization**: Interactive 3D plotting with matplotlib or Plotly
- **Finite Element Analysis**: Structural simulations using FEniCSx
- **Mesh Operations**: Advanced geometry trimming and manipulation
- **Optimization**: Topology and parameter optimization tools

Getting Started
---------------

1. Follow the :doc:`Installation_tutorial` to set up your environment
2. Check out the :doc:`Examples` for practical usage scenarios
3. Learn about :doc:`cell_geometries` to create custom unit cells
4. Explore the API documentation in :doc:`modules`

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   guides/Installation_tutorial
   guides/Examples
   guides/cell_geometries
   guides/Boundary_conditions
   guides/MeshTrimmer_class
   JSON_input_parameters

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/test

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


