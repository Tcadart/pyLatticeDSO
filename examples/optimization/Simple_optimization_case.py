"""
Examples of a simple optimization case.
"""
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeOpti.lattice_opti import LatticeOpti

name_file = "optimization/optimization_beam_flexion"

lattice_object = LatticeOpti(name_file, verbose=1, convergence_plotting = True)

lattice_object.optimize_lattice()

# Visualization optimized lattice
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_boundary_conditions=True)
