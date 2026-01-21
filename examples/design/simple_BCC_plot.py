"""
Simple example of how to plot a simple BCC lattice using Matplotlib.
"""

from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

name_file = "design/"
name_lattice = "simple_BCC"

lattice_object = Lattice(name_file + name_lattice, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_system_coordinates=False)
