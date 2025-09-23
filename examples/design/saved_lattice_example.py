"""
This example shows how to load a saved lattice and visualize it.
"""
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

name_lattice = "Kelvin_helmet"

lattice = Lattice.open_pickle_lattice(name_lattice)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice, beam_color_type="radii")
vizualizer.show()
