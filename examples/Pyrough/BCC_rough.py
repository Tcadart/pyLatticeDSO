"""
Simple example of how to use pyrough to create a BCC lattice cell with rough surfaces.
"""

from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

name_file = "design/simple_BCC"

lattice_object = Lattice(name_file)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_system_coordinates=False,
                             file_save_path="simple_BCC_plot.png")

