"""
Example of lattice structure design with all entries from the package.
"""

from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

# name_file = "design/all_design_parameters"
name_file = "design/hybrid_cell"

lattice_object = Lattice(name_file, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
