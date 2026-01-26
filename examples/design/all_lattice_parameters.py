"""
Example of lattice structure design with all entries from the package.
"""

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "design/all_design_parameters"

lattice_object = Lattice(name_file, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object)
