"""
Simple example of how to export a lattice structure for Grasshopper.
"""

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.utils import save_JSON_to_Grasshopper

name_file = "design/"
name_lattice = "simple_BCC"

lattice_object = Lattice(name_file + name_lattice, verbose=1)

save_JSON_to_Grasshopper(lattice_object, name_file)

