"""
This example shows how to save a pickle lattice object.
"""
from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.plotting_lattice import LatticePlotting
from pyLatticeDesign.utils import save_lattice_object


path = "design/"
name_file = "L_logo"

lattice_object = Lattice(path + name_file, verbose=1)

save_lattice_object(lattice_object, name_file + "_saved")

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
