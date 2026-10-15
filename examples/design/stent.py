"""
Simple example of how to plot a simple BCC lattice using Matplotlib.
"""

from math import cos, sin

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.design_transformation import cylindrical_transform
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "design/"
name_lattice = "stent"

lattice_object = Lattice(name_file + name_lattice, verbose=1)

# Modify the curvature of the beams
cylindrical_transform(lattice_object, radius=3, wall_thickness=False)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_system_coordinates=True)
