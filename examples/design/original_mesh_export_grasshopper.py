"""
Simple example of how to plot a simple BCC lattice using Matplotlib.
"""

from math import cos, sin

from pyLatticeDesign.lattice import Lattice
from pyLatticeDesign.design_transformation import fitToSurface
from pyLatticeDesign.plotting_lattice import LatticePlotting

name_file = "design/"
name_lattice = "original_mesh"

lattice_object = Lattice(name_file + name_lattice, verbose=1)

# Modify the curvature of the beams
fitToSurface(lattice_object, lambda x, y: sin(x) + sin(y), mode="z", params=None)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", enable_system_coordinates=True)
