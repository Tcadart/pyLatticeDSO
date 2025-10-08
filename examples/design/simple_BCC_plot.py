"""
Simple example of how to plot a simple BCC lattice using Matplotlib.
"""

from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeSim.lattice_sim import LatticeSim

name_file = "design/simple_BCC"

# lattice_object = Lattice(name_file)
lattice_object = LatticeSim(name_file, verbose=1)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice_object, beam_color_type="radii",use_radius_grad_color =True,
                             enable_system_coordinates=False)
