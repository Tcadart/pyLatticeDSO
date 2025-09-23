"""
This example shows how to load a saved lattice and visualize it.
"""
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

name_lattice = "Kelvin_helmet"
name_lattice = "optimization_DDM_surrogate_saved"

lattice = Lattice.open_pickle_lattice(name_lattice)

vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice, beam_color_type="radii", use_radius_grad_color=True)
vizualizer.show()
