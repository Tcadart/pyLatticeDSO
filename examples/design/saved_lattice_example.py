"""
This example shows how to load a saved lattice and visualize it.
"""
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting
from pyLattice.utils import save_JSON_to_Grasshopper

# name_lattice = "Kelvin_helmet"
name_lattice = "Cantilever_L_beam_optimized"
# name_lattice = "Three_point_bending_optimized"
# name_lattice = "optimization_DDM_surrogate_saved"
# name_lattice = "Inversion_mechanism_optimized"

lattice = Lattice.open_pickle_lattice(name_lattice)


lattice.delete_beams_under_radius_threshold(0.05)
lattice.merge_degree2_nodes()
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice, beam_color_type="radii", use_radius_grad_color=True)
save_JSON_to_Grasshopper(lattice, name_lattice + "_deleted_small_radii")


lattice.delete_unconnected_beams()
vizualizer = LatticePlotting()
vizualizer.visualize_lattice(lattice, beam_color_type="radii", use_radius_grad_color=True)
save_JSON_to_Grasshopper(lattice, name_lattice + "_deleted_small_radii_and_unconnected")
