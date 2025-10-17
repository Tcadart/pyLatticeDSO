"""
Example script to generate a lattice mesh
"""
from pyLattice.lattice import Lattice
from pyLattice.plotting_lattice import LatticePlotting

path = "design/"
name_file = "hybrid_cell"

lattice_object = Lattice(path + name_file)

# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")

volume = lattice_object.generate_mesh_lattice_Gmsh(volume_computation=True, cut_mesh_at_boundary=True,
                                                name_mesh=name_file, mesh_refinement = 0.5)
print("Volume of the lattice structure:", volume, "m3")
lattice_object.timing.summary(group_by_category=True, name_width=48)

# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii")
