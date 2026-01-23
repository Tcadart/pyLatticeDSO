"""
Simple example of how to use pyrough to create a BCC lattice cell with rough surfaces.
"""

from pyLattice.lattice import Lattice

path_file_lattice = "Pyrough/BCC_cell"

lattice_object = Lattice(path_file_lattice)

name_file_rough_parameters = "lattice_wire.json"
name_stl_out = "BCC_rough_mesh"

lattice_object.generate_mesh_lattice_rough(name_file_rough_parameters, name_stl_out, cut_mesh_at_boundary=True)

