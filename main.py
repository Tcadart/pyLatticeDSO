"""
Main
"""
from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer
from pyLatticeOpti.lattice_opti import LatticeOpti
from pyLatticeSim.lattice_sim import LatticeSim
from src.pyLattice.plotting_lattice import LatticePlotting
from src.pyLattice.lattice import Lattice

from src.pyLattice.utils import save_JSON_to_Grasshopper, save_lattice_object
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX
from pyLatticeSim.export_simulation_results import exportSimulationResults


name_mesh = "CutedBone"  # get from https://anatomytool.org/content/thunthu-3d-model-bones-lower-limb
mesh_trimmer = MeshTrimmer(name_mesh)
mesh_trimmer.scale_mesh(1.5)
# mesh_trimmer.plot_mesh(zoom = 3, camera_position=(8.7, -178.7))

name_file = "optimization/"
name_lattice = "Bone_cuted_hybrid"
lattice_object = LatticeOpti(name_file + name_lattice, mesh_trimmer = mesh_trimmer, verbose=1,
                             convergence_plotting =True)


lattice_object.optimize_lattice()

lattice_object.reset_penalized_beams()

lattice_object.delete_beams_under_radius_threshold(0.02)
lattice_object.merge_degree2_nodes()
save_JSON_to_Grasshopper(lattice_object, name_lattice + "_deleted_small_radii_and_unconnected")


# lattice_object.cut_beam_with_mesh_trimmer()
# lattice_object.print_statistics_lattice()

# sol, simulation_lattice = solve_FEM_FenicsX(lattice_object)

# save_JSON_to_Grasshopper(lattice_object, name_lattice)

# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii", voxelViz=False, camera_position=(8.7, -178.7),
#                              enable_system_coordinates=False, deformedForm=True, enable_boundary_conditions=True)
#
# lattice_object.save_optimization_json(name_file = name_lattice + "_optimized")
# save_JSON_to_Grasshopper(lattice_object, name_lattice + "_optimized")
# save_lattice_object(lattice_object, name_lattice + "_optimized")

lattice_object.cut_beam_with_mesh_trimmer()
name_lattice += "_cuted"

save_JSON_to_Grasshopper(lattice_object, name_lattice + "_optimized")
# save_lattice_object(lattice_object, name_lattice + "_optimized")

# sol, simulation_lattice = solve_FEM_FenicsX(lattice_object)
#
# export_results = exportSimulationResults(simulation_lattice, name_lattice)
# export_results.export_displacement_rotation()
# export_results.export_finalize()