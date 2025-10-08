"""
Example script to compare FEM simulation with DDM simulation for a beam under flexion.
"""
import numpy as np
import time

from pyLatticeSim.lattice_sim import LatticeSim
from pyLattice.plotting_lattice import LatticePlotting
from pyLatticeSim.utils_simulation import solve_FEM_FenicsX

path = "simulation/"
name_file = "Cantilever_L_beam"
# name_file = "Three_point_bending"
# name_file = "Inversion_mechanism"


start_time85 = time.time()
lattice_Sim_object = LatticeSim(path + name_file, verbose = 1)
print("Lattice generation time --- %s seconds ---" % (time.time() - start_time85))

# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii",
#                              use_radius_grad_color=True)

sol_FEM = solve_FEM_FenicsX(lattice_Sim_object)[0]
print("FEM simulation time --- %s seconds ---" % (time.time() - start_time85))

# np.save("sol_array.npy", sol_FEM)


# U, RF = lattice_Sim_object.get_boundary_conditions_displacements_and_reactions()
# for i, val in enumerate(U):
#     print(f"Displacement at DOF {i}: {val}")
#     print(f"Reaction force at DOF {i}: {RF[i]}")

# disp, force = lattice_Sim_object.get_global_force_displacement_curve(dof=2)  # Z direction
# print("Force:", force)


# Visualization with matplotlib
# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_Sim_object, beam_color_type="radii",
#                              enable_boundary_conditions=True,
#                              deformedForm=True)

# sol_FEM = np.load("sol_array.npy")
start_time85 = time.time()
lattice_object = LatticeSim(path + name_file, enable_domain_decomposition_solver = True, verbose=1)
print("Lattice generation time --- %s seconds ---" % (time.time() - start_time85))


sol_DDM = lattice_object.solve_DDM()[0]
print("DDM simulation time --- %s seconds ---" % (time.time() - start_time85))

relative_error = np.linalg.norm(sol_FEM - sol_DDM) / np.linalg.norm(sol_FEM)
print("Relative error between FEM and DDM", relative_error)

np.save("ddm_array.npy", sol_DDM)


# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii",
#                              use_radius_grad_color=True)

# vizualizer = LatticePlotting()
# vizualizer.visualize_lattice(lattice_object, beam_color_type="radii",
#                              enable_boundary_conditions=True,
#                              deformedForm=True)