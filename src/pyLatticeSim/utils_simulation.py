# =============================================================================
# UTILS FOR SIMULATION WITH FENICSX
#
# DESCRIPTION:
# This module provides utility functions for performing finite element
# simulations on lattice structures using the FenicsX library. It includes
# functions for solving full-scale lattice simulations, cell-specific
# simulations, and homogenization analysis.
# =============================================================================

from mpi4py import MPI

from .beam_model import *
from .full_scale_lattice_simulation import *
from .homogenization_cell import HomogenizedCell
from .solid_model import SolidModel
from .solid_lattice_simulation import SolidLatticeSimulation
from .solid_homogenization_cell import SolidHomogenizedCell

if TYPE_CHECKING:
    from pyLatticeSim.lattice_sim import LatticeSim


def solve_FEM_FenicsX(lattice : "LatticeSim"):
    """
    Solve the finite element method problem using FenicsX for a given lattice.

    Parameters:
    -----------
    lattice: Lattice object
        The lattice structure to be simulated.

    Returns:
    --------
    xsol: numpy.ndarray
        The solution vector containing displacements.

    simulationModel: FullScaleLatticeSimulation
        The simulation model containing the results of the simulation.
    """
    # Generate the lattice model and mesh
    LatticeModel = BeamModel(MPI.COMM_SELF, lattice=lattice)

    # Define the FE model and apply boundary conditions
    simulationModel = FullScaleLatticeSimulation(LatticeModel)
    simulationModel.apply_displacement_all_nodes_with_lattice_data()
    simulationModel.apply_force_on_all_nodes_with_lattice_data()
    simulationModel.define_L_form_null()

    # Solve the problem
    simulationModel.solve_problem()

    # Assign results to lattice object
    simulationModel.set_result_diplacement_on_lattice_object()
    simulationModel.set_reaction_force_on_lattice_with_FEM_results()

    # Get results to return
    xsol, globalDisplacementIndex = lattice.get_global_displacement()
    return xsol, simulationModel

def solve_FEM_solid_FenicsX(lattice: "LatticeSim", mesh_refinement: float = 0.5, element_degree: int = 2,
                            solver: str = "direct", verbose: int = 1):
    """
    Solve the 3D linear elastic problem using FenicsX for a given lattice meshed as a solid (tetrahedra).
    Boundary conditions are read from the "boundary_conditions" block of the lattice input file.

    Parameters:
    -----------
    lattice: LatticeSim
        The lattice structure to be simulated.

    mesh_refinement: float
        Refinement factor of the tetrahedral mesh (higher values lead to finer meshes).

    element_degree: int
        Degree of the Lagrange elements (1: linear, 2: quadratic). Linear tetrahedra are much too stiff in bending:
        on a BCC lattice, P2 with mesh_refinement=0.5 is within ~2% of the converged stiffness while P1 is still
        ~5% too stiff with mesh_refinement=1.5.

    solver: str
        "direct" (LU) or "iterative" (conjugate gradient with algebraic multigrid, for large meshes).

    verbose: int
        Verbosity level.

    Returns:
    --------
    simulationModel: SolidLatticeSimulation
        The simulation model containing the results of the simulation.
    """
    # Generate the solid model and mesh
    # Curved elements (same order as the displacement) follow the cylinders exactly
    solid_model = SolidModel(MPI.COMM_SELF, lattice=lattice, mesh_refinement=mesh_refinement,
                             geometry_order=element_degree, verbose=verbose)

    # Define the FE model and apply boundary conditions
    simulationModel = SolidLatticeSimulation(solid_model, element_degree=element_degree, verbose=verbose)
    simulationModel.apply_boundary_conditions_from_lattice()

    # Solve the problem
    simulationModel.solve_problem(solver=solver)

    # Assign results to lattice object
    simulationModel.set_result_displacement_on_lattice_object()
    return simulationModel

def solve_FEM_cell(lattice: "LatticeSim", cell):
    """
    Solve the finite element method problem for a specific cell in the lattice using FenicsX.

    Parameters:
    -----------
    lattice: Lattice object
        The lattice structure to be simulated.

    cell: object
        The specific cell within the lattice to be analyzed.

    """
    # Generate the lattice model and mesh for the specific cell
    cell_model = BeamModel(MPI.COMM_SELF, lattice=lattice, cell_index=cell.index)

    # Define the FE model and apply boundary conditions
    simulation_model = FullScaleLatticeSimulation(cell_model, verbose=0)
    simulation_model.apply_all_boundary_condition_on_cell_without_distinction(cell)

    # Solve the problem
    simulation_model.solve_problem()

    return simulation_model

def get_homogenized_properties(lattice: "LatticeSim"):
    """
    Perform homogenization analysis on a lattice structure.

    Parameters:
    -----------
    lattice: Lattice object
        The lattice structure to be homogenized.

    Returns:
    --------
    mat_Sorthotropic: numpy.ndarray
        The homogenized orthotropic stiffness matrix.

    homogenization_analysis: HomogenizedCell
        The homogenization analysis object containing results and methods.
    """
    if lattice.get_number_cells() > 1:
        raise ValueError("The lattice must contain only one cell for homogenization.")

    cell_model = BeamModel(MPI.COMM_SELF, lattice=lattice)
    # Beams on the faces/edges of the periodic cell are shared with the neighbouring cells
    cell_model.define_periodic_cell_weights()

    # Initialization simulation
    homogenization_analysis = HomogenizedCell(cell_model)
    homogenization_analysis.prepare_simulation()

    homogenization_analysis.apply_dirichlet_for_homogenization()
    homogenization_analysis.periodic_boundary_condition()

    homogenization_analysis.solve_full_homogenization()

    homogenization_analysis.print_homogenized_matrix()
    homogenization_analysis.print_errors()

    mat_Sorthotropic = homogenization_analysis.get_S_orthotropic()

    return mat_Sorthotropic, homogenization_analysis


def get_homogenized_properties_solid(lattice: "LatticeSim", mesh_refinement: float = 0.5, element_degree: int = 2,
                                     verbose: int = 1):
    """
    Perform homogenization analysis on a lattice unit cell meshed as a 3D solid with a periodic mesh.

    Parameters:
    -----------
    lattice: LatticeSim
        The lattice structure to be homogenized (one single cell with a periodic geometry).

    mesh_refinement: float
        Refinement factor of the tetrahedral mesh (higher values lead to finer meshes).

    element_degree: int
        Degree of the Lagrange elements (1: linear, 2: quadratic).

    verbose: int
        Verbosity level.

    Returns:
    --------
    mat_Sorthotropic: numpy.ndarray
        The homogenized orthotropic compliance matrix.

    homogenization_analysis: SolidHomogenizedCell
        The homogenization analysis object containing results and methods.
    """
    if lattice.get_number_cells() > 1:
        raise ValueError("The lattice must contain only one cell for homogenization.")

    # Curved elements (same order as the displacement) follow the cylinders exactly: straight tetrahedra approximate
    # the beam sections by polygons and underestimate the stiffness by a few percent
    solid_model = SolidModel(MPI.COMM_SELF, lattice=lattice, mesh_refinement=mesh_refinement, periodic_mesh=True,
                             geometry_order=element_degree, verbose=verbose)

    homogenization_analysis = SolidHomogenizedCell(solid_model, element_degree=element_degree, verbose=verbose)
    homogenization_analysis.apply_dirichlet_for_homogenization()
    homogenization_analysis.periodic_boundary_condition()

    homogenization_analysis.solve_full_homogenization()

    if verbose > 0:
        homogenization_analysis.print_homogenized_matrix()
        homogenization_analysis.print_errors()

    mat_Sorthotropic = homogenization_analysis.get_S_orthotropic()

    return mat_Sorthotropic, homogenization_analysis
