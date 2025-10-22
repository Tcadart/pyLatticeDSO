# =============================================================================
# CLASS: LatticeSim
#
# DESCRIPTION:
# This class extends the Lattice class to include simulation capabilities, beam modeling and
# domain decomposition methods.
# =============================================================================

from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from colorama import Fore, Style
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, splu, spilu
from sklearn.neighbors import NearestNeighbors

from pyLattice.beam import Beam
from pyLattice.cell import Cell
from pyLattice.lattice import Lattice
from pyLattice.utils import open_lattice_parameters
from pyLatticeSim.greedy_algorithm import load_reduced_basis
from pyLatticeSim.utils_rbf import ThinPlateSplineRBF
from pyLatticeSim.utils_schur import get_schur_complement
from pyLatticeSim.conjugate_gradient_solver import conjugate_gradient_solver
from pyLatticeSim.utils_simulation import solve_FEM_cell

if TYPE_CHECKING:
    from data.inputs.mesh_file.mesh_trimmer import MeshTrimmer

from pyLattice.timing import timing


class LatticeSim(Lattice):
    def __init__(self, name_file: str, mesh_trimmer: "MeshTrimmer" = None, verbose: int = 0,
                 enable_domain_decomposition_solver: bool = False):
        super().__init__(name_file, mesh_trimmer, verbose)
        self.domain_decomposition_solver = enable_domain_decomposition_solver

        self._simulation_flag = True

        self.enable_periodicity = None  # Warning not working for graded structures
        self.boundary_conditions = None
        self.enable_simulation_properties = None
        self.material_name = None
        self.number_iteration_max = None
        self.enable_preconditioner = None
        self.preconditioner_type = None
        self.used_schur_preconditioner = None
        self.type_schur_complement_computation = None
        self.precision_greedy: float | None = None
        self.radial_basis_function = None
        self.shape_schur_complement = None
        self._parameters_define = False

        self.define_simulation_parameters(name_file)
        assert self.material_name is not None, "Material name_lattice must be defined for simulation properties."

        self.free_DOF = None  # Free DOF gradient conjugate gradient method
        self.max_index_boundary = None
        self.global_displacement_index = None
        self.n_DOF_per_node: int = 6  # Number of DOF per node (3 translation + 3 rotation)
        self.penalization_coefficient: float = 1.5  # Fixed with previous optimization
        self.surrogate_model_implemented = ["exact", "FE2", "nearest_neighbor", "linear", "RBF"]
        self.enable_gradient_computing = False # Enable gradient computing for optimization
        self.is_penalized = False # True if the lattice has penalized beams at nodes
        self.neigh_function = None # Nearest neighbor function for nearest neighbor surrogate model

        self.define_connected_beams_for_all_nodes()
        if self.enable_simulation_properties and (not self.domain_decomposition_solver or
                                                  self.type_schur_complement_computation == "exact"):
            self.define_angles_between_beams()
            self.set_penalized_beams()
        # Define global indexation
        self.define_node_index_boundary()
        self.define_node_local_tags()
        self.set_boundary_conditions()
        self.are_cells_identical()

        if self.domain_decomposition_solver:
            if not self.type_schur_complement_computation in ["exact", "FE2"]:
                self.reduce_basis_dict = load_reduced_basis(self, self.precision_greedy)
                self.alpha_coefficients_greedy = self.reduce_basis_dict["alpha_ortho"].T
                if self.type_schur_complement_computation == "nearest_neighbor":
                    self.neigh_function = NearestNeighbors(n_neighbors=1, algorithm='auto')
                    self.neigh_function.fit(self.reduce_basis_dict["list_elements"])

            if self.enable_preconditioner:
                self._define_preconditioner_approximation()

            self.calculate_schur_complement_cells()

            self.preconditioner = None
            self.iteration = 0
            self.residuals = []

    @classmethod
    def open_pickle_lattice(cls, file_name: str = "LatticeObject", sim_config: str | None = None) -> "LatticeSim":
        return super(LatticeSim, cls).open_pickle_lattice(file_name=file_name, sim_config=sim_config)

    def _post_load_init(self, sim_config: str | None = None) -> None:
        """Finalize a pickled Lattice upgraded to LatticeSim without regenerating geometry."""
        self._simulation_flag = True
        # sensible defaults
        self.domain_decomposition_solver = getattr(self, "domain_decomposition_solver", False)
        self.enable_periodicity = None
        self.boundary_conditions = None
        self.enable_simulation_properties = None
        self.material_name = getattr(self, "material_name", None)
        self.number_iteration_max = getattr(self, "number_iteration_max", None)
        self.enable_preconditioner = getattr(self, "enable_preconditioner", None)
        self.preconditioner_type = getattr(self, "preconditioner_type", None)
        self.type_schur_complement_computation = getattr(self, "type_schur_complement_computation", "exact")
        self.precision_greedy = getattr(self, "precision_greedy", None)
        self.radial_basis_function = getattr(self, "radial_basis_function", None)
        self.shape_schur_complement = getattr(self, "shape_schur_complement", None)
        self._parameters_define = getattr(self, "_parameters_define", False)

        # load simulation parameters from a JSON if provided
        if sim_config is not None:
            self.define_simulation_parameters(sim_config)
            assert self.material_name is not None, "Material name_lattice must be defined for simulation properties."

        # rebuild simulation-dependent structures
        self.define_connected_beams_for_all_nodes()
        if not self.domain_decomposition_solver or self.type_schur_complement_computation == "exact":
            self.define_angles_between_beams()
            self.set_penalized_beams()

        self.define_node_index_boundary()
        self.define_node_local_tags()
        self.set_boundary_conditions()
        self.are_cells_identical()

        if self.domain_decomposition_solver:
            if self.type_schur_complement_computation not in ["exact", "FE2"]:
                self.reduce_basis_dict = load_reduced_basis(self, self.precision_greedy)
                self.alpha_coefficients_greedy = self.reduce_basis_dict["alpha_ortho"].T
            self.calculate_schur_complement_cells()
            self.preconditioner = None
            self.iteration = 0
            self.residuals = []

    @staticmethod
    def _sorted_nodes(nodes, ndigits: int = 9):
        """Deterministic ordering for any node iterable (kills randomness from set iteration)."""
        return sorted(
            nodes,
            key=lambda n: (round(n.x, ndigits), round(n.y, ndigits), round(n.z, ndigits), getattr(n, "index", -1)),
        )

    def define_simulation_parameters(self, name_file: str):
        """
        Define simulation parameters from the input file.

        Parameters
        ----------
        name_file : str
            Name of the input file
        """
        lattice_parameters = open_lattice_parameters(name_file)

        sim_params = lattice_parameters.get("simulation_parameters", {})
        self.enable_simulation_properties = bool(sim_params.get("enable", False))
        self.material_name = sim_params.get("material", "VeroClear")
        self.enable_periodicity = sim_params.get("periodicity", False)
        DDM_parameters = sim_params.get("DDM", None)
        if DDM_parameters is not None:
            self.enable_preconditioner = DDM_parameters.get("enable_preconditioner", False)
            self.preconditioner_type = DDM_parameters.get("preconditioner_type", None)
            if self.preconditioner_type is None and self.enable_preconditioner:
                raise ValueError("Preconditioner type must be defined in the input file.")
            self.number_iteration_max = DDM_parameters.get("max_iterations", 1000)
            if self.enable_preconditioner is not None and self.number_iteration_max is not None:
                self._parameters_define = True
            schur_complement_computation = DDM_parameters.get("schur_complement_computation", None)
            if schur_complement_computation is not None:
                self.type_schur_complement_computation = schur_complement_computation.get("type", None)
                if not self.type_schur_complement_computation in ["exact", "FE2"]:
                    self.precision_greedy = schur_complement_computation.get("precision_greedy", None)
                    if self.precision_greedy is None:
                        raise ValueError("Precision for greedy algorithm must be defined in the input file.")
            else:
                raise ValueError("Schur complement computation method must be defined in the input file.")
        else:
            if self.domain_decomposition_solver:
                raise ValueError("Schur complement computation method must be defined in the input file.")

        self.boundary_conditions = lattice_parameters.get("boundary_conditions", {})

    # =============================================================================
    # SECTION: Beam penalization Methods
    # =============================================================================
    @timing.category("simulation")
    @timing.timeit
    def set_penalized_beams(self) -> None:
        """
        Set penalization on beams at nodes based on L_zone values defined at beam endpoints.
        This method allows for better simulation of beam junctions by introducing penalized segments.
        """
        for cell in self.cells:
            beams_to_remove = []
            beams_to_add = []
            points_to_add = []
            for beam in list(cell.beams_cell):
                L1 = beam.angle_point_1.get("L_zone", 0)
                L2 = beam.angle_point_2.get("L_zone", 0)
                # No modification if both are zero or negative
                if L1 <= 0 and L2 <= 0:
                    continue

                # Start from original endpoints
                start = beam.point1
                end = beam.point2

                # Left/end-1 modification
                if L1 > 0:
                    pointExt1 = beam.get_point_on_beam_at_distance(L1, 1)
                    pointExt1.node_mod = True
                    points_to_add.append(pointExt1)
                    b1 = Beam(start, pointExt1, beam.radius, beam.material, beam.type_beam, beam.cell_belongings)
                    b1.set_beam_mod()
                    beams_to_add.append(b1)
                    start = pointExt1  # middle starts here

                # Right/end-2 modification
                if L2 > 0:
                    pointExt2 = beam.get_point_on_beam_at_distance(L2, 2)
                    pointExt2.node_mod = True
                    points_to_add.append(pointExt2)
                    mid_end = pointExt2
                else:
                    mid_end = end

                # Middle (unmodified) segment
                b_mid = Beam(start, mid_end, beam.radius, beam.material, beam.type_beam, beam.cell_belongings)
                beams_to_add.append(b_mid)

                # Final penalized segment at end-2 if needed
                if L2 > 0:
                    b3 = Beam(mid_end, end, beam.radius, beam.material, beam.type_beam, beam.cell_belongings)
                    b3.set_beam_mod()
                    beams_to_add.append(b3)

                if L1 > 0:
                    b_mid.associated_beams_mod.append(b1)
                if L2 > 0:
                    b_mid.associated_beams_mod.append(b3)

                beams_to_remove.append(beam)

            cell.add_beam(beams_to_add)
            cell.add_point(points_to_add)
            cell.remove_beam(beams_to_remove)

        # Update index
        self.define_beam_node_index()
        self.is_penalized = True
        print(Fore.GREEN + "Lattice penalization applied to beams at nodes." + Style.RESET_ALL)


    @timing.category("simulation")
    @timing.timeit
    def reset_penalized_beams(self) -> None:
        """
        Revert the penalization modifications made by set_penalized_beams.
        This involves:
        - Rewiring beams to connect original nodes directly, bypassing penalized segments.
        - Removing penalized beams and any orphaned nodes.
        Note: This operation is only valid if the lattice has been previously penalized.
        """
        if not self.is_penalized:
            print(Fore.YELLOW + "Warning: lattice does not appear to be penalized." + Style.RESET_ALL)
            return

        def _ensure_set(x):
            if isinstance(x, set):
                return x
            return set(x) if x is not None else set()

        def _remove_beams(cell, beams):
            if not beams:
                return
            cell.remove_beam(list(beams))

        def _remove_points(cell, pts):
            if not pts:
                return
            cell.remove_point(list(pts))

        def _rewire_end(bmid, old_node, new_node):
            """Replace old_node by new_node at one end of bmid."""
            if old_node is new_node:
                return
            old_node.connected_beams = _ensure_set(getattr(old_node, "connected_beams", None))
            old_node.connected_beams.discard(bmid)

            if bmid.point1 is old_node:
                bmid.point1 = new_node
            elif bmid.point2 is old_node:
                bmid.point2 = new_node

            new_node.connected_beams = _ensure_set(getattr(new_node, "connected_beams", None))
            new_node.connected_beams.add(bmid)

        for cell in self.cells:
            beams = list(cell.beams_cell)

            beams_to_delete = set()
            orphan_points = set()

            for bmid in beams:
                if getattr(bmid, "beam_mod", False):
                    continue

                assoc = getattr(bmid, "associated_beams_mod", None)
                if not assoc:
                    continue

                left_pen, right_pen = None, None
                for b in assoc:
                    if b.point1 is bmid.point1 or b.point2 is bmid.point1:
                        left_pen = b
                    elif b.point1 is bmid.point2 or b.point2 is bmid.point2:
                        right_pen = b

                if left_pen is not None:
                    new_start = left_pen.point1 if left_pen.point2 is bmid.point1 else left_pen.point2
                    _rewire_end(bmid, bmid.point1, new_start)
                    beams_to_delete.add(left_pen)

                if right_pen is not None:
                    new_end = right_pen.point1 if right_pen.point2 is bmid.point2 else right_pen.point2
                    _rewire_end(bmid, bmid.point2, new_end)
                    beams_to_delete.add(right_pen)

            _remove_beams(cell, beams_to_delete)

            for p in list(cell.points_cell):
                if getattr(p, "node_mod", False):
                    p.connected_beams = _ensure_set(getattr(p, "connected_beams", None))
                    if len(p.connected_beams) == 0:
                        orphan_points.add(p)

            _remove_points(cell, orphan_points)

        self._refresh_nodes_and_beams()
        self.define_beam_node_index()
        self.is_penalized = False
        print(Fore.GREEN + "Lattice penalization reverted." + Style.RESET_ALL)

# =============================================================================
# SECTION: Boundary condition Methods
# =============================================================================

    def apply_constraints_nodes(self, surfaces: list[str], value: list[float], DOF: list[int],
                                type_constraint: str = "Displacement", surface_cells: list[str] = None) -> None:
        """
        Apply boundary conditions to the lattice

        Parameters:
        -----------
        surfaces: list[str]
            List of surfaces to apply constraint (e.g., ["Xmin", "Xmax", "Ymin"])

        value: list of float
            Values to apply to the constraint

        DOF: list of int
            Degree of freedom to apply constraint (0: x, 1: y, 2: z, 3: Rx, 4: Ry, 5: Rz)

        type_beam: str
            Type of constraint (Displacement, Force)

        surface_cells: list[str], optional
            List of surfaces to find points on cells (e.g., ["Xmin", "Xmax", "Ymin"]). If None, uses surfaceNames.
        """
        pointSet = self.find_point_on_lattice_surface(surfaces, surface_cells)
        indexBoundaryList = {p.index_boundary for p in pointSet}
        if not indexBoundaryList:
            raise ValueError("No nodes found on the specified surfaces for constraint application.")

        # Count how many TARGET boundary nodes are FREE on each requested DOF
        # (so a total surface force is split only over DOFs that actually receive it)
        targets_per_dof = {d: 0 for d in DOF}
        seen = set()
        for cell in self.cells:
            for node in cell.points_cell:
                ib = node.index_boundary
                if ib is None or ib not in indexBoundaryList or ib in seen:
                    continue
                for d in DOF:
                    if node.fixed_DOF[d] == 0:
                        targets_per_dof[d] += 1
                seen.add(ib)

        for cell in self.cells:
            for node in cell.points_cell:
                ib = node.index_boundary
                if ib in indexBoundaryList:
                    for val, d in zip(value, DOF):
                        if type_constraint == "Displacement":
                            node.displacement_vector[d] = val
                            node.fix_DOF([d])
                        elif type_constraint == "Force":
                            n_tgt = max(1, targets_per_dof[d])
                            node.applied_force[d] = val / n_tgt
                        else:
                            raise ValueError("Invalid type of constraint. Use 'Displacement' or 'Force'.")

    def set_boundary_conditions(self) -> None:
        """
        Set boundary conditions on the lattice.
        """
        def check_data_boundary_condition_validity(data_dict_valid: dict) -> None:
            """
            Check if the data of the boundary condition is valid
            """
            if "Surface" not in data_dict_valid or "Value" not in data_dict_valid or "DOF" not in data_dict_valid:
                raise ValueError("Invalid boundary condition data. 'Surface', 'Value' and 'DOF' are required.")
            if not isinstance(data_dict_valid["Surface"], list):
                raise ValueError("Surface must be a list of strings.")
            if not isinstance(data_dict_valid["Value"], list):
                raise ValueError("Value must be a list of floats.")
            if not isinstance(data_dict_valid["DOF"], list):
                raise ValueError("DOF must be a list of strings.")
            if len(data_dict_valid["Value"]) != len(data_dict_valid["DOF"]):
                raise ValueError("Value and DOF must have the same length.")
            if not all(dof in ["X", "Y", "Z", "RX", "RY", "RZ"] for dof in data_dict_valid["DOF"]):
                raise ValueError("DOF must be one of 'X', 'Y', 'Z', 'RX', 'RY', 'RZ'.")
            if not all(surface in ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax", "Xmid", "Ymid", "Zmid"]
                       for surface in data_dict_valid["Surface"]):
                raise ValueError("Surface must be one of 'Xmin', 'Xmax', 'Ymin', 'Ymax', 'Zmin', 'Zmax', "
                                 "'Xmid', 'Ymid', 'Zmid'.")


        DOF_map = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}
        for key, dict_data in self.boundary_conditions.items():
            if key not in ["Force", "Displacement"]:
                raise ValueError(f"Invalid boundary condition type: {key}. Must be 'Force' or 'Displacement'.")
            for name_condition, data in dict_data.items():
                check_data_boundary_condition_validity(data)
                numeric_DOFs = [DOF_map[dof] for dof in data["DOF"]]
                surface_cells = data.get("SurfaceCells", None)
                self.apply_constraints_nodes(data["Surface"], data["Value"], numeric_DOFs, key, surface_cells)

    # =============================================================================
    # SECTION: Simulation core Methods
    # =============================================================================

    @timing.category("simulation")
    @timing.timeit
    def get_global_displacement(self, withFixed: bool = False, OnlyImposed: bool = False) \
            -> tuple[list[float], list[int]]:
        """
        Get global displacement of the lattice
        Parameters:
        -----------
        withFixed: bool
            If True, return displacement including fixed DOF
        OnlyImposed: bool
            If True, only return imposed displacement, else return all displacement

        Returns:
        --------
        globalDisplacement: dict
            Dictionary of global displacement with index_boundary as key and displacement vector as value
        globalDisplacementIndex: list
            List of boundary index per DOF (optional info)
        """
        globalDisplacement = []
        globalDisplacementIndex = []
        processed_nodes = set()
        for cell in self.cells:
            for node in self._sorted_nodes(cell.points_cell):
                if node.index_boundary is not None and node.index_boundary not in processed_nodes:
                    for i in range(6):
                        if node.fixed_DOF[i] == 0 and not OnlyImposed:
                            globalDisplacement.append(node.displacement_vector[i])
                            globalDisplacementIndex.append(node.index_boundary)
                        elif node.fixed_DOF[i] == 0 and node.applied_force[i] == 0:
                            globalDisplacement.append(0)
                        elif withFixed or OnlyImposed:
                            globalDisplacement.append(node.displacement_vector[i])
                            globalDisplacementIndex.append(node.index_boundary)
                    processed_nodes.add(node.index_boundary)
        if not OnlyImposed:
            self.global_displacement_index = globalDisplacementIndex
        if self._verbose > 2:
            print("globalDisplacement: ", globalDisplacement)
            print("global_displacement_index: ", globalDisplacementIndex)
        globalDisplacement = np.array(globalDisplacement)
        return globalDisplacement, globalDisplacementIndex

    @timing.category("simulation")
    @timing.timeit
    def define_node_index_boundary(self) -> None:
        """
        Define boundary tag for all boundary nodes and calculate the total number of boundary nodes
        """
        IndexCounter = 0
        nodeAlreadyIndexed = {}
        self.max_index_boundary = 0
        for cell in self.cells:
            for node in cell.points_cell:
                localTag = node.tag_point(cell.boundary_box)
                if localTag:
                    if node in nodeAlreadyIndexed:
                        node.index_boundary = nodeAlreadyIndexed[node]
                    else:
                        nodeAlreadyIndexed[node] = IndexCounter
                        node.index_boundary = IndexCounter
                        IndexCounter += 1
        self.max_index_boundary = IndexCounter - 1

    @timing.category("simulation")
    @timing.timeit
    def get_global_reaction_force(self, appliedForceAdded: bool = False) -> dict:
        """
        Get local reaction force of the lattice and sum if identical TagIndex

        Parameters:
        -----------
        appliedForceAdded: bool
            If True, add applied force to reaction force

        Returns:
        --------
        globalReactionForce: dict
            Dictionary of global reaction force with index_boundary as key and reaction force vector as value
        """
        globalReactionForce = {i: [0, 0, 0, 0, 0, 0] for i in range(self.max_index_boundary + 1)}
        for node in self.nodes:
            globalReactionForce[node.index_boundary] = node.reaction_force_vector
            if appliedForceAdded and sum(node.applied_force) > 0:
                for i in range(6):
                    if node.applied_force[i] != 0:
                        globalReactionForce[node.index_boundary][i] += node.applied_force[i]
        return globalReactionForce

    @timing.category("simulation")
    @timing.timeit
    def get_global_reaction_force_without_fixed_DOF(self, globalReactionForce: dict,
                                                    rightHandSide: bool = False) -> np.ndarray:
        """
        Get global reaction force of the lattice without fixed DOF

        Parameters:
        -----------
        globalReactionForce: dict
            Dictionary of global reaction force with index_boundary as key and reaction force vector as value

        rightHandSide: bool
            If True, get applied forces (right-hand side), else get reaction forces

        Returns:
        --------
        y: np.ndarray
            Array of global reaction force without fixed DOF
        """
        y = np.zeros(self.free_DOF, dtype=float)
        processed_nodes = set()

        for cell in self.cells:
            for node in cell.points_cell:
                if node.index_boundary is None or node.index_boundary in processed_nodes:
                    continue

                for i in range(6):
                    gi = node.global_free_DOF_index[i]
                    if gi is None:
                        continue

                    if rightHandSide:
                        # Strictly take the applied forces; zeros where none.
                        y[gi] = float(node.applied_force[i])
                    else:
                        # Reactions (e.g., due to imposed displacements)
                        y[gi] = float(globalReactionForce[node.index_boundary][i])

                processed_nodes.add(node.index_boundary)

        return y

    # =============================================================================
    # SECTION: Simulation initialization Methods
    # =============================================================================

    @timing.category("simulation")
    @timing.timeit
    def define_free_DOF(self):
        """
        Get total number of degrees of freedom in the lattice
        """
        self.free_DOF = 0
        processed_nodes = set()
        for cell in self.cells:
            for node in cell.points_cell:
                if node.index_boundary is not None and node.index_boundary not in processed_nodes:
                    self.free_DOF += node.fixed_DOF.count(0)
                    processed_nodes.add(node.index_boundary)

    @timing.category("simulation")
    @timing.timeit
    def set_global_free_DOF_index(self) -> None:
        """
        Set global free degree of freedom index for all nodes in boundary
        """
        counter = 0
        processed_nodes = {}
        for cell in self.cells:
            for node in cell.points_cell:
                if node.index_boundary is not None:
                    if node.index_boundary not in processed_nodes.keys():
                        for i in np.where(np.array(node.fixed_DOF) == 0)[0]:
                            node.global_free_DOF_index[i] = counter
                            counter += 1
                        processed_nodes[node.index_boundary] = node.global_free_DOF_index
                    else:
                        node.global_free_DOF_index[:] = processed_nodes[node.index_boundary]

    @timing.category("simulation")
    @timing.timeit
    def _initialize_reaction_force(self) -> None:
        """
        Initialize reaction force of all nodes to 0 on each DOF
        """
        for cell in self.cells:
            for node in cell.points_cell:
                node.initialize_reaction_force()

    @timing.category("simulation")
    @timing.timeit
    def _initialize_displacement(self) -> None:
        """
        Initialize displacement of all nodes to zero on each DOF
        """
        for cell in self.cells:
            for node in cell.points_cell:
                node.initialize_displacement()

    @timing.category("simulation")
    @timing.timeit
    def _initialize_simulation_parameters(self):
        """
        Initialize simulation parameters for each node in the lattice
        """
        for node in self.nodes:
            node.initialize_reaction_force()
            node.initialize_displacement()
        self.set_boundary_conditions()

    # =============================================================================
    # SECTION: DDM core Methods
    # =============================================================================

    @timing.category("simulation")
    @timing.timeit
    def get_global_displacement_DDM(self, OnlyImposed: bool = False) \
            -> tuple[list[float], list[int]]:
        """
        Get global displacement of the lattice

        Parameters:
        -----------
        OnlyImposed: bool
            If True, only return imposed displacement, else return all displacement

        Returns:
        --------
        x: list of float
            List of global displacement

        idx_list: list of int
            List of boundary index per DOF (optional info)
        """
        x = [0.0] * self.free_DOF
        idx_list = [None] * self.free_DOF  # boundary index per DOF (optional info)

        processed_nodes = set()
        for cell in self.cells:
            for node in cell.points_cell:
                if node.index_boundary is None or node.index_boundary in processed_nodes:
                    continue
                for i in range(6):
                    gi = node.global_free_DOF_index[i]
                    if gi is None:
                        continue
                    if not OnlyImposed:
                        x[gi] = node.displacement_vector[i]
                        idx_list[gi] = node.index_boundary
                    else:
                        # If OnlyImposed, keep imposed values where present, else 0
                        x[gi] = node.displacement_vector[i]
                        idx_list[gi] = node.index_boundary
                processed_nodes.add(node.index_boundary)

        self.global_displacement_index = idx_list
        if self._verbose > 2:
            print("globalDisplacement (canonically ordered): ", x)
            print("global_displacement_index (per DOF): ", idx_list)
        return x, idx_list

    @timing.category("simulation")
    @timing.timeit
    def evaluate_alphas_linear_surrogate(self, geometric_params: list[float]) -> np.ndarray:
        """
        Robust linear surrogate with safe extrapolation:
          • 1D: np.interp (fast, stable).
          • ND: LinearNDInterpolator inside convex hull; fallback to NearestNDInterpolator outside.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        mu = np.array(geometric_params, dtype=float).ravel()
        evalParams = np.asarray(self.reduce_basis_dict["list_elements"], dtype=float)
        alphaCoeffs = np.asarray(self.alpha_coefficients_greedy, dtype=float)

        # --- Ensure (N_points, n_alpha) orientation for values ---
        if alphaCoeffs.ndim == 1:
            alphaCoeffs = alphaCoeffs.reshape(-1, 1)
        if alphaCoeffs.shape[0] != evalParams.shape[0]:
            if alphaCoeffs.shape[1] == evalParams.shape[0]:
                alphaCoeffs = alphaCoeffs.T
            else:
                raise ValueError(
                    f"Incompatible shapes: points={evalParams.shape}, values={alphaCoeffs.shape}. "
                    "alphaCoeffs must be (N_points, n_alpha)."
                )

        N = evalParams.shape[0]

        # ---- 1D fast path ----
        if evalParams.ndim == 1 or (evalParams.ndim == 2 and evalParams.shape[1] == 1):
            x = float(mu[0])
            x_grid = evalParams.ravel()
            idx_sorted = np.argsort(x_grid)
            x_sorted = x_grid[idx_sorted]
            A_sorted = alphaCoeffs[idx_sorted]  # (N, k)
            # np.interp works per scalar -> interpolate each column
            y = np.vstack([np.interp(x, x_sorted, A_sorted[:, j],
                                     left=A_sorted[0, j], right=A_sorted[-1, j])
                           for j in range(A_sorted.shape[1])]).T
            return y.squeeze()

        # ---- ND general case ----
        # Cache interpolators to avoid rebuilding every call
        if not hasattr(self, "_alpha_lin_nd") or not hasattr(self, "_alpha_nn_nd"):
            print(evalParams.shape, alphaCoeffs.shape)
            self._alpha_lin_nd = LinearNDInterpolator(evalParams, alphaCoeffs)  # NaN outside hull
            self._alpha_nn_nd = NearestNDInterpolator(evalParams, alphaCoeffs)  # nearest neighbor fallback

        y = self._alpha_lin_nd(mu)
        if y is None or np.any(np.isnan(y)):
            # Outside convex hull -> robust fallback
            y = self._alpha_nn_nd(mu)
            if getattr(self, "_verbose", 0) > 0:
                print(f"⚠️ Extrapolation (nearest-neighbor) used for mu={mu}.")
        return np.asarray(y).squeeze()

    def _define_radial_basis_functions(self):
        mu_train = np.array(self.reduce_basis_dict["list_elements"])  # shape (N, d)
        alpha_train = np.array(self.alpha_coefficients_greedy)  # shape (N, m)
        self.radial_basis_function = ThinPlateSplineRBF(mu_train, alpha_train)

    def _check_parameters_defined(self):
        if not self._parameters_define:
            print(Fore.YELLOW + "You can define parameters with define_parameters method."+ Style.RESET_ALL)
            print(Fore.YELLOW + "Default parameters are used. (Preconditioner activated and 1000 max iterations"+
                  Style.RESET_ALL)
            self.define_parameters()

    def define_parameters(self, enable_precondioner: bool = True, numberIterationMax: int = 1000):
        """
        Define parameters for the domain decomposition solver.

        Parameters:
        enable_preconditioner: bool
            Enable the preconditioner for the conjugate gradient solver.
        number_iteration_max: int
            Maximum number of iterations for the conjugate gradient solver.
        """
        self.enable_preconditioner = enable_precondioner
        self.number_iteration_max = numberIterationMax
        self._parameters_define = True

    # =============================================================================
    # SUBSECTION: Schur Complement Methods
    # =============================================================================

    def build_coupling_operator_cells(self) -> None:
        """
        Build coupling operator for each cell in the lattice
        """
        for cell in self.cells:
            cell.build_coupling_operator(self.free_DOF)

    def calculate_schur_complement_cells(self):
        """
        Calculate the Schur complement for each cell in the lattice.
        Batch all unique radius cases per geometry and compute them at once.
        """
        schur_cache: dict = {}
        nb_computed = 0

        # 1) Group cells by (geom_key, radius_key)
        groups: dict = {}
        for cell in self.cells:
            geom_key = tuple(cell.geom_types) if isinstance(cell.geom_types, list) else cell.geom_types
            radius_key = tuple(round(float(r), 8) for r in cell.radii)

            if geom_key not in groups:
                groups[geom_key] = {}
            groups[geom_key].setdefault(radius_key, []).append(cell)

        # 2) For each geometry, batch-compute missing Schur complements
        for geom_key, radius_map in groups.items():
            if geom_key not in schur_cache:
                schur_cache[geom_key] = {}

            # Find which radius sets are not cached yet
            missing_keys = [rk for rk in radius_map.keys() if rk not in schur_cache[geom_key]]
            if missing_keys:
                # Compute missing Schur complements
                if self.type_schur_complement_computation in ["exact", "FE2"]:
                    # No batch available -> compute one by one
                    for rk in missing_keys:
                        nb_computed += 1
                        # Using the first cell with this (geom, radii) to get index
                        ref_cell = radius_map[rk][0]
                        S = get_schur_complement(self, ref_cell.index)
                        dS_list = None
                        if self.enable_gradient_computing:
                            dS_list = self._compute_schur_gradients(ref_cell, list(rk))
                        schur_cache[geom_key][rk] = {"S": S, "dS": dS_list}
                        if self._verbose > 1:
                            print(
                                f"Schur complement (+ grads) computed (exact/FE2) for geom {geom_key} with radii {rk}.")
                elif self.type_schur_complement_computation in self.surrogate_model_implemented:
                    # Batch surrogate evaluation per geometry
                    nb_computed += len(missing_keys)
                    radii_batch = [list(rk) for rk in missing_keys]
                    S_batch = self.get_schur_complement_from_reduced_basis_batch(radii_batch)  # (n_q, n, n)

                    for rk, S in zip(missing_keys, S_batch):
                        dS_list = None
                        if self.enable_gradient_computing:
                            if self.type_schur_complement_computation == "RBF":
                                dS_list = self._compute_schur_gradients_RBF(list(rk))
                            else:
                                # fallback FD for other surrogates
                                ref_cell = radius_map[rk][0]
                                dS_list = self._compute_schur_gradients(ref_cell, list(rk))

                        schur_cache[geom_key][rk] = {"S": S, "dS": dS_list}
                        if self._verbose > 1:
                            print(
                                f"Schur complement (+ grads) computed (batch surrogate) for geom {geom_key} with radii {rk}.")
                else:
                    raise NotImplementedError("Not implemented schur complement computation method.")

            # 3) Assign cached results to all cells
            for rk, cells in radius_map.items():
                S = schur_cache[geom_key][rk]["S"]
                dS_list = schur_cache[geom_key][rk]["dS"]
                for c in cells:
                    c.schur_complement = S
                    c.schur_complement_gradient = dS_list

        if self._verbose > 1:
            print("Number of unique Schur complements computed:", nb_computed)

    def get_schur_complement_from_reduced_basis_batch(self, geometric_params_list: list[list[float]]) -> np.ndarray:
        """
        Vectorized version: compute Schur complements for many queries in one GEMM (faster than many GEMVs).

        Parameters
        ----------
        geometric_params_list : list of list of float
            Each inner list is a set of geometric parameters for one query.

        Returns
        -------
        schur_batch : np.ndarray
            Array of shape (n_queries, n_schur, n_schur) with Fortran-ordered per-matrix layout.
        """
        # 1) Build all alpha vectors and stack them as columns -> (n_alpha, n_queries)
        alphas_list = []
        t = self.type_schur_complement_computation

        if t == "nearest_neighbor":
            Xq = np.asarray(geometric_params_list, dtype=float)
            _, indices = self.neigh_function.kneighbors(Xq)
            alphas_list = [self.alpha_coefficients_greedy[i0] for i0 in indices[:, 0]]

        elif t == "linear":
            alphas_list = [self.evaluate_alphas_linear_surrogate(gp) for gp in geometric_params_list]

        elif t == "RBF":
            if self.radial_basis_function is None:
                self._define_radial_basis_functions()
            Xq = np.asarray(geometric_params_list, dtype=float)
            alphas_list = self.radial_basis_function.evaluate(Xq).squeeze()
            if alphas_list.ndim == 1:
                alphas_list = alphas_list[None, :]  # (1, n_queries) if degenerate

            # convert to list of 1D arrays for consistency below
            alphas_list = [alphas_list[i, :] for i in range(alphas_list.shape[0])]

        else:
            raise NotImplementedError("Not implemented schur complement computation method.")

        A = np.column_stack([np.asarray(a, dtype=float) for a in alphas_list])  # (n_alpha, n_queries)

        # 2) Single GEMM: (n_schur^2, n_alpha) @ (n_alpha, n_queries) -> (n_schur^2, n_queries)
        basis = self.reduce_basis_dict["basis_reduced_ortho"]
        basisF = np.asfortranarray(basis)  # BLAS-friendly
        S_flat = basisF @ A  # GEMM, fast path

        # 3) Reshape each column to (n_schur, n_schur) with Fortran order
        if self.shape_schur_complement is None:
            self.shape_schur_complement = int(sqrt(S_flat.shape[0]))
        n = self.shape_schur_complement
        n_q = S_flat.shape[1]

        schur_batch = np.empty((n_q, n, n), dtype=S_flat.dtype, order="C")
        for j in range(n_q):
            schur_batch[j] = S_flat[:, j].reshape((n, n), order="F")

        return schur_batch

    def get_schur_complement_from_reduced_basis(self, geometric_params: list[float]) -> np.ndarray:
        """
        Get the Schur complement from the reduced basis with a giver surrogate method

        Parameters:
        -----------
        geometric_params: list of float
            List of geometric parameters to define the Schur complement

        Returns:
        -----------
        schur_complement_approx: np.ndarray
            Approximated Schur complement
        """
        import time
        time_schur = time.time()
        # Evaluate alpha coefficients based on the chosen surrogate method
        if self.type_schur_complement_computation == "nearest_neighbor":
            distances, indices = self.neigh_function.kneighbors(np.array(geometric_params).reshape(1, -1))
            alphas = self.alpha_coefficients_greedy[indices[0][0]]
        elif self.type_schur_complement_computation == "linear":
            alphas = self.evaluate_alphas_linear_surrogate(geometric_params)
        elif self.type_schur_complement_computation == "RBF":
            if self.radial_basis_function is None:
                self._define_radial_basis_functions()
            alphas = self.radial_basis_function.evaluate(np.array(geometric_params).reshape(1, -1)).squeeze()
        else:
            raise NotImplementedError("Not implemented schur complement computation method.")
        print("Time to get alphas:", time.time() - time_schur)
        # Reconstruct Schur complement
        schur_complement_approx = self.reduce_basis_dict["basis_reduced_ortho"] @ alphas
        print("Time to get Schur:", time.time() - time_schur)
        if self.shape_schur_complement is None:
            self.shape_schur_complement = int(sqrt(schur_complement_approx.shape[0]))
        schur_complement_approx_reshape = schur_complement_approx.reshape(
            (self.shape_schur_complement, self.shape_schur_complement), order='F')
        print("Time to get Schur reshaped:", time.time() - time_schur)

        return schur_complement_approx_reshape

    def _compute_schur_gradients(self, cell: "Cell", radii_params: list[float]) -> list[np.ndarray]:
        """
        Central finite-difference of the Schur complement w.r.t. each radius parameter.

        Parameters
        ----------
        cell : Cell
            The target cell.
        radii_params : list[float]
            Current radii for this cell's beam types.

        Returns
        -------
        list[np.ndarray]
            dS/dr_j for each radius parameter j, same shape as S_base.
        """
        grads = []
        eps_rel = 1e-6

        for j, rj in enumerate(radii_params):
            h = max(1e-8, eps_rel * max(1.0, abs(rj)))

            rp = radii_params.copy()
            rm = radii_params.copy()
            rp[j] = rj + h
            rm[j] = max(1e-12, rj - h)  # keep positive

            S_p = self._schur_for_params(cell, rp)
            S_m = self._schur_for_params(cell, rm)

            dS = (S_p - S_m) / (rp[j] - rm[j])

            grads.append(dS)

        return grads

    def _compute_schur_gradients_RBF(self, radii_params: list[float]) -> list[np.ndarray]:
        """
        Compute the gradients of the Schur complement efficiently using RBF interpolation class.

        Parameters
        ----------
        radii_params : list[float]
            Current radii for this cell's beam types.
        """
        grad_alpha = self.radial_basis_function.gradient(radii_params)  # (d, m)

        B = np.asarray(self.reduce_basis_dict["basis_reduced_ortho"], float)  # (nS, m)
        nS = B.shape[0]
        # taille carrée de S
        shapeS = self.shape_schur_complement
        if shapeS is None:
            shapeS = int(np.sqrt(nS))
            self.shape_schur_complement = shapeS

        grads_rbf = []
        for j in range(len(radii_params)):
            dalpha = np.asarray(grad_alpha[j], float)  # (m,)
            dS_vec = B @ dalpha  # (nS,)
            dS_mat = dS_vec.reshape((shapeS, shapeS), order='F')  # même ordre que pour S
            grads_rbf.append(dS_mat)

        return grads_rbf

    def _schur_for_params(self, cell: "Cell", radii_params: list[float]) -> np.ndarray:
        """
        Helper: evaluate the Schur complement for a given set of radii parameters
        using the currently selected computation mode.
        """
        if self.type_schur_complement_computation == "exact":
            # Temporarily modify cell radii, evaluate, then restore
            old_radii = list(cell.radii)
            try:
                cell.change_beam_radius(list(radii_params))
                S = get_schur_complement(self, cell.index)
            finally:
                cell.change_beam_radius(old_radii)
            return S

        elif self.type_schur_complement_computation in self.surrogate_model_implemented:
            return self.get_schur_complement_from_reduced_basis(list(radii_params))

        else:
            raise NotImplementedError("Not implemented schur complement computation method.")

    # =============================================================================
    # SUBSECTION: DDM Core Methods
    # =============================================================================

    @timing.category("simulation")
    @timing.timeit
    def solve_DDM(self):
        """
        Solve the problem with the domain decomposition method.
        """
        import time
        start_time = time.time()
        self._check_parameters_defined()

        # Free DOF
        self.define_free_DOF()
        if self.free_DOF == 0:
            raise ValueError(Fore.RED + "No free DOF in the lattice. Process aborted."+ Style.RESET_ALL)
        if self._verbose > 0:
            print(Fore.GREEN + "Free DOF", self.free_DOF, Style.RESET_ALL)

        self.set_global_free_DOF_index()

        # Calculate b
        if self._verbose > -1:
            print(Fore.GREEN + "Assemble right-hand side" + Style.RESET_ALL)

        # Reactions induced by imposed (Dirichlet) displacements on the boundary
        r_free = self.calculate_reaction_force_global(np.zeros(self.free_DOF, dtype=float), rightHandSide=False)

        # External applied forces on free DOFs
        f_free = self.get_global_reaction_force_without_fixed_DOF(
            self.get_global_reaction_force(appliedForceAdded=True), rightHandSide=True)

        b = f_free - r_free
        if np.linalg.norm(b) == 0:
            print(Fore.YELLOW + "No external forces or imposed displacements in the lattice. Process aborted."+
                  Style.RESET_ALL)
            return None, None, None, None

        # Initialize local displacement to zero
        self._initialize_displacement()

        # Define the preconditioner
        self.define_preconditioner()

        A_operator = LinearOperator(shape=(self.free_DOF, self.free_DOF),
                                    matvec=self.calculate_reaction_force_global)

        print(Fore.GREEN + "Conjugate Gradient started."+ Style.RESET_ALL)

        tol = 1e-6
        mintol = 1e-12
        restart_every = 500000
        alpha_max = 100
        xsol, info = conjugate_gradient_solver(A_operator, b, M = self.preconditioner, maxiter=self.number_iteration_max,
                                               tol = tol, mintol = mintol, restart_every = restart_every, alpha_max= alpha_max,
                                               callback=lambda xk: self.cg_progress(xk, b, A_operator))
        print(f"Conjugate Gradient finished in {time.time() - start_time:.2f} seconds.")
        if self._verbose > -1:
            if info == 0:
                print(Fore.GREEN + "Conjugate Gradient converged."+ Style.RESET_ALL)
            else:
                print(Fore.RED + "Conjugate Gradient did not converge."+ Style.RESET_ALL)

        self.update_reaction_force_each_cell(xsol)

        # Reset boundary conditions
        self.set_boundary_conditions()

        xsol, globalDisplacementIndex = self.get_global_displacement()
        return xsol, info, self.global_displacement_index, b

    @timing.category("simulation")
    @timing.timeit
    def calculate_reaction_force_global(self, globalDisplacement, rightHandSide:bool = False):
        """
        Calculate the global reaction force of the lattice

        Parameters:
        -----------
        globalDisplacement: np.ndarray
            The global displacement vector.
        rightHandSide: bool
            If True, get applied forces (right-hand side), else get reaction forces.
        """
        # Calculate RF local
        self.update_reaction_force_each_cell(globalDisplacement)

        # Calculate the RF global
        globalReactionForce = self.get_global_reaction_force()

        globalReactionForceWithoutFixedDOF = self.get_global_reaction_force_without_fixed_DOF(globalReactionForce,
                                                                                              rightHandSide)

        return globalReactionForceWithoutFixedDOF

    @timing.category("simulation")
    @timing.timeit
    def update_reaction_force_each_cell(self, global_displacement):
        """
        Update RF local

        Parameters:
        -----------
        global_displacement: np.ndarray
            The global displacement vector.
        """
        self._initialize_reaction_force()
        datasetDataCell = []
        for cell in self.cells:
            # Set Displacement on nodes (Global to Local)
            cell.set_displacement_at_boundary_nodes(global_displacement)

            reaction_force_cell = self.solve_sub_problem(cell)

            # Update the RF local in the cell
            cell.set_reaction_force_on_nodes(reaction_force_cell)
        return datasetDataCell

    @timing.category("simulation")
    @timing.timeit
    def solve_sub_problem(self, cell):
        """
        Solve the subproblem on a cell to get the reaction force
        """
        if cell.node_in_order_simulation is None:
            cell.define_node_order_to_simulate()
        if self._verbose > 1:
            print("self.node_in_order", cell.node_in_order_simulation)

        # Check displacement null
        displacement_cell = cell.get_displacement_at_nodes(cell.node_in_order_simulation)
        if self.type_schur_complement_computation != "FE2":
            if np.sum(displacement_cell) != 0:
                # Solve the local problem
                displacement = np.array(displacement_cell).flatten()
                reaction_force_cell = np.dot(cell.schur_complement, displacement)
                reaction_force_cell = reaction_force_cell.reshape(-1, 6)
            else:
                # If displacement is null, set reaction force to zero
                reaction_force_cell = displacement_cell
                if self._verbose > 1:
                    print("Displacement is null")
        else:
            simulation_model = solve_FEM_cell(self, cell)
            reaction_force_cell = simulation_model.calculate_reaction_force_and_moment_all_boundary_nodes(cell)[0]
        return reaction_force_cell

    @timing.category("simulation")
    @timing.timeit
    def cg_progress(self, xk, b, A_operator):
        """
        Callback function to track and plot progress of the CG method.

        Parameters:
        -----------
        xk: np.array
            The current solution vector at the k-th iteration.

        b: np.array
            The right-hand side vector of the system.

        A_operator: callable or matrix
            The operator or matrix for the system Ax = b.
        """
        if np.isnan(xk).any():
            raise ValueError(Fore.RED + "NaN detected in the Conjugate Gradient solution. Process aborted."+ Style.RESET_ALL)

        plotting = False
        # Increment iteration count
        self.iteration += 1

        if plotting:
            # Calculate the residual norm
            residual = b - A_operator @ xk
            residual_norm = np.linalg.norm(residual)

            # Append the residual norm for tracking
            self.residuals.append(residual_norm)
            # Save the data at the last iteration
            if residual_norm < 1e-8 or self.iteration == len(b):  # Last iteration condition
                save_file = "ConjugateGradientMethod/ProgressFile/progress_cg.txt"
                with open(save_file, "w") as f:
                    f.write("Conjugate Gradient Progress\n")
                    f.write("Iteration, Residual Norm\n")
                    for i, res in enumerate(self.residuals, 1):
                        f.write(f"{i}, {res:.6e}\n")
                print(f"Progress saved to {save_file}")
        else:
            if self._verbose > 0:
                print(Fore.BLUE + f"Iteration {self.iteration}" + Style.RESET_ALL)
            if self._verbose > 1:
                # Calculate the residual norm
                residual = b - A_operator @ xk
                residual_norm = np.linalg.norm(residual) / np.linalg.norm(b)

                # Append the residual norm for tracking
                self.residuals.append(residual_norm)
                print(f"Residual norm: {residual_norm:.6e}")

    # =============================================================================
    # SUBSECTION: Preconditioner Methods
    # =============================================================================

    @timing.category("preconditioner")
    @timing.timeit
    def _define_preconditioner_approximation(self):
        path_dataset_schur = Path(__file__).parents[2] / "data" / "outputs" / "schur_complement"
        geom_type_str = "_" + "_".join(str(gt) for gt in self.geom_types)

        if self.preconditioner_type == "mean":
            name_file = Path("Schur_complement_mean" + geom_type_str)
        elif self.preconditioner_type == "nearest_reference":
            name_file = Path("Schur_complement" + geom_type_str)
        elif self.preconditioner_type == "exact":
            return # no precomputed data needed
        else:
            raise NotImplementedError("Not implemented preconditioner approximation method.")

        if name_file.suffix.lower() != ".npz":
            name_file = name_file.with_suffix(".npz")

        path_file = path_dataset_schur / name_file
        self.used_schur_preconditioner = np.load(path_file, allow_pickle=True)

    @timing.category("preconditioner")
    @timing.timeit
    def define_preconditioner(self):
        """
        Define the preconditioner for the conjugate gradient solver.
        """
        if self.enable_preconditioner:
            if self._verbose > 0:
                print(Fore.GREEN + "Define the preconditioner" + Style.RESET_ALL)
            LUSchurComplement, inverseSchurComplement = self.build_preconditioner()

            if LUSchurComplement is not None:
                self.preconditioner = LinearOperator(shape = (self.free_DOF, self.free_DOF),
                                                matvec=lambda x: LUSchurComplement.solve(x))
            elif inverseSchurComplement is not None:
                self.preconditioner = LinearOperator(shape = (self.free_DOF, self.free_DOF),
                                                matvec=lambda x: inverseSchurComplement @ x)

    @timing.category("preconditioner")
    @timing.timeit
    def build_preconditioner(self):
        """
        Build a sparse LU/ILU preconditioner of the global Schur complement.
        Faster assembly by accumulating triplets and avoiding dense ops.
        """
        print(Fore.GREEN + "Build the preconditioner" + Style.RESET_ALL)
        self.build_coupling_operator_cells()

        # Fast assembly via triplet accumulation
        rows_acc, cols_acc, data_acc = [], [], []
        n = self.free_DOF
        if not self.preconditioner_type in ["mean", "nearest_reference"]:
            print(Fore.YELLOW + "Preconditioner exact is used." + Style.RESET_ALL)

        neigh = None
        if self.preconditioner_type == "nearest_reference":
            neigh = NearestNeighbors(n_neighbors=1, algorithm='auto')
            neigh.fit(self.used_schur_preconditioner["radius_values"])

        for cell in self.cells:
            if self.preconditioner_type == "mean":
                schur_matrices = self.used_schur_preconditioner["schur_matrices"]
            elif self.preconditioner_type == "nearest_reference":
                _, radii_nearest = neigh.kneighbors(np.array(cell.radii).reshape(1, -1))
                schur_matrices = self.used_schur_preconditioner["schur_matrices"][radii_nearest[0][0]]
            else:
                schur_matrices = cell.schur_complement

            local = cell.build_local_preconditioner(schur_matrices).tocoo()
            rows_acc.append(local.row)
            cols_acc.append(local.col)
            data_acc.append(local.data)

        if rows_acc:
            rows_all = np.concatenate(rows_acc)
            cols_all = np.concatenate(cols_acc)
            data_all = np.concatenate(data_acc)
        else:
            rows_all = cols_all = data_all = np.array([], dtype=int)

        global_schur_complement = coo_matrix((data_all, (rows_all, cols_all)), shape=(n, n))
        global_schur_complement.sum_duplicates()

        # Sanity checks
        csr_g = global_schur_complement.tocsr()
        zero_row_mask = np.diff(csr_g.indptr) == 0
        if np.any(zero_row_mask):
            print("Attention : There are some rows with all zeros in the Schur complement matrix.")

        # Factorization
        inverseSchurComplement = None

        try:
            LUSchurComplement = splu(global_schur_complement.tocsc())
            if self._verbose > 0:
                print("Using LU decomposition of the Schur complement matrix.")
        except RuntimeError:
            # If exact LU fails or is too ill-conditioned, try an ILU preconditioner
            # Adjust drop_tol/fill_factor if needed for robustness vs. speed.
            ilu = spilu(global_schur_complement.tocsc(), drop_tol=1e-4, fill_factor=10)
            LUSchurComplement = ilu  # returns an object with solve() as well
            if self._verbose > 0:
                print("Using ILU (spilu) preconditioner for the Schur complement matrix.")

        return LUSchurComplement, inverseSchurComplement

    # =============================================================================
    # SECTION: Utils Methods
    # =============================================================================

    def reset_cell_with_new_radii(self, new_radii: list[float], index_cell: int = 0) -> None:
        """
        Reset cell with new radii for each beam type
        WARNING: BUG for multiple geometry cells

        Parameters:
        ------------
        new_radii: list of float
            List of new radii for each beam type

        index_cell: int
            Index of the cell to reset
        """
        if len(new_radii) != len(self.radii):
            raise ValueError("Invalid hybrid radii data.")
        self.radii = new_radii
        if not (0 <= index_cell < len(self.cells)):
            raise IndexError("Invalid cell index.")

        # Delete only orphan beams and nodes
        cell_to_reset = self.cells[index_cell]
        other_cells = [c for i, c in enumerate(self.cells) if i != index_cell]
        beams_in_others = set().union(*(c.beams_cell for c in other_cells)) if other_cells else set()
        nodes_in_others = set().union(*(c.points_cell for c in other_cells)) if other_cells else set()

        orphan_beams = cell_to_reset.beams_cell - beams_in_others
        orphan_nodes = cell_to_reset.points_cell - nodes_in_others
        self.beams.difference_update(orphan_beams)
        self.nodes.difference_update(orphan_nodes)

        # Dispose the cell to free memory
        cell_to_reset.dispose()

        # Prepare already defined nodes and beams
        def _wkey_point(p, ndigits=9):
            return round(p.x, ndigits), round(p.y, ndigits), round(p.z, ndigits)

        nodes_already_defined = {_wkey_point(p): p for p in nodes_in_others}
        beams_already_defined = {}
        for b in beams_in_others:
            k1, k2 = _wkey_point(b.point1), _wkey_point(b.point2)
            beams_already_defined[tuple(sorted((k1, k2)))] = b

        # Create a new cell with the same position and size but new radii
        initial_cell_size = [self.cell_size_x, self.cell_size_y, self.cell_size_z]
        new_cell = Cell(
            cell_to_reset.pos, initial_cell_size, cell_to_reset.coordinate,
            self.geom_types, self.radii,
            self.grad_radius, self.grad_dim, self.grad_mat,
            self.uncertainty_node, self._verbose,
            beams_already_defined=beams_already_defined,
            nodes_already_defined=nodes_already_defined
        )

        # Update the cell in the lattice
        self.cells[index_cell] = new_cell
        self.beams.update(new_cell.beams_cell)
        self.nodes.update(new_cell.points_cell)

        # IMPORTANT for hybrid cells: rebuild split segments created by geometry collisions
        if len(self.geom_types) > 1:
            self.check_hybrid_collision()
            self._refresh_nodes_and_beams()

        # Refresh all data (after potential hybrid collision updates)
        self.define_beam_node_index()
        self.define_cell_index()
        self.define_cell_neighbours()
        self.define_connected_beams_for_all_nodes()
        self.apply_tag_all_point()
        if self._simulation_flag:
            self.define_angles_between_beams()
            self.set_penalized_beams()
            # Define global indexation
            self.define_node_index_boundary()
            self.define_node_local_tags()
            self.set_boundary_conditions()


    def _update_DDM_after_geometry_change(self):
        """
        Update the DDM after a change in the geometry of the lattice.
        """
        self.calculate_schur_complement_cells()

        self.preconditioner = None
        self.iteration = 0
        self.residuals = []

    def get_global_force_displacement_curve(self, dof: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate global force (sum of reactions) vs imposed displacement for comparison with experiment.

        Parameters
        ----------
        dof : int
            Degree of freedom to extract (default=2 for Z direction).

        Returns
        -------
        disp : np.ndarray
            Imposed displacement values (at the loading boundary nodes).

        force : np.ndarray
            Total reaction force corresponding to that displacement.
        """
        disp_list = []
        force_list = []

        seen = set()
        for cell in self.cells:
            for node in cell.points_cell:
                if node.index_boundary is None or node in seen:
                    continue

                # Only keep nodes with applied BC
                if any(node.applied_force) or any(node.fixed_DOF):
                    # Get imposed displacement in DOF
                    disp_list.append(node.displacement_vector[dof])
                    # Sum reaction force on that DOF
                    force_list.append(node.reaction_force_vector[dof])
                seen.add(node)

        disp = np.asarray(disp_list, dtype=float)

        force_arr = np.asarray(force_list, dtype=float)

        force = np.sum(abs(force_arr), axis=0)

        return disp, force


