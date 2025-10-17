# =============================================================================
# CLASS: Point
# =============================================================================
import math
import random
from typing import List, Tuple, Optional, TYPE_CHECKING

from .utils import _discard
from .timing import timing

if TYPE_CHECKING:
    from .cell import Cell

class Point:
    """
    Represents a point in 3D space with additional attributes for simulation.
    """

    def __init__(self, x: float, y: float, z: float, cell_belongings: List['Cell'],
                 node_uncertainty_SD: float = 0.0) -> None:
        """
        Initialize a point object.

        Parameters
        ----------
        x : float
            X-coordinate of the point.
        y : float
            Y-coordinate of the point.
        z : float
            Z-coordinate of the point.
        cell_belongings : List[Cell]
            List of cells that contain this point.
        node_uncertainty_SD : float, optional
            Standard deviation for node position uncertainty (default is 0.0).

        Raises
        ------
        ValueError
            If coordinates are not numeric or if node_uncertainty_SD is negative.
        """
        # Validate input types and values
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not isinstance(z, (int, float)):
            raise ValueError("Coordinates must be numeric (int or float).")
        if not isinstance(node_uncertainty_SD, (int, float)):
            raise ValueError("Node uncertainty standard deviation must be numeric (int or float).")
        if node_uncertainty_SD < 0:
            raise ValueError("Node uncertainty standard deviation cannot be negative.")

        # Initialize coordinates with added uncertainty
        self.x: float = float(x) + random.gauss(0, node_uncertainty_SD)
        self.y: float = float(y) + random.gauss(0, node_uncertainty_SD)
        self.z: float = float(z) + random.gauss(0, node_uncertainty_SD)
        self.cell_belongings: List["Cell"] = cell_belongings  # List of cells containing the point
        self.connected_beams: set = set()  # List of beams connected to the point.

        # Indexation and tagging
        self.index: Optional[int] = None  # Global index of the point
        self.tag: Optional[int] = None  # Global boundary tag
        self.cell_local_tag: Optional[dict] = {} # Dictionary of local tags for each cell containing the point
        self.index_boundary: Optional[int] = None  # Global index for boundary cell

        # Simulation attributes
        self.displacement_vector: List[float] = [0.0] * 6  # Displacement vector of 6 DOF (Degrees of Freedom).
        self.reaction_force_vector: List[float] = [0.0] * 6  # Reaction force vector of 6 DOF.
        self.applied_force: List[float] = [0.0] * 6  # Applied force vector of 6 DOF.
        self.fixed_DOF: List[bool] = [False] * 6  # Fixed DOF vector (False: free, True: fixed).
        self.global_free_DOF_index: List[Optional[float]] = [None] * 6  # Global free DOF index.
        self.node_mod: bool = False

        # Visualization attributes
        self.magnification_factor: float = 5.0  # Magnification factor for visualization.

    def destroy(self) -> None:
        """
        Remove all references to this point from connected beams and cells.
        """
        self.connected_beams.clear()

        for cell in list(getattr(self, "cell_belongings", [])):
            _discard(getattr(cell, "points_cell", None), self)
        self.cell_belongings.clear()

        # Clear other attributes
        self.index = None
        self.index_boundary = None
        self.tag = None
        self.cell_local_tag = {}
        self.global_free_DOF_index = [None] * 6


    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __sub__(self, other: 'Point') -> List[float]:
        return [self.x - other.x, self.y - other.y, self.z - other.z]

    def __repr__(self) -> str:
        return f"point({self.x}, {self.y}, {self.z}, Index:{self.index})"

    @property
    def coordinates(self) -> Tuple[float, float, float]:
        """
        Retrieve the current position of the point.

        Returns:
        -----------
            Tuple[float, float, float]: (x, y, z) coordinates of the point.
        """
        return self.x, self.y, self.z

    @property
    def data(self) -> List[float]:
        """
        Retrieve point data for exporting.

        Returns:
        -----------
            List[float]: [index, x, y, z] of the point.
        """
        return [self.index, self.x, self.y, self.z]

    @property
    def deformed_coordinates(self) -> Tuple[float, float, float]:
        """
        Retrieve the deformed position of the point.

        Returns:
        -----------
            Tuple[float, float, float]: (x, y, z) coordinates including displacements.
        """
        return (self.x + self.displacement_vector[0] * self.magnification_factor,
                self.y + self.displacement_vector[1] * self.magnification_factor,
                self.z + self.displacement_vector[2] * self.magnification_factor)

# =============================================================================
# SECTION: Design Methods
# =============================================================================


    @timing.category("design")
    @timing.timeit
    def move_to(self, xNew: float, yNew: float, zNew: float) -> None:
        """
        Move the point to new coordinates.

        Parameters
        ----------
        xNew : float
            New x-coordinate.
        yNew : float
            New y-coordinate.
        zNew : float
            New z-coordinate.
        """
        self.x, self.y, self.z = xNew, yNew, zNew

    @timing.category("design")
    @timing.timeit
    def tag_point(self, boundary_box_domain: list[float]) -> int:
        """
        Generate standardized tags for the point based on its position.
        Check : https://docs.fenicsproject.org/basix/v0.2.0/index.html for more informations

        Parameters
        ----------
        boundary_box_domain : List[float]
            Boundary box domain containing [x_min, x_max, y_min, y_max, z_min, z_max].

        Returns
        -------
        int
            tag of the point
        """
        if len(boundary_box_domain) != 6:
            raise ValueError("Boundary box domain must contain 6 values.")
        xMin, xMax, yMin, yMax, zMin, zMax = boundary_box_domain

        def is_in(v, min_v, max_v):
            """ Check if a value is strictly between min_v and max_v. """
            return min_v < v < max_v

        face_codes = {(self.x == xMin, is_in(self.y, yMin, yMax), is_in(self.z, zMin, zMax)): 12,
                      (self.x == xMax, is_in(self.y, yMin, yMax), is_in(self.z, zMin, zMax)): 13,
                      (is_in(self.x, xMin, xMax), self.y == yMin, is_in(self.z, zMin, zMax)): 11,
                      (is_in(self.x, xMin, xMax), self.y == yMax, is_in(self.z, zMin, zMax)): 14,
                      (is_in(self.x, xMin, xMax), is_in(self.y, yMin, yMax), self.z == zMin): 10,
                      (is_in(self.x, xMin, xMax), is_in(self.y, yMin, yMax), self.z == zMax): 15}

        edge_codes = {
            (self.x == xMin, self.y == yMin, is_in(self.z, zMin, zMax)): 102,
            (is_in(self.x, xMin, xMax), self.y == yMin, self.z == zMin): 100,
            (self.x == xMax, self.y == yMin, is_in(self.z, zMin, zMax)): 104,
            (is_in(self.x, xMin, xMax), self.y == yMin, self.z == zMax): 108,
            (self.x == xMin, is_in(self.y, yMin, yMax), self.z == zMin): 101,
            (self.x == xMax, is_in(self.y, yMin, yMax), self.z == zMin): 103,
            (self.x == xMin, self.y == yMax, is_in(self.z, zMin, zMax)): 106,
            (is_in(self.x, xMin, xMax), self.y == yMax, self.z == zMin): 105,
            (self.x == xMax, self.y == yMax, is_in(self.z, zMin, zMax)): 107,
            (is_in(self.x, xMin, xMax), self.y == yMax, self.z == zMax): 111,
            (self.x == xMin, is_in(self.y, yMin, yMax), self.z == zMax): 109,
            (self.x == xMax, is_in(self.y, yMin, yMax), self.z == zMax): 110,
        }

        corner_codes = {
            (self.x == xMin, self.y == yMin, self.z == zMin): 1000,
            (self.x == xMax, self.y == yMin, self.z == zMin): 1001,
            (self.x == xMin, self.y == yMax, self.z == zMin): 1002,
            (self.x == xMax, self.y == yMax, self.z == zMin): 1003,
            (self.x == xMin, self.y == yMin, self.z == zMax): 1004,
            (self.x == xMax, self.y == yMin, self.z == zMax): 1005,
            (self.x == xMin, self.y == yMax, self.z == zMax): 1006,
            (self.x == xMax, self.y == yMax, self.z == zMax): 1007,
        }

        for condition, code in face_codes.items():
            if all(condition):
                return code

        for condition, code in edge_codes.items():
            if all(condition):
                return code

        for condition, code in corner_codes.items():
            if all(condition):
                return code

    @timing.category("Design")
    @timing.timeit
    def is_identical_to(self, other: 'Point', cell_size: list[float]) -> bool:
        """
        Check if this point is identical to another point, modulo the cell size (periodicity).

        Parameters
        ----------
        other : Point
            The other point to compare with.
        cell_size : list[float]
            Size of the unit cell in x, y, z directions.

        Returns
        -------
        bool
            True if the points are identical modulo the cell size, False otherwise.
        """
        return all(
            min(abs(getattr(self, coord)) - abs(getattr(other, coord)),
                size - abs(getattr(self, coord)) - abs(getattr(other, coord))) < 1e-6
            for coord, size in zip(['x', 'y', 'z'], cell_size)
        )

    @timing.category("Design")
    @timing.timeit
    def is_on_boundary(self, boundary_box_lattice) -> bool:
        """
        Get boolean that give information of boundary node

        Parameters:
        -----------
        boundary_box_lattice: list[float]
            Boundary box of the lattice containing [x_min, x_max, y_min, y_max, z_min, z_max].

        Returns:
        ----------
        boolean: (True if node on boundary)
        """
        return (self.x == boundary_box_lattice[0] or
                self.x == boundary_box_lattice[1] or
                self.y == boundary_box_lattice[2] or
                self.y == boundary_box_lattice[3] or
                self.z == boundary_box_lattice[4] or
                self.z == boundary_box_lattice[5])

    @timing.category("Design")
    @timing.timeit
    def distance_to(self, other: 'Point') -> float:
        """
        Calculate the distance to another point.

        Parameters:
        -----------
        other : Point
            The other point to calculate the distance to.

        Returns:
        -------
        float
            The Euclidean distance between this point and the other point.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    @timing.category("Design")
    @timing.timeit
    def set_local_tag(self, cell_index: int, local_tag: int) -> None:
        """
        Set the local tag for a specific cell containing the point.

        Parameters:
        -----------
        cell_index : int
            Index of the cell.
        local_tag : int
            Local tag to assign to the point for the specified cell.
        """
        self.cell_local_tag[cell_index] = local_tag

    @timing.category("Design")
    @timing.timeit
    def add_cell_belonging(self, cell: "Cell") -> None:
        """
        Add a cell to the list of cells containing this point.

        Parameters:
        -----------
        cell : Cell
            The cell to add.
        """
        if cell not in self.cell_belongings:
            self.cell_belongings.append(cell)

# =============================================================================
# SECTION: Simulation Methods
# =============================================================================

    def initialize_reaction_force(self) -> None:
        """
        Reset the reaction force vector to zero.
        """
        self.reaction_force_vector = [0.0] * 6

    def initialize_displacement(self) -> None:
        """
        Reset displacement values to zero for all DOF.
        """
        self.displacement_vector = [0.0] * 6

    @timing.category("Simulation")
    @timing.timeit
    def set_applied_force(self, appliedForce: List[float], DOF: list[int]) -> None:
        """
        Assign applied force to the point.

        Parameters
        ----------
        appliedForce : List[float]
            Applied force values for each DOF.
        DOF : list[int]
            List of DOF to assign (0: x, 1: y, 2: z, 3: Rx, 4: Ry, 5: Rz).
        """
        if len(DOF) != len(appliedForce):
            raise ValueError("Length of DOF and applied_force must be equal.")
        for i in range(len(DOF)):
            self.applied_force[DOF[i]] = appliedForce[i]

    @timing.category("Simulation")
    @timing.timeit
    def set_reaction_force(self, reactionForce: List[float]) -> None:
        """
        Assign reaction force to the point.

        Parameters
        ----------
        reactionForce : List[float]
            Reaction force values for each DOF.
        """
        if len(reactionForce) != 6:
            raise ValueError("Reaction force must have exactly 6 values.")
        for i in range(len(self.reaction_force_vector)):
            self.reaction_force_vector[i] += reactionForce[i]

    @timing.category("Simulation")
    @timing.timeit
    def fix_DOF(self, DOF: List[int]) -> None:
        """
        Fix specific degrees of freedom for the point.

        Parameters
        ----------
        DOF : List[int]
            List of DOF to fix (0: x, 1: y, 2: z, 3: Rx, 4: Ry, 5: Rz).
        """
        for i in DOF:
            self.fixed_DOF[i] = True

    @timing.category("Simulation")
    @timing.timeit
    def calculate_point_energy(self) -> float:
        """
        Calculate the internal energy of the point.

        Returns:
        -------
        float
            The internal energy of the point.
        """
        rf = self.reaction_force_vector + self.applied_force
        u = self.displacement_vector
        return sum(rf[i] * u[i] for i in range(6))
