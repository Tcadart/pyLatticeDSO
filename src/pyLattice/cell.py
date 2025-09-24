"""
cell.py
"""
import json
from typing import Iterable, Union, Optional, List
from collections import OrderedDict


import numpy as np
from colorama import Fore, Style

from pyLatticeOpti.surrogate_model_relative_densities import _gp_mean_gradient_rbf_pipeline
from .beam import *
from .point import Point
from .geometries.geometries_utils import *
from .utils import _validate_inputs_cell

class Cell(object):
    """
    Define Cell data for lattice structure
    """

    def __init__(self, pos: list, initial_size: list, coordinate: list, geom_types: list[str],
                 radii: list[float], grad_radius: list or None, grad_dim: list or None, grad_mat: list or None,
                 uncertainty_node: float = 0.0, _verbose: int = 0,
                 beams_already_defined: Optional[dict] = None, nodes_already_defined: Optional[dict] = None) -> None:
        """
        Initialize a Cell with its dimensions and position

        Parameters:
        -----------
        pos: list
            Position of the cell in the lattice
        initial_cell_size: list
            Initial size of the cell
        coordinate: list
            Coordinates of the cell minimum corner in the lattice
        geom_types: list[str]
            Type of lattice geometry
        radii: float
            Base radii of the beam
        grad_radius: list
            Gradient of the radii
        grad_dim: list
            Gradient of the dimensions
        grad_mat: list
            Gradient of the material
        uncertainty_node: float
            Standard deviation for adding uncertainty to node coordinates. Defaults to 0.0.
        _verbose: bool
            If True, prints additional information during initialization. Defaults to False.
        beams_already_defined: set
            Set of beams already defined in the lattice to avoid duplication. Defaults to None.
        nodes_already_defined: set
            Set of nodes already defined in the lattice to avoid duplication. Defaults to None.
        """

        _validate_inputs_cell(pos, initial_size, coordinate, geom_types,
                         radii, grad_radius, grad_dim, grad_mat,
                         uncertainty_node, _verbose)

        self.original_cell_geom = None
        self.original_tags = None
        self.center_point = None
        self._beamMaterial = None
        self.size = None
        self.pos: list[int] = pos
        self.coordinate: list[float] = coordinate
        self.beams_cell: Optional[set] = set()  # Set of beams in the cell
        self.points_cell: Optional[set] = set()  # Set of points in the cell
        self.index: Optional[int] = None
        self.geom_types: list[str] = geom_types
        self.radii: list[float] = radii
        self.coupling_matrix_B: Optional = None  # B matrix (Coupling matrix)
        self.uncertainty_node: float = uncertainty_node
        self.grad_radius: list = grad_radius
        self.grad_mat: list = grad_mat
        self.grad_dim: list = grad_dim
        self._verbose: int = _verbose
        self.neighbour_cells: dict = {}
        self.schur_complement: list[list[float]] or None = None
        self.schur_complement_gradient: list[list[float]] or None = None
        self.node_in_order_simulation: Optional[List[Point]] = None

        self.generate_cell_properties(initial_size, beams_already_defined, nodes_already_defined)
        if self.relative_density > 1 and self._verbose > 0:
            print(Fore.YELLOW + "WARNING: Approximated relative density of the cell is greater than 1. "
                                "Beam radius and cell size is probably not well defined" + Style.RESET_ALL)

    def __del__(self):
        try:
            if hasattr(self, "dispose"):
                self.dispose()
        except Exception:
            pass

    def dispose(self) -> None:
        """
        Dispose of the cell by detaching beams and points, and cleaning up references.
        """
        try:
            beams = list(self.beams_cell or [])
            points = list(self.points_cell or [])

            # Detach the beams
            for b in beams:
                try:
                    if hasattr(b, "cell_belongings") and self in b.cell_belongings:
                        b.cell_belongings.remove(self)
                    # If the beam no longer belongs to any cell, we prudently purge its links
                    if not getattr(b, "cell_belongings", []):
                        if hasattr(b, "delete_beam"):
                            b.delete_beam()
                except Exception:
                    pass

            # Detach the points
            for p in points:
                try:
                    if hasattr(p, "cell_belongings") and self in p.cell_belongings:
                        p.cell_belongings.remove(self)
                    # If the point no longer belongs to any cell, we prudently purge its links
                    if not getattr(p, "cell_belongings", []):
                        if hasattr(p, "connected_beams"):
                            for b in list(getattr(p, "connected_beams")):
                                try:
                                    # Remove the beam from the point's connected beams
                                    if not getattr(b, "cell_belongings", []):
                                        p.connected_beams.remove(b)
                                except Exception:
                                    pass
                except Exception:
                    pass

            # Clear the sets
            try:
                self.beams_cell.clear()
            except Exception:
                pass
            try:
                self.points_cell.clear()
            except Exception:
                pass
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"Cell(Coordinates:{self.coordinate}, Size: {self.size}, Index:{self.index})"

    @property
    def volume(self):
        """ Calculate the volume of the cell."""
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def relative_density(self) -> float:
        """
        Calculate the relative density of the cell based on the volume of beams and the cell volume.
        """
        volumeBeams = 0
        for beam in self.beams_cell:
            if beam.beam_mod is False:
                volumeBeams += beam.volume
        return volumeBeams / self.volume

    @property
    def volume_each_geom(self) -> np.ndarray:
        """
        Get the volume of the cell separated by geometry type_beam.
        """
        volumes = np.zeros(len(self.radii))
        for beam in self.beams_cell:
            if not beam.beam_mod:
                volumeBeam = beam.volume
                volumes[beam.type_beam] += volumeBeam
        return volumes

    @property
    def boundary_box(self) -> list:
        """
        Get the boundary box of the cell

        Returns:
        --------
        list
            List of the boundary box coordinates
        """
        xMin = self.coordinate[0]
        xMax = self.coordinate[0] + self.size[0]
        yMin = self.coordinate[1]
        yMax = self.coordinate[1] + self.size[1]
        zMin = self.coordinate[2]
        zMax = self.coordinate[2] + self.size[2]
        return [xMin, xMax, yMin, yMax, zMin, zMax]

    @property
    def boundary_edges(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """
        Return the 12 edge segments of the cell's axis-aligned bounding box
        as pairs of 3D points ((x,y,z), (x,y,z)).
        """
        x0, x1, y0, y1, z0, z1 = self.boundary_box
        # 8 vertices
        v = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        # 12 edges (by vertex indices)
        idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
            (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        return [(v[i], v[j]) for i, j in idx]

    @property
    def corner_coordinates(self) -> list:
        """
        Get the corner coordinates of the cell.

        Returns:
        --------
        list of tuples
            List of (x, y, z) coordinates of the corner points.
        """
        x0, y0, z0 = self.coordinate
        dx, dy, dz = self.size

        corners = [
            (x0, y0, z0),
            (x0 + dx, y0, z0),
            (x0, y0 + dy, z0),
            (x0 + dx, y0 + dy, z0),
            (x0, y0, z0 + dz),
            (x0 + dx, y0, z0 + dz),
            (x0, y0 + dy, z0 + dz),
            (x0 + dx, y0 + dy, z0 + dz),
        ]

        return corners

    def generate_cell_properties(self, initial_cell_size, beams_already_defined: Optional[set] = None,
                                 nodes_already_defined: Optional[set] = None):
        """
        Generate a cell object with beams and nodes based on the lattice type_beam and radii.

        Parameters:
        -----------
        initialCellSize: list
            Initial size of the cell without modification
        beams_already_defined: set
            Set of beams already defined in the lattice to avoid duplication
        nodes_already_defined: set
            Set of nodes already defined in the lattice to avoid duplication
        """
        idxCell = 0
        for idx, radius in enumerate(self.radii):
            if radius > 0.0:
                if idxCell == 0:
                    self.get_beam_material()
                    beamRadius = self.get_radius(radius)
                    self.get_cell_size(initial_cell_size)
                    self.generate_beams(self.geom_types[idx], beamRadius, idx, beams_already_defined, nodes_already_defined)
                    self.get_cell_center()
                else:
                    hybridRadius = self.get_radius(radius)
                    self.generate_beams(self.geom_types[idx], hybridRadius, idx, beams_already_defined, nodes_already_defined)
                idxCell += 1

    def generate_beams(self, latticeType: str, beamRadius: float, beamType: int = 0,
                       beams_already_defined: Optional[dict] = None, nodes_already_defined: Optional[dict] = None) \
            -> None:
        """
        Generate beams and nodes using a given lattice type_beam and parameters.

        Parameters:
        -----------
        latticeType: str
            Type of lattice structure (e.g., 'BCC', 'Hybrid1', etc.)
        beamRadius: float
            Radius of the beam
        beamType: int
            Type index of the beam
        beams_already_defined: set
            Set of beams already defined in the lattice to avoid duplication
        nodes_already_defined: set
            Set of nodes already defined in the lattice to avoid duplication
        """

        def wkey(x, y, z, ndigits=9):
            """ Create a unique key for a point based on its coordinates rounded to a specified number of digits."""
            return round(x, ndigits), round(y, ndigits), round(z, ndigits)

        frac_cache = {}

        for line in get_beam_structure(latticeType):
            x1f, y1f, z1f, x2f, y2f, z2f = map(float, line)

            x1 = x1f * self.size[0] + self.coordinate[0]
            y1 = y1f * self.size[1] + self.coordinate[1]
            z1 = z1f * self.size[2] + self.coordinate[2]
            x2 = x2f * self.size[0] + self.coordinate[0]
            y2 = y2f * self.size[1] + self.coordinate[1]
            z2 = z2f * self.size[2] + self.coordinate[2]

            k1 = wkey(x1, y1, z1)
            k2 = wkey(x2, y2, z2)

            if nodes_already_defined is not None:
                p1 = nodes_already_defined.get(k1)
                p2 = nodes_already_defined.get(k2)
            else:
                p1 = p2 = None

            if p1 is None:
                p1 = frac_cache.get((x1f, y1f, z1f))
                if p1 is None:
                    p1 = Point(x1, y1, z1, [self], self.uncertainty_node)
                    frac_cache[(x1f, y1f, z1f)] = p1
                else:
                    p1.add_cell_belonging(self)
                if nodes_already_defined is not None:
                    nodes_already_defined[k1] = p1
            else:
                p1.add_cell_belonging(self)

            if p2 is None:
                p2 = frac_cache.get((x2f, y2f, z2f))
                if p2 is None:
                    p2 = Point(x2, y2, z2, [self], self.uncertainty_node)
                    frac_cache[(x2f, y2f, z2f)] = p2
                else:
                    p2.add_cell_belonging(self)
                if nodes_already_defined is not None:
                    nodes_already_defined[k2] = p2
            else:
                p2.add_cell_belonging(self)

            if beams_already_defined is not None:
                bkey = tuple(sorted((k1, k2)))
                beam = beams_already_defined.get(bkey)
            else:
                bkey = None
                beam = None

            if beam is None:
                beam = Beam(p1, p2, beamRadius, self._beamMaterial, beamType, cell_belongings=self)
                if beams_already_defined is not None:
                    beams_already_defined[bkey] = beam
            else: # Already defined beam, just add the cell belonging
                beam.add_cell_belonging(self)

            self.beams_cell.add(beam)
            self.points_cell.add(p1)
            self.points_cell.add(p2)


    def get_beam_material(self) -> None:
        """
        Get the material of the beam based on the gradient and position.

        Parameters:
        -----------
        grad_mat: list
            Gradient of the material

        Returns:
        ---------
        materialType: int
            Material index of the beam
        """
        if self.grad_mat is None:
            self._beamMaterial = 0
        else:
            self._beamMaterial = self.grad_mat[self.pos[2]][self.pos[1]][self.pos[0]]

    def get_radius(self, base_radius: float) -> float:
        """
        Calculate and return the beam radii

        Parameters:
        -----------
        grad_radius: list
            Gradient of the radii
        BaseRadius: float
            Base radii of the beam

        Returns:
        ---------
        actualBeamRadius: float
            Calculated beam radii
        """
        if self.grad_radius is None:
            return base_radius
        else:
            beamRadius = (base_radius * self.grad_radius[self.pos[0]][0] *
                          self.grad_radius[self.pos[1]][1] *
                          self.grad_radius[self.pos[2]][2])
            return beamRadius

    def get_cell_size(self, initial_cell_size: list) -> None:
        """
        Calculate and return the cell size

        Parameters:
        -----------
        initialCellSize: 3-array
            Dimension of the initial cell without modification
        grad_dim:

        Returns:
        ---------
        size : float
            Calculated beam radii
        """
        if self.grad_dim is None:
            self.size = initial_cell_size
        else:
            self.size = [initial_size * self.grad_dim[pos][i] for i, (initial_size, pos) in
                         enumerate(zip(initial_cell_size, self.pos))]

    def get_cell_center(self) -> None:
        """
        Calculate the center point of the cell
        """
        self.center_point = [self.coordinate[i] + self.size[i] / 2 for i in range(3)]

    def remove_beam(self, beam_to_delete: Union["Beam", Iterable["Beam"]]) -> None:
        """
        Removing beam from cell
        """
        if isinstance(beam_to_delete, Beam):
            self.beams_cell.discard(beam_to_delete)
        else:
            for b in beam_to_delete:
                self.beams_cell.discard(b)

    def remove_point(self, point_to_delete: Union["Point", Iterable["Point"]]) -> None:
        """
        Removing point from cell
        """
        if isinstance(point_to_delete, Point):
            self.points_cell.discard(point_to_delete)
        else:
            for p in point_to_delete:
                self.points_cell.discard(p)

    def add_beam(self, beam_to_add: Union["Beam", Iterable["Beam"]]) -> None:
        """
        Adding beam to cell
        """
        if isinstance(beam_to_add, Beam):
            self.beams_cell.add(beam_to_add)
        else:
            self.beams_cell.update(beam_to_add)

    def add_point(self, point_to_add: List["Point"] or "Point") -> None:
        """
        Adding point to cell
        """
        if isinstance(point_to_add, Point):
            self.points_cell.add(point_to_add)
        else:
            self.points_cell.update(point_to_add)

    def get_point_on_surface(self, surfaceName: str) -> list:
        """
        Get the points on the surface specified in the global reference frame.

        Parameters:
        -----------
        surfaceName: str
            Name of the surface. Choose from 'Xmin', 'Xmax', 'Ymin', 'Ymax', 'Zmin', 'Zmax', 'Xmid', 'Ymid', 'Zmid'.
            If 'Xmid', 'Ymid', or 'Zmid' is specified, it returns the points at the bottom of the cell

        Returns:
        --------
        list
           List of points on the specified surface.
        """
        boundaryBox = self.boundary_box
        surface_map = {
            "Xmin": boundaryBox[0],
            "Xmax": boundaryBox[1],
            "Ymin": boundaryBox[2],
            "Ymax": boundaryBox[3],
            "Zmin": boundaryBox[4],
            "Zmax": boundaryBox[5],
            "Xmid": self.coordinate[0],
            "Ymid": self.coordinate[1],
            "Zmid": self.coordinate[2]
        }

        coord_index = {
            "Xmin": "x", "Xmax": "x", "Xmid": "x",
            "Ymin": "y", "Ymax": "y", "Ymid": "y",
            "Zmin": "z", "Zmax": "z", "Zmid": "z"
        }

        if surfaceName not in surface_map:
            raise ValueError(
                f"Surface '{surfaceName}' is not valid. Choose from 'Xmin', 'Xmax', 'Ymin', 'Ymax', 'Zmin', 'Zmax', "
                f"'Xmid', 'Ymid', 'Zmid'.")

        surface_value = surface_map[surfaceName]
        axis = coord_index[surfaceName]

        return list({
            point for beam in self.beams_cell
            for point in [beam.point1, beam.point2]
            if getattr(point, axis) == surface_value
        })

    # REPLACE define_node_order_to_simulate with this version
    def define_node_order_to_simulate(self, face_priority: Optional[List[str]] = None, tol: float = 1e-9) -> None:
        """
        Build a deterministic local ordering of boundary nodes
        Strategy:
          1) Classify each boundary node to one face in a priority order.
          2) Within each face, sort lexicographically by the two in-plane coordinates, then by (x,y,z) for tie-break.
          3) Concatenate faces to get the final order.
        The resulting order is stored as a list of Points in self.node_in_order_simulation.
        """
        if face_priority is None:
            face_priority = ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"]

        x0, x1, y0, y1, z0, z1 = self.boundary_box

        def _faces_of_point(p: Point) -> List[str]:
            faces = []
            if abs(p.x - x0) <= tol: faces.append("Xmin")
            if abs(p.x - x1) <= tol: faces.append("Xmax")
            if abs(p.y - y0) <= tol: faces.append("Ymin")
            if abs(p.y - y1) <= tol: faces.append("Ymax")
            if abs(p.z - z0) <= tol: faces.append("Zmin")
            if abs(p.z - z1) <= tol: faces.append("Zmax")
            return faces

        def _inplane_key(face: str, p: Point):
            # Use the two in-plane coordinates as primary sort keys
            if face in ("Xmin", "Xmax"):
                return p.y, p.z, p.x
            if face in ("Ymin", "Ymax"):
                return p.x, p.z, p.y
            # Z faces
            return p.x, p.y, p.z

        # Unique boundary points by boundary index to avoid duplicates
        bnd_points: dict[int, Point] = {}
        for beam in self.beams_cell:
            for p in (beam.point1, beam.point2):
                if p.index_boundary is not None:
                    bnd_points.setdefault(p.index_boundary, p)

        # Assign each point to a single face using the given priority
        face_buckets: dict[str, List[Point]] = {f: [] for f in face_priority}
        for p in bnd_points.values():
            faces = _faces_of_point(p)
            # pick the first face that appears in face_priority
            chosen = next((f for f in face_priority if f in faces), None)
            if chosen is None:
                # Extremely rare: numerical drift puts a node just off all faces; choose nearest
                dists = {
                    "Xmin": abs(p.x - x0),
                    "Xmax": abs(p.x - x1),
                    "Ymin": abs(p.y - y0),
                    "Ymax": abs(p.y - y1),
                    "Zmin": abs(p.z - z0),
                    "Zmax": abs(p.z - z1),
                }
                chosen = min(dists, key=dists.get)
            face_buckets[chosen].append(p)

        ordered: List[Point] = []
        for face in face_priority:
            pts = face_buckets[face]
            pts.sort(key=lambda q: _inplane_key(face, q))
            ordered.extend(pts)

        self.node_in_order_simulation = ordered

    def set_reaction_force_on_nodes(self, reactionForce: list) -> None:
        """
        Set reaction force on each boundary node in the established local order.
        """
        if not self.node_in_order_simulation:
            raise ValueError("Boundary node order not defined. Call define_node_order_to_simulate() first.")
        if len(reactionForce) < len(self.node_in_order_simulation):
            print(f"Number of reaction force entries: {len(reactionForce)} vs. boundary nodes: {len(self.node_in_order_simulation)}")
            raise ValueError("Not enough reaction force entries for boundary nodes.")

        for idx, node in enumerate(self.node_in_order_simulation):
            node.set_reaction_force(reactionForce[idx])

    def get_displacement_at_nodes(self, nodeList: Union[List["Point"], "OrderedDict[int, Point]"]) -> list:
        """
        Return displacement vectors ordered consistently with the provided local list/dict of Points.
        """
        displacementList = []
        if isinstance(nodeList, list):
            for node in nodeList:
                if node:
                    displacementList.append(node.displacement_vector)
        else:  # OrderedDict-like
            for _, node in nodeList.items():
                if node:
                    displacementList.append(node.displacement_vector)
        return displacementList

    def set_displacement_at_boundary_nodes(self, displacementArray: list) -> None:
        """
        Set displacement at nodes.

        Parameters:
        ------------
        displacementArray: list or array-like
            Flattened array of displacement values.
        """
        if self._verbose > 1:
            print("Non-zero displacements:", np.asarray(displacementArray)[np.asarray(displacementArray) != 0])

        for beam in self.beams_cell:
            for point in [beam.point1, beam.point2]:
                if point.index_boundary is None:
                    continue
                for i in range(6):
                    gi = point.global_free_DOF_index[i]
                    if gi is not None:
                        point.displacement_vector[i] = displacementArray[gi]

    def get_number_boundary_nodes(self) -> int:
        """
        Get the number of unique boundary nodes in the cell.
        """
        return len({p.index_boundary for p in self.points_cell if p.index_boundary is not None})

    def build_coupling_operator(self, nb_free_DOF: int) -> None:
        """
        Build the coupling operator B using the deterministic local boundary-node order.
        """
        if not self.node_in_order_simulation:
            self.define_node_order_to_simulate()

        from scipy.sparse import coo_matrix
        data, row, col = [], [], []

        print(len(self.node_in_order_simulation), self.node_in_order_simulation)
        # Map boundary index
        for local_idx, point in enumerate(self.node_in_order_simulation):
            for i in range(6):
                gi = point.global_free_DOF_index[i]
                if gi is not None:
                    data.append(1)
                    row.append(gi)
                    col.append(local_idx * 6 + i)

        nbBndDOFloc = len(self.node_in_order_simulation) * 6
        self.coupling_matrix_B = coo_matrix((data, (row, col)), shape=(nb_free_DOF, nbBndDOFloc))

    def build_local_preconditioner(self):
        """
        Efficiently compute B * S * B^T but only over the active global rows touched by B.
        """
        from scipy.sparse import coo_matrix, csc_matrix, isspmatrix

        if self.coupling_matrix_B is None:
            raise ValueError("Coupling matrix has not been built yet. Please build it first.")
        if self.coupling_matrix_B.shape[1] != np.shape(self.schur_complement)[0]:
            print("Shape of B matrix", self.coupling_matrix_B.shape)
            print("Shape of Schur matrix", np.shape(self.schur_complement))
            raise ValueError("Incompatible dimensions between the coupling matrix and the Schur matrix.")

        # Ensure sparse Schur complement to avoid dense products
        S = self.schur_complement
        if not isspmatrix(S):
            S = csc_matrix(S)
        else:
            S = S.tocsc()

        B_csr = self.coupling_matrix_B.tocsr()
        active_rows = np.unique(B_csr.nonzero()[0])
        n_global = B_csr.shape[0]

        if active_rows.size == 0:
            return coo_matrix((n_global, n_global))

        subB = B_csr[active_rows, :]  # (r x nb)
        tmp = S @ subB.transpose()  # (nb x r)
        local_block = (subB @ tmp).tocoo()  # (r x r)

        # Map back to absolute global indices without creating a huge intermediate
        row_idx = active_rows[local_block.row]
        col_idx = active_rows[local_block.col]

        return coo_matrix((local_block.data, (row_idx, col_idx)), shape=(n_global, n_global))

    def get_internal_energy(self) -> float:
        """
        Get the internal energy of the cell
        """
        min_energy = -1e-12
        internalEnergy = 0
        for point in self.points_cell:
            if point.index_boundary is not None:
                pointEnergy = point.calculate_point_energy()
                if pointEnergy < min_energy:
                    print("Negative energy = ", pointEnergy)
                    raise ValueError("Negative energy detected at point with index " + str(point.index_boundary))
                internalEnergy += pointEnergy
        return internalEnergy

    def get_displacement_data(self) -> list:
        """
        Build and return displacement data on cell for dataset generation
        """
        allBoundaryDisplacementData = []
        for beam in self.beams_cell:
            for point in [beam.point1, beam.point2]:
                if point.index_boundary is not None:
                    allBoundaryDisplacementData.append(point.displacement_vector)
        return allBoundaryDisplacementData

    def change_beam_radius(self, new_radius: list) -> None:
        """
        ATTENTION: BEAM MOD IS NOT WORKING
        Change beam radii in the cell

        Parameters:
        -----------
        newRadius: list
            beam radii wanted to assign
        hybridData: list
            Hybrid data type_beam
        """
        if self._verbose > 1:
            print(Fore.RED + "WARNING: Beam modification is not implemented yet. " + Style.RESET_ALL)
        assert len(new_radius) == len(self.radii), ("Length of new radii vector and already cell radii vector needs "
                                                    "to be equal ")
        beamRadius = []
        for rad in new_radius:
            beamRadius.append(self.get_radius(rad))

        for beam in self.beams_cell:
            beam.change_beam_radius(beamRadius[beam.type_beam])

        self.radii = new_radius


    def get_relative_density_kriging(self, kriging_model) -> float:
        """
        Get the relative density of the cell using kriging model

        Parameters:
        -----------
        krigingModel: Kriging
            Kriging model to use for prediction
        """
        relative_density = kriging_model.predict(np.array([self.radii]))[0]
        return relative_density

    def get_relative_density_gradient(self, relativeDensityPolyDeriv) -> float:
        """
        Get the gradient of the relative density

        Parameters:
        -----------
        relative_density_poly_deriv: list
            List of polynomial derivative functions

        Returns:
        --------
        deriv: float
            Derivative of the relative density
        """
        deriv = 0
        for idx, polyDeriv in enumerate(relativeDensityPolyDeriv):
            deriv += polyDeriv(self.radii[idx])
        return deriv

    def get_relative_density_gradient_kriging(self, model, geometries_types) -> np.ndarray:
        """
        Finite difference gradient of the relative density (predictive mean) w.r.t. the radii using the trained
        Pipeline(StandardScaler -> GaussianProcessRegressor) with Constant*RBF kernel.
        Returns an array of size len(geometries_types) where entries for geometries not present in the cell are 0.
        """
        epsilon = 1e-3
        grad = np.zeros(len(self.radii))

        for idx, rad in enumerate(self.radii):
            if self.geom_types[idx] not in geometries_types:
                print("geometry types in cell", self.geom_types, " Not in the trained kriging model", geometries_types)
                raise ValueError("Incompatible geometry types between the cell and the kriging model.")
            perturbed_radii = np.zeros(len(geometries_types))
            radii = np.zeros(len(geometries_types))
            geom_index = geometries_types.index(self.geom_types[idx])
            perturbed_radii[geom_index] = rad + epsilon
            radii[geom_index] = rad
            grad[idx] = (model.predict([perturbed_radii]) - model.predict([radii])) / epsilon
        return grad

    def get_relative_density_gradient_kriging_exact(self, model, geometries_types) -> np.ndarray:
        """
        Exact gradient of the relative density (predictive mean) w.r.t. the radii using the trained
        Pipeline(StandardScaler -> GaussianProcessRegressor) with Constant*RBF kernel.

        Returns an array of size len(geometries_types) where entries for geometries not present in the cell are 0.

        Parameters:
        -----------
        model: Pipeline
            Trained kriging model
        geometries_types: list
            List of geometry types in the trained kriging model
        """
        # Build full input vector (one radius per geometry in 'geometries_types')
        x = np.zeros(len(geometries_types), dtype=float)
        for idx, rad in enumerate(self.radii):
            g = self.geom_types[idx]
            if g not in geometries_types:
                print("geometry types in cell", self.geom_types, " not in the trained kriging model", geometries_types)
                raise ValueError("Incompatible geometry types between the cell and the kriging model.")
            x[geometries_types.index(g)] = float(rad)

        # Exact gradient in original feature space
        grad_full = _gp_mean_gradient_rbf_pipeline(model, x)  # shape (len(geometries_types),)

        # Reorder to match the cell's local parameter order (same as self.radii / self.geom_types)
        grad_local = np.zeros(len(self.radii), dtype=float)
        for idx, g in enumerate(self.geom_types):
            grad_local[idx] = grad_full[geometries_types.index(g)]

        return grad_local

    def get_number_nodes_at_boundary(self):
        """
        Get the number of nodes at the boundary

        Returns:
        --------
        int
            Number of nodes at the boundary
        """
        counterNodes = 0
        nodeAlreadyCounted = []
        for beam in self.beams_cell:
            for point in [beam.point1, beam.point2]:
                if point.index_boundary is not None and point.index_boundary not in nodeAlreadyCounted:
                    counterNodes += 1
                    nodeAlreadyCounted.append(point.index_boundary)
        return counterNodes

    def get_RGBcolor_depending_of_radius(self):
        """
        Get the RGB color of the cell depending on the radii.
        """
        return tuple(r / 0.1 for r in self.radii)

    def add_cell_neighbour(self, direction: str, sign: str, neighbour_cell: "Cell") -> None:
        """
        Add a neighbour cell in a structured dict format.

        Parameters
        ----------
        direction : str
            One of "x", "y", "z"
        sign : str
            Either "positif" or "negatif"
        neighbour_cell : Cell
            Neighbour cell to add
        """
        if direction not in self.neighbour_cells:
            self.neighbour_cells[direction] = {}

        if sign not in self.neighbour_cells[direction]:
            self.neighbour_cells[direction][sign] = neighbour_cell

    def get_all_cell_neighbours(self) -> list["Cell"]:
        """
        Get all neighbour cells in a flat list.

        Returns
        -------
        list of Cell
            List of all neighbour cells
        """
        neighbours = []
        for direction in self.neighbour_cells:
            for sign in self.neighbour_cells[direction]:
                neighbours.append(self.neighbour_cells[direction][sign])
        return neighbours

    def print_data(self):
        """
        Print the data of the cell for debugging purposes.
        """
        print("Cell position: ", self.pos)
        print("Cell coordinates: ", self.coordinate)
        print("Cell size: ", self.size)
        print("Lattice type_beam: ", self.geom_types)
        print("Beam radii: ", self.radii)
        print("Beam material: ", self._beamMaterial)
        print("beams in cell: ", self.beams_cell)
        print("Cell center point: ", self.center_point)
        print("Cell index: ", self.index)
        print("Beam material: ", self._beamMaterial)
        print("Coupling matrix: ", self.coupling_matrix_B)
        print("Number of beams: ", len(self.beams_cell))
        print("Volume of the cell: ", self.volume)
        print("Relative density: ", self.relative_density)
        print("Number of nodes at boundary: ", self.get_number_nodes_at_boundary())

    def get_translation_rigid_body(self):
        """
        Get the translation of the rigid body
        """
        translation = np.zeros(3)
        for beam in self.beams_cell:
            for point in [beam.point1, beam.point2]:
                if point.index_boundary is not None:
                    translation += point.displacement_vector[:3]
        return translation / self.get_number_nodes_at_boundary()

    def get_rotation_rigid_body(self):
        """
        Get the rotation matrix of the rigid body using SVD.
        """
        all_points = self.points_cell
        initial_positions = np.array([point.coordinates for point in all_points])  # P_i
        final_positions = np.array([point.deformed_coordinates for point in all_points])  # P_i'

        # Soustraction du centre de gravité
        center_initial = np.mean(initial_positions, axis=0)
        center_final = np.mean(final_positions, axis=0)
        P = initial_positions - center_initial
        P_prime = final_positions - center_final

        # Matrice de covariance
        H = P.T @ P_prime

        # Décomposition SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Correction si nécessaire (assurer que R est une rotation propre)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return R

    def refresh_from_global(self, lattice: "Lattice") -> None:
        """
        Rebuild self.beams_cell and self.points_cell from the current lattice state.
        Keeps only beams that still declare this cell as a belonging, and derives points from those beams.
        """
        live_beams = {
            b for b in lattice.beams
            if getattr(b, "cell_belongings", None) and self in b.cell_belongings
        }
        self.beams_cell = live_beams
        self.points_cell = {p for b in live_beams for p in (b.point1, b.point2)}

