"""
MeshTrimmer class for handling mesh operations with lattice structures.
"""
from pathlib import Path
from typing import cast, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyLattice.beam import Beam
from pyLattice.point import Point
from pyLattice.cell import Cell
from pyLattice.utils import plot_coordinate_system

class MeshTrimmer:
    """
    A class to handle mesh operations such as loading, saving, scaling.
    Also provides functionality to check if a point is inside the mesh.
    """

    def __init__(self, mesh_name: str = None):
        """
        Initialize a mesh object.
        """
        self.mesh = None
        if mesh_name is not None:
            self.load_mesh(mesh_name)
        self.move_mesh_to_origin()

    def load_mesh(self, mesh_name: str):
        """
        Load a mesh from a file.

        Parameters
        ----------
        mesh_name : str
            Name of the mesh file to load.
            The file should be in STL format and located in the 'mesh_file' directory.
        """
        project_root = Path(__file__).resolve().parents[1]
        mesh_path = Path(mesh_name)
        if mesh_path.suffix != ".stl":
            mesh_path = mesh_path.with_suffix(".stl")
        if not mesh_path.parts[0] == "mesh_file":
            mesh_path = Path("mesh_file") / mesh_path
        mesh_path = project_root / mesh_path

        self.mesh = trimesh.load_mesh(mesh_path)

    def save_mesh(self, output_name: str):
        """
        Save the mesh to a file.

        Parameters
        ----------
        output_name : str
            Name of the output mesh file.
            The file will be saved in STL format in the 'mesh_file' directory.
        """
        project_root = Path(__file__).resolve().parents[2] / "outputs" / "mesh_file"
        output_path = Path(output_name)
        if output_path.suffix != ".stl":
            output_path = output_path.with_suffix(".stl")
        if not output_path.parts[0] == "mesh_file":
            output_path = project_root / output_path

        self.mesh.export(str(output_path))
        print(f"mesh_file saved to {output_path}")

    def scale_mesh(self, scale: float):
        """
        Scale the mesh.

        Parameters
        ----------
        scale : float
            Scale factor.
        """
        self.mesh.apply_scale(scale)
        self.move_mesh_to_origin()

    def move_mesh_to_origin(self):
        """
        Move the mesh to the origin by translating it based on its bounds.
        """
        translation = -self.mesh.bounds[0]
        self.mesh.apply_translation(translation)

    def is_inside_mesh(self, point) -> bool:
        """
        Check if a point is inside the mesh.

        Parameters
        ----------
        point : array-like
            A point in 3D space to check if it is inside the mesh.

        Returns
        -------
        bool
            True if the point is inside the mesh, False otherwise.
        """
        if self.mesh is not None:
            return self.mesh.contains([point])[0]

    def is_cell_in_mesh(self, cell: "Cell") -> bool:
        """
        Check if a cell is inside the mesh.

        Parameters
        ----------
        cell : Cell
            A Cell object whose center point will be checked against the mesh.

        Returns
        -------
        bool
            True if one or more corners of the cell is inside the mesh, False otherwise.
        """
        if self.mesh is not None:
            for point in cell.corner_coordinates:
                if self.is_inside_mesh(point):
                    return True
            return False
        else:
            return True

    def cut_beams_at_mesh_intersection(self, cells: list[Cell]):
        """
        Cut beams at the intersection with the mesh

        Parameters
        ----------
        cells : list[Cell]
            List of Cell objects containing beams to be cut.
        """
        new_beams = []
        beams_to_remove = []

        for cell in cells:
            for beam in cell.beams_cell:
                if not beam.beam_mod:
                    p1_inside = self.is_inside_mesh([beam.point1.x, beam.point1.y, beam.point1.z])
                    p2_inside = self.is_inside_mesh([beam.point2.x, beam.point2.y, beam.point2.z])

                    if not p1_inside and not p2_inside:
                        # The Beam is outside the mesh, remove it
                        beams_to_remove.append(beam)
                    elif not p1_inside or not p2_inside:
                        # The Beam intersects the mesh, cut it
                        intersection_point = self.find_intersection_with_mesh(beam)
                        if intersection_point is not None:
                            new_point = Point(intersection_point[0], intersection_point[1], intersection_point[2], cell_belongings=[])
                            if not p1_inside:
                                new_beam = Beam(new_point, beam.point2, beam.radius, beam.material, beam.type_beam, cell_belongings=[])
                            else:
                                new_beam = Beam(beam.point1, new_point, beam.radius, beam.material, beam.type_beam, cell_belongings=[])

                            new_beams.append(new_beam)
                            beams_to_remove.append(beam)
                else:
                    raise ValueError("Cutting is only available for non modified lattice, because simulation "
                                     "modifications are not planned for trimmed lattice.")
            # Apply changes
            for beam in beams_to_remove:
                cell.remove_beam(beam)
            for beam in new_beams:
                cell.add_beam(beam)

    def find_intersection_with_mesh(self, beam: 'Beam') -> Tuple[float, float, float] | None:
        """
        Find the intersection point of the beam with a mesh.
        Returns the first intersection point if it exists, None otherwise.

        Parameters:
        -----
            mesh_trimmer (mesh_file): The mesh object to check for intersection.
        Returns:
            Tuple[float, float, float] | None: The intersection point (x, y, z) or None if no intersection.
        """
        ray_origin = np.array([beam.point1.x, beam.point1.y, beam.point1.z])
        ray_direction = np.array([beam.point2.x - beam.point1.x,
                                  beam.point2.y - beam.point1.y,
                                  beam.point2.z - beam.point1.z])

        norm = np.linalg.norm(ray_direction)
        if norm == 0:
            return None
        ray_direction /= norm  # Norm

        try:
            from trimesh.ray import ray_pyembree
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        except Exception:
            # Fallback si pyembree n'est pas dispo
            # from trimesh.ray import ray_triangle
            # intersector = ray_triangle.RayMeshIntersector(self.mesh)
            raise ImportError("pyembree is required for ray-mesh intersection. ")

        locations, _, _ = intersector.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )

        if len(locations) > 0:
            return cast(Tuple[float, float, float], tuple(locations[0]))
        return None

    def plot_mesh(self, zoom: float = 0.0, camera_position: Tuple[float, float] = None):
        """
        Visualize a mesh object in 3D.

        Parameters
        ----------
        zoom : float, optional
            Zoom factor for the mesh visualization. Default is 0.0 (no zoom).
        camera_position : Tuple[float, float], optional
            Camera position for the 3D plot, specified as (elevation, azimuth). Default is None (automatic view).
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        faces = self.mesh.vertices[self.mesh.faces]
        mesh_collection = Poly3DCollection(faces, facecolors='cyan', linewidths=0.1, edgecolors='k', alpha=0.7)
        ax.add_collection3d(mesh_collection)

        limit = max(self.mesh.extents) / zoom
        center = self.mesh.centroid

        ax.set_xlim(center[0] - limit, center[0] + limit)
        ax.set_ylim(center[1] - limit, center[1] + limit)
        ax.set_zlim(center[2] - limit, center[2] + limit)

        plot_coordinate_system(ax)

        # Remove background (axes panes and grid)
        ax.set_axis_off()
        ax.grid(False)
        if camera_position is not None:
            ax.view_init(elev=camera_position[0], azim=camera_position[1])
        plt.show()
