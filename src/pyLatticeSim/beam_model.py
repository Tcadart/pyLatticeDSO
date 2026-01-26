# =============================================================================
# CLASS: BeamModel
#
# DESCRIPTION:
#  This class defines a beam model for FenicsX analysis based on a lattice (LatticeSim) geometry.
# =============================================================================
from typing import TYPE_CHECKING
from pathlib import Path
import sys
import ufl
from ufl import as_vector, sqrt, dot, cross
import numpy as np

from .lattice_generation import *
from .material_definition import Material

try:
    from pyLatticeDesign.materials import MatProperties
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pyLatticeDesign.materials import MatProperties

if TYPE_CHECKING:
    from src.pyLatticeSim.lattice_sim import LatticeSim

from pyLatticeDesign.timing import timing

def _import_dolfinx():
    try:
        from dolfinx import io as _io, fem as _fem  # type: ignore
        return _io, _fem
    except Exception as e:
        class _Missing:
            def __getattr__(self, _name):
                raise RuntimeError(
                    "dolfinx (and petsc4py) is required at runtime. "
                    "For documentation builds this import is mocked. "
                    f"Original import error: {e}"
                )
        return _Missing(), _Missing()

class BeamModel:
    """
    Class to define a beam model for FenicsX analysis from a lattice geometry.

    Parameters
    ----------
    COMM : MPI communicator
        MPI communicator for parallel computing

    lattice : LatticeObject
        Lattice object from pyLatticeDesign

    cell_index : int, optional
        Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.
    """
    def __init__(self, COMM, lattice: "LatticeSim", cell_index=None):
        self.COMM = COMM
        self.facets = None
        self.markers = None
        self.rad_tag_beam = None
        self.tag_beam = None
        self.name_lattice_geometry = None
        self.lattice = None

        self.radius = None
        self.domain = None
        self._material = None
        self.t = None
        self.a1 = None
        self.a2 = None

        if lattice is not None:
            self.define_model(lattice, cell_index)

    @property
    def material(self):
        """Returns material properties"""
        if self._material is None:
            raise AttributeError("Material properties have not been initialized.")
        return self._material

    # =============================================================================
    # SECTION: Model Definition Methods
    # =============================================================================

    @timing.category("beam_model_definition")
    @timing.timeit
    def define_model(self, lattice, cellIndex: int = None):
        """
        Initialize the beam model

        Parameters
        ----------
        lattice : Lattice
            The lattice object containing the mesh generation parameters.

        cellIndex : int, optional
            Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.
        """
        self.lattice = lattice
        self.generate_mesh(self.lattice, cellIndex)
        self.define_material_model(self.lattice)
        self.define_radius()
        self.define_mechanical_properties()
        # self.define_beam_geometry_data_circular_gradient()

    @timing.category("beam_model_definition")
    @timing.timeit
    def open_lattice_geometry(self, name_lattice_geometry: str):
        """
        Open geometry from GMSH mesh file.

        Parameters
        ----------
        name_lattice_geometry : str
            Name of the GMSH mesh file (should end with .msh).
        """
        self.name_lattice_geometry = name_lattice_geometry
        io, _ = _import_dolfinx()
        if not self.name_lattice_geometry.endswith(".msh"):
            raise ValueError(f"The file '{self.name_lattice_geometry}' should have a .msh extension (GMSH)")
        self.domain, self.markers, self.facets = io.gmshio.read_from_msh(self.name_lattice_geometry, self.COMM)

    @timing.category("beam_model_definition")
    @timing.timeit
    def generate_mesh(self, lattice, cell_index=None):
        """
        Generate the mesh of the lattice or a specific cell

        Parameters
        ----------
        lattice : LatticeObject
            The lattice object containing the mesh generation parameters.

        cell_index : int, optional
            Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.
        """
        latticeMesh = latticeGeneration(lattice, self.COMM)
        self.domain, self.markers, self.facets, self.rad_tag_beam, self.tag_beam = (
            latticeMesh.mesh_lattice_cells(cell_index, save_mesh=False))

    @timing.category("beam_model_definition")
    @timing.timeit
    def define_radius(self, radius_dict: dict = None):
        """
        Define radius values from a dictionary

        Parameters
        ----------
        radius_dict : dict, optional
            Dictionary containing radius values keyed by beam tags.
        """
        _, fem = _import_dolfinx()
        if self.domain is None:
            raise RuntimeError("Mesh must be initialized before defining radius.")

        V0 = fem.functionspace(self.domain, ("DG", 0))
        self.radius = fem.Function(V0)

        if radius_dict:
            self.rad_tag_beam = radius_dict

        for rad, tag in self.rad_tag_beam.items():
            elem = self.markers.find(tag)
            self.radius.x.array[elem] = np.full_like(elem, rad, dtype=np.float64)

    @timing.category("beam_model_definition")
    @timing.timeit
    def define_mechanical_properties(self):
        """Compute and set mechanical properties for the beam"""
        if self.radius is None:
            raise RuntimeError("Radius must be defined before computing beam properties.")

        self.material.compute_mechanical_properties(self.radius)
        self.calculate_local_coordinate_system()

    @timing.category("beam_model_definition")
    @timing.timeit
    def define_beam_geometry_data_circular_gradient(self):
        """Define beam geometry data for circular gradient"""
        if self.radius is None:
            raise RuntimeError("Radius must be defined before computing beam properties.")

        self.material.compute_gradient(self.radius, self.tag_beam, self.lattice)

    @timing.category("beam_model_definition")
    @timing.timeit
    def define_material_model(self, lattice):
        """Define all material properties with preset data"""
        self._material = Material(self.domain)
        lattice_material = MatProperties(lattice.material_name)
        self._material.set_material(lattice_material)

    @timing.category("beam_model_definition")
    @timing.timeit
    def calculate_local_coordinate_system(self):
        """Calculate local coordinate system on each beam of the mesh"""
        def tangent(mesh):
            """
            Calculate the tangent vector at each point in the mesh.
            """
            t = ufl.Jacobian(mesh)
            return as_vector([t[0, 0], t[1, 0], t[2, 0]]) / sqrt(ufl.inner(t, t))

        self.t = tangent(self.domain)
        ex = as_vector([1, 0, 0])
        ey = as_vector([0, 1, 0])
        ez = as_vector([0, 0, 1])
        e1 = ufl.conditional(ufl.lt(ufl.sign(self.t[1]) * self.t[1], ufl.sign(self.t[0]) * self.t[0]), ey, ex)
        te1 = dot(self.t, e1)
        e2 = ufl.conditional(ufl.lt(ufl.sign(self.t[2]) * self.t[2], ufl.sign(te1) * te1), ez, e1)
        self.a1 = cross(self.t, e2)
        self.a1 /= sqrt(dot(self.a1, self.a1))
        self.a2 = cross(self.t, self.a1)
        self.a2 /= sqrt(dot(self.a2, self.a2))

