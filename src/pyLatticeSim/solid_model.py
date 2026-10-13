# =============================================================================
# CLASS: SolidModel
#
# DESCRIPTION:
#  This class defines a 3D solid model (tetrahedral mesh of the beams volume) for FenicsX analysis based on a
#  lattice (LatticeSim) geometry. It is the volumetric counterpart of BeamModel.
# =============================================================================
from typing import TYPE_CHECKING

import gmsh
import numpy as np

from pyLatticeDesign.materials import MatProperties
from pyLatticeDesign.timing import timing

if TYPE_CHECKING:
    from pyLatticeSim.lattice_sim import LatticeSim


def _import_dolfinx():
    try:
        from dolfinx import fem as _fem, mesh as _mesh  # type: ignore
        from dolfinx.io import gmshio as _gmshio  # type: ignore
        return _fem, _mesh, _gmshio
    except Exception as e:
        class _Missing:
            def __getattr__(self, _name):
                raise RuntimeError(
                    "dolfinx (and petsc4py) is required at runtime. "
                    "For documentation builds this import is mocked. "
                    f"Original import error: {e}"
                )
        return _Missing(), _Missing(), _Missing()


class SolidModel:
    """
    Class to define a 3D solid model for FenicsX analysis from a lattice geometry.
    The beams are meshed as fused cylinders cut at the lattice bounding box, and the faces of the bounding box
    are tagged so that boundary conditions can be applied on them.

    Parameters
    ----------
    COMM : MPI communicator
        MPI communicator for parallel computing

    lattice : LatticeSim
        Lattice object with simulation parameters

    mesh_refinement : float
        Refinement factor for the mesh (higher values lead to finer meshes).

    cell_index : int, optional
        Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.

    min_beam_length : float
        Beams shorter than this length are not meshed.

    verbose : int
        Verbosity level.
    """
    # Tags of the boundary faces of the lattice bounding box
    SURFACE_TAGS = {"Xmin": 1, "Xmax": 2, "Ymin": 3, "Ymax": 4, "Zmin": 5, "Zmax": 6}

    def __init__(self, COMM, lattice: "LatticeSim", mesh_refinement: float = 1.0, cell_index: int = None,
                 min_beam_length: float = 1e-6, verbose: int = 1):
        self.COMM = COMM
        self._verbose = verbose
        self.lattice = None
        self.domain = None
        self.markers = None
        self.facets = None
        self.mesh_refinement = mesh_refinement
        self.min_beam_length = min_beam_length
        self.tolerance = None

        self.E = None
        self.nu = None
        self.mu = None
        self.lmbda = None

        if lattice is not None:
            self.define_model(lattice, cell_index)

    # =============================================================================
    # SECTION: Model Definition Methods
    # =============================================================================

    @timing.category("solid_model_definition")
    @timing.timeit
    def define_model(self, lattice: "LatticeSim", cell_index: int = None):
        """
        Initialize the solid model

        Parameters
        ----------
        lattice : LatticeSim
            The lattice object containing the geometry and the simulation parameters.

        cell_index : int, optional
            Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.
        """
        self.lattice = lattice
        self.generate_mesh(cell_index)
        self.define_boundary_facets()
        self.define_material_model()

    @timing.category("solid_model_definition")
    @timing.timeit
    def generate_mesh(self, cell_index: int = None):
        """
        Generate the tetrahedral mesh of the lattice with GMSH and convert it to a dolfinx mesh.

        Parameters
        ----------
        cell_index : int, optional
            Index of the cell to generate the mesh for. If None, the entire lattice mesh is generated.
        """
        _, _, gmshio = _import_dolfinx()
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.model.add("solid_lattice")
            if self.COMM.rank == 0:
                # Cut at the bounding box so that the loaded faces are planar
                axes = self.lattice.build_lattice_geometry_gmsh(cut_mesh_at_boundary=True, cell_index=cell_index,
                                                                min_beam_length=self.min_beam_length)
                if axes is None:
                    raise RuntimeError("Failed to build the solid geometry of the lattice.")
                volumes = [tag for _, tag in gmsh.model.getEntities(3)]
                gmsh.model.addPhysicalGroup(3, volumes, tag=1)
                self.lattice.set_mesh_size_gmsh(axes, mesh_refinement=self.mesh_refinement)
                gmsh.model.mesh.generate(3)
            self.domain, self.markers, self.facets = gmshio.model_to_mesh(gmsh.model, self.COMM, 0, gdim=3)
        finally:
            gmsh.finalize()

        if self._verbose > 0:
            n_cells = self.domain.topology.index_map(3).size_global
            n_nodes = self.domain.geometry.index_map().size_global
            print(f"Solid mesh: {n_cells} tetrahedra, {n_nodes} nodes")

    @timing.category("solid_model_definition")
    @timing.timeit
    def define_boundary_facets(self):
        """
        Tag the exterior facets lying on the faces of the lattice bounding box with SURFACE_TAGS.
        """
        _, dmesh, _ = _import_dolfinx()
        fdim = self.domain.topology.dim - 1
        bounds = {
            "Xmin": (0, self.lattice.x_min), "Xmax": (0, self.lattice.x_max),
            "Ymin": (1, self.lattice.y_min), "Ymax": (1, self.lattice.y_max),
            "Zmin": (2, self.lattice.z_min), "Zmax": (2, self.lattice.z_max),
        }
        size = max(self.lattice.x_max - self.lattice.x_min, self.lattice.y_max - self.lattice.y_min,
                   self.lattice.z_max - self.lattice.z_min)
        self.tolerance = 1e-6 * size

        facet_indices, facet_values = [], []
        for name, (axis, value) in bounds.items():
            facets = dmesh.locate_entities_boundary(
                self.domain, fdim, lambda x, a=axis, v=value: np.isclose(x[a], v, atol=self.tolerance))
            facet_indices.append(facets)
            facet_values.append(np.full_like(facets, self.SURFACE_TAGS[name]))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_values = np.hstack(facet_values).astype(np.int32)
        order = np.argsort(facet_indices)
        self.facets = dmesh.meshtags(self.domain, fdim, facet_indices[order], facet_values[order])

    @timing.category("solid_model_definition")
    @timing.timeit
    def define_material_model(self):
        """Define the isotropic linear elastic material (Lamé coefficients) from the lattice material."""
        fem, _, _ = _import_dolfinx()
        lattice_material = MatProperties(self.lattice.material_name)
        E = float(lattice_material.young_modulus)
        nu = float(lattice_material.poisson_ratio)
        self.E = fem.Constant(self.domain, E)
        self.nu = fem.Constant(self.domain, nu)
        self.mu = fem.Constant(self.domain, E / (2.0 * (1.0 + nu)))
        self.lmbda = fem.Constant(self.domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    def get_volume(self) -> float:
        """Volume of the meshed solid."""
        import ufl
        fem, _, _ = _import_dolfinx()
        local = fem.assemble_scalar(fem.form(fem.Constant(self.domain, 1.0) * ufl.dx(domain=self.domain)))
        return self.COMM.allreduce(local)
