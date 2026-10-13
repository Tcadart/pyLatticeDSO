# =============================================================================
# CLASS: SolidLatticeSimulation
#
# DESCRIPTION:
# This class handles 3D linear elastic simulations of lattice structures meshed as solids (SolidModel)
# using FenicsX. Boundary conditions are read from the same input file format as the beam simulation.
# =============================================================================
from typing import TYPE_CHECKING

import numpy as np
import ufl
from mpi4py import MPI
from colorama import Fore, Style

from pyLatticeDesign.timing import timing
from .simulation_base import set_direct_lu_solver

if TYPE_CHECKING:
    from .solid_model import SolidModel


def _import_dolfinx():
    try:
        from dolfinx import fem as _fem, geometry as _geometry  # type: ignore
        return _fem, _geometry
    except Exception as e:
        class _Missing:
            def __getattr__(self, _name):
                raise RuntimeError(
                    "dolfinx (and petsc4py) is required at runtime. "
                    "For documentation builds this import is mocked. "
                    f"Original import error: {e}"
                )
        return _Missing(), _Missing()


fem, geometry = _import_dolfinx()


class SolidLatticeSimulation:
    """
    3D linear elastic simulation of a lattice structure meshed with tetrahedra.

    Boundary conditions use the "boundary_conditions" block of the lattice input file:
    - one surface (e.g. ["Xmin"]) applies the condition on the whole face of the bounding box,
      a "Force" value being the total force spread as a uniform traction over the face;
    - several surfaces (e.g. ["Xmax", "Zmax"]) apply the condition on their intersection (edge or corner),
      a "Force" value being split equally over the nodes found there.
    Rotational DOFs (RX, RY, RZ) have no meaning for a solid and are ignored.

    Parameters:
    -----------
    SolidModel: SolidModel
        Solid model containing the mesh, the boundary tags and the material.

    element_degree: int
        Degree of the Lagrange elements for the displacement (1: linear, 2: quadratic tetrahedra).
        Quadratic elements are much more accurate for bending dominated lattices.

    verbose: int
        Verbosity level.
    """
    DOF_MAP = {"X": 0, "Y": 1, "Z": 2}
    ROTATION_DOFS = {"RX", "RY", "RZ"}

    def __init__(self, SolidModel: "SolidModel", element_degree: int = 2, verbose: int = 1):
        self.SolidModel = SolidModel
        self.domain = SolidModel.domain
        self._COMM = SolidModel.COMM
        self._verbose = verbose
        self.element_degree = element_degree

        self._V = fem.functionspace(self.domain, ("Lagrange", element_degree, (3,)))
        self._u_ = ufl.TestFunction(self._V)
        self._du = ufl.TrialFunction(self._V)
        self._dx = ufl.Measure("dx", domain=self.domain)
        self._ds = ufl.Measure("ds", domain=self.domain, subdomain_data=SolidModel.facets)
        self._k_form = ufl.inner(self.stress(self._du), self.strain(self._u_)) * self._dx
        self._l_form = ufl.dot(fem.Constant(self.domain, np.zeros(3)), self._u_) * self._dx

        self._bcs = []
        self._point_loads = []  # list of (local unrolled dof index, value)
        self.boundary_groups = {}  # name of the boundary condition -> dofs per component
        self.u = None

    # =============================================================================
    # SECTION: Constitutive law
    # =============================================================================
    @staticmethod
    def strain(u):
        """Small strain tensor."""
        return ufl.sym(ufl.grad(u))

    def stress(self, u):
        """Cauchy stress tensor of the isotropic linear elastic material."""
        eps = self.strain(u)
        return 2.0 * self.SolidModel.mu * eps + self.SolidModel.lmbda * ufl.tr(eps) * ufl.Identity(3)

    # =============================================================================
    # SECTION: Boundary Condition Applications
    # =============================================================================
    def _locate_component_dofs(self, surfaces: list[str], component: int) -> np.ndarray:
        """
        Local dofs (indices in V, unrolled) of a displacement component on a face or on the intersection of faces.
        """
        tags = self.SolidModel.SURFACE_TAGS
        for surface in surfaces:
            if surface not in tags:
                raise ValueError(f"Surface '{surface}' is not supported by the solid simulation. "
                                 f"Use one of {list(tags)}.")
        fdim = self.domain.topology.dim - 1
        V_sub = self._V.sub(component)
        if len(surfaces) == 1:
            facets = self.SolidModel.facets.find(tags[surfaces[0]])
            return np.asarray(fem.locate_dofs_topological(V_sub, fdim, facets), dtype=np.int32)

        # Intersection of several faces of the bounding box
        lattice = self.SolidModel.lattice
        planes = {"Xmin": (0, lattice.x_min), "Xmax": (0, lattice.x_max), "Ymin": (1, lattice.y_min),
                  "Ymax": (1, lattice.y_max), "Zmin": (2, lattice.z_min), "Zmax": (2, lattice.z_max)}
        tol = self.SolidModel.tolerance

        def marker(x):
            mask = np.ones(x.shape[1], dtype=bool)
            for surface in surfaces:
                axis, value = planes[surface]
                mask &= np.isclose(x[axis], value, atol=tol)
            return mask

        V_collapsed, _ = V_sub.collapse()
        dofs = fem.locate_dofs_geometrical((V_sub, V_collapsed), marker)
        return np.asarray(dofs[0], dtype=np.int32)

    @timing.category("solid_simulation")
    @timing.timeit
    def apply_displacement(self, surfaces: list[str], value: float, component: int, name: str = None):
        """
        Impose a displacement component on a face or on the intersection of faces.

        Parameters:
        -----------
        surfaces: list[str]
            Faces of the bounding box (e.g. ["Xmin"] or ["Xmax", "Zmax"]).

        value: float
            Imposed displacement.

        component: int
            Displacement component (0: x, 1: y, 2: z).

        name: str, optional
            Name of the boundary group, used to retrieve reaction forces.
        """
        dofs = self._locate_component_dofs(surfaces, component)
        if self._COMM.allreduce(dofs.size) == 0:
            raise ValueError(f"No node found on surfaces {surfaces} to impose a displacement.")
        C = fem.Constant(self.domain, float(value))
        self._bcs.append(fem.dirichletbc(C, dofs, self._V.sub(component)))
        if name is not None:
            self.boundary_groups.setdefault(name, {})[component] = dofs

    @timing.category("solid_simulation")
    @timing.timeit
    def apply_force(self, surfaces: list[str], value: float, component: int):
        """
        Apply a total force component on a face (uniform traction) or on the intersection of faces (equal nodal
        loads).

        Parameters:
        -----------
        surfaces: list[str]
            Faces of the bounding box (e.g. ["Xmax"] or ["Xmax", "Zmin"]).

        value: float
            Total force applied.

        component: int
            Force component (0: x, 1: y, 2: z).
        """
        if len(surfaces) == 1:
            tag = self.SolidModel.SURFACE_TAGS.get(surfaces[0])
            if tag is None:
                raise ValueError(f"Surface '{surfaces[0]}' is not supported by the solid simulation.")
            area_form = fem.form(fem.Constant(self.domain, 1.0) * self._ds(tag))
            area = self._COMM.allreduce(fem.assemble_scalar(area_form))
            if area <= 0.0:
                raise ValueError(f"No solid face found on surface {surfaces[0]} to apply a force.")
            traction = np.zeros(3)
            traction[component] = value / area
            self._l_form += ufl.dot(fem.Constant(self.domain, traction), self._u_) * self._ds(tag)
        else:
            dofs = self._locate_component_dofs(surfaces, component)
            n_dofs = self._COMM.allreduce(dofs.size)
            if n_dofs == 0:
                raise ValueError(f"No node found on surfaces {surfaces} to apply a force.")
            for dof in dofs:
                self._point_loads.append((int(dof), value / n_dofs))

    @timing.category("solid_simulation")
    @timing.timeit
    def apply_boundary_conditions_from_lattice(self):
        """
        Apply the boundary conditions defined in the "boundary_conditions" block of the lattice input file.
        """
        boundary_conditions = self.SolidModel.lattice.boundary_conditions or {}
        for type_constraint, dict_data in boundary_conditions.items():
            if type_constraint not in ["Force", "Displacement"]:
                raise ValueError(f"Invalid boundary condition type: {type_constraint}. "
                                 f"Must be 'Force' or 'Displacement'.")
            for name_condition, data in dict_data.items():
                if data.get("SurfaceCells") is not None and self._verbose > 0:
                    print(Fore.YELLOW + f"'SurfaceCells' of '{name_condition}' is ignored by the solid simulation."
                          + Style.RESET_ALL)
                for value, dof in zip(data["Value"], data["DOF"]):
                    if dof in self.ROTATION_DOFS:
                        if value != 0 and self._verbose > 0:
                            print(Fore.YELLOW + f"Rotation {dof} of '{name_condition}' is ignored by the solid "
                                                f"simulation." + Style.RESET_ALL)
                        continue
                    if type_constraint == "Displacement":
                        self.apply_displacement(data["Surface"], value, self.DOF_MAP[dof], name=name_condition)
                    else:
                        self.apply_force(data["Surface"], value, self.DOF_MAP[dof])

    # =============================================================================
    # SECTION: Problem Solving
    # =============================================================================
    def _build_near_nullspace(self):
        """Rigid body modes of the displacement space, used by the algebraic multigrid preconditioner."""
        from petsc4py import PETSc
        from dolfinx import la

        index_map = self._V.dofmap.index_map
        bs = self._V.dofmap.index_map_bs
        basis = [la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(6)]
        b = [v.array for v in basis]
        dofs = [self._V.sub(i).dofmap.list.flatten() for i in range(3)]
        for i in range(3):  # translations
            b[i][dofs[i]] = 1.0
        x = self._V.tabulate_dof_coordinates()
        nodes = self._V.dofmap.list.flatten()
        x0, x1, x2 = x[nodes, 0], x[nodes, 1], x[nodes, 2]
        b[3][dofs[0]] = -x1
        b[3][dofs[1]] = x0
        b[4][dofs[0]] = x2
        b[4][dofs[2]] = -x0
        b[5][dofs[2]] = x1
        b[5][dofs[1]] = -x2
        la.orthonormalize(basis)
        return PETSc.NullSpace().create(vectors=[v.petsc_vec for v in basis], comm=self._COMM)

    @timing.category("solid_simulation")
    @timing.timeit
    def solve_problem(self, solver: str = "direct", rtol: float = 1e-8, max_iterations: int = 1000):
        """
        Assemble and solve the linear elastic problem.

        Parameters:
        -----------
        solver: str
            "direct" (LU factorization with MUMPS when available, robust, memory hungry) or "iterative" (conjugate gradient with algebraic
            multigrid, for large meshes).

        rtol: float
            Relative tolerance of the iterative solver.

        max_iterations: int
            Maximum number of iterations of the iterative solver.
        """
        from dolfinx.fem import petsc as fem_petsc
        from petsc4py import PETSc

        a_form = fem.form(self._k_form)
        l_form = fem.form(self._l_form)

        A = fem_petsc.assemble_matrix(a_form, bcs=self._bcs)
        A.assemble()

        b = fem_petsc.create_vector(l_form)
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        fem_petsc.assemble_vector(b, l_form)
        fem_petsc.apply_lifting(b, [a_form], bcs=[self._bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        if self._point_loads:
            with b.localForm() as b_loc:
                for dof, value in self._point_loads:
                    b_loc.array[dof] += value
        fem_petsc.set_bc(b, self._bcs)

        ksp = PETSc.KSP().create(self._COMM)
        ksp.setOperators(A)
        if solver == "direct":
            ksp.setType(PETSc.KSP.Type.PREONLY)
            if PETSc.Sys.hasExternalPackage("mumps"):
                # MUMPS is much faster than SuperLU_DIST on 3D solid meshes
                ksp.getPC().setType(PETSc.PC.Type.LU)
                ksp.getPC().setFactorSolverType("mumps")
            else:
                set_direct_lu_solver(ksp.getPC())
        elif solver == "iterative":
            A.setNearNullSpace(self._build_near_nullspace())
            ksp.setType(PETSc.KSP.Type.CG)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)
            ksp.setTolerances(rtol=rtol, max_it=max_iterations)
        else:
            raise ValueError("solver must be 'direct' or 'iterative'.")

        self.u = fem.Function(self._V, name="Displacement")
        if self._verbose > 0:
            n_dofs = self._V.dofmap.index_map.size_global * self._V.dofmap.index_map_bs
            print(f"Solving linear elastic problem ({n_dofs} DOFs, {solver} solver)...")
        ksp.solve(b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"Solver did not converge (reason {ksp.getConvergedReason()}).")
        if self._verbose > 0:
            print(Fore.GREEN + "Problem solved" + Style.RESET_ALL)
        ksp.destroy()
        A.destroy()
        b.destroy()

    # =============================================================================
    # SECTION: Post-Processing Utilities
    # =============================================================================
    def _check_solved(self):
        if self.u is None:
            raise RuntimeError("The problem must be solved first (solve_problem).")

    @timing.category("solid_simulation")
    @timing.timeit
    def compute_reaction_forces(self) -> dict:
        """
        Reaction forces on every group of imposed displacements, as the sum of the internal force vector K u on the
        constrained dofs.

        Returns:
        --------
        dict
            Name of the boundary condition -> reaction force [Fx, Fy, Fz] (None for components not constrained).
        """
        from dolfinx import la
        self._check_solved()
        residual = fem.assemble_vector(fem.form(ufl.action(self._k_form, self.u)))
        residual.scatter_reverse(la.InsertMode.add)
        n_owned = self._V.dofmap.index_map.size_local * self._V.dofmap.index_map_bs
        reactions = {}
        for name, dofs_per_component in self.boundary_groups.items():
            force = [None, None, None]
            for component, dofs in dofs_per_component.items():
                owned = dofs[dofs < n_owned]
                force[component] = self._COMM.allreduce(float(np.sum(residual.array[owned])))
            reactions[name] = force
        return reactions

    @timing.category("solid_simulation")
    @timing.timeit
    def compute_strain_energy(self) -> float:
        """Total strain energy 1/2 int sigma:epsilon of the structure."""
        self._check_solved()
        energy = fem.form(0.5 * ufl.inner(self.stress(self.u), self.strain(self.u)) * self._dx)
        return self._COMM.allreduce(fem.assemble_scalar(energy))

    @timing.category("solid_simulation")
    @timing.timeit
    def compute_von_mises_stress(self) -> "fem.Function":
        """
        Von Mises stress, constant per element.

        Returns:
        --------
        fem.Function
            DG0 function of the von Mises stress.
        """
        self._check_solved()
        sigma = self.stress(self.u)
        s = ufl.dev(sigma)
        von_mises = ufl.sqrt(1.5 * ufl.inner(s, s))
        W = fem.functionspace(self.domain, ("DG", 0))
        von_mises_function = fem.Function(W, name="VonMises")
        von_mises_function.interpolate(fem.Expression(von_mises, W.element.interpolation_points()))
        return von_mises_function

    def get_max_displacement(self) -> float:
        """Maximum norm of the displacement."""
        self._check_solved()
        values = self.u.x.array.reshape(-1, 3)
        local_max = float(np.max(np.linalg.norm(values, axis=1))) if values.size else 0.0
        return self._COMM.allreduce(local_max, op=MPI.MAX)

    def evaluate_displacement(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the displacement at given positions.

        Parameters:
        -----------
        points: np.ndarray
            (n, 3) positions.

        Returns:
        --------
        np.ndarray
            (n, 3) displacements, NaN for points outside the local mesh.
        """
        self._check_solved()
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, points)
        colliding = geometry.compute_colliding_cells(self.domain, candidates, points)
        values = np.full((points.shape[0], 3), np.nan)
        for i in range(points.shape[0]):
            cells = colliding.links(i)
            if len(cells) > 0:
                values[i] = self.u.eval(points[i], cells[:1])
        return values

    @timing.category("solid_simulation")
    @timing.timeit
    def set_result_displacement_on_lattice_object(self):
        """
        Assign the displacement of the solid at each lattice node (beam axis) to node.displacement_vector.
        Rotations are left to zero.
        """
        nodes = {id(node): node for cell in self.SolidModel.lattice.cells for node in cell.points_cell}
        nodes = list(nodes.values())
        positions = np.array([[node.x, node.y, node.z] for node in nodes])
        displacements = self.evaluate_displacement(positions)
        n_missing = 0
        for node, disp in zip(nodes, displacements):
            if np.any(np.isnan(disp)):
                n_missing += 1
                continue
            node.displacement_vector[:3] = [float(d) for d in disp]
            node.displacement_vector[3:] = [0.0, 0.0, 0.0]
        if n_missing and self._verbose > 0:
            print(Fore.YELLOW + f"{n_missing} lattice nodes are outside the solid mesh (no displacement assigned)."
                  + Style.RESET_ALL)

    def print_number_DOFs(self):
        """
        Print the total number of degrees of freedom (DOFs) in the simulation.
        """
        num_dofs = self._V.dofmap.index_map.size_global * self._V.dofmap.index_map_bs
        print(f"Total number of DOFs: {num_dofs}")
