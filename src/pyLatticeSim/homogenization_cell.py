# =============================================================================
# CLASS: HomogenizedCell
#
# DESCRIPTION:
# This class performs lattice homogenization analysis using finite element methods.
# It calculates the homogenized stiffness matrix of a unit cell under various loading conditions.
# =============================================================================

import dolfinx_mpc
import numpy as np
import ufl
from dolfinx import fem, mesh, common
from dolfinx_mpc import MultiPointConstraint
from petsc4py import PETSc
from ufl import as_vector, dot

from .simulation_base import SimulationBase
from pyLattice.timing import timing


class HomogenizedCell(SimulationBase):
    """
    Lattice homogeneization analysis

    Parameter:
    -----------
    BeamModel : BeamModel class object
        An object of class BeamModel with beam properties
    """

    def __init__(self, BeamModel):
        super().__init__(BeamModel)
        self.generalizedStress = None
        self._u_tot = None
        self._k_form_boundary = None
        self._orthotropyError = None
        self._symmetryError = None
        self.orthotropicMatrix = None
        self.homogenizeMatrix = None
        self.saveDataToExport = None
        self._solver = None
        self._mpc = None
        self._SigImposed = None
        self._EpsImposed = None

    # =============================================================================
    # SECTION: Applying boundary conditions and defining forms
    # =============================================================================

    @timing.category("homogenization")
    @timing.timeit
    def calculate_imposed_stress_strain(self, case: int):
        """
        Calculate stress and strain from an imposed study case

        Parameters:
        -----------
        case: int from 1 to 6
            Strain case for homogenization
            1: X direction strain
            2: Y direction strain
            3: Z direction strain
            4: XY direction strain
            5: XZ direction strain
            6: YZ direction strain
        """
        w = self.find_imposed_strain(case)
        self._EpsImposed = self.get_imposed_strains(w)
        self._SigImposed = self.generalized_stress(self._u_)

    @timing.category("homogenization")
    @timing.timeit
    def find_imposed_strain(self, case: int):
        """
        Find imposed strain on domain

        Parameters:
        -----------
        case: int from 1 to 6
            Strain case for homogenization
            1: X direction strain
            2: Y direction strain
            3: Z direction strain
            4: XY direction strain
            5: XZ direction strain
            6: YZ direction strain

        Output:
        --------
        w: ufl.vector on domain
            Strain on domain depending of the strain case
        """
        x = ufl.SpatialCoordinate(self.domain)
        if case == 1:
            w = ufl.as_vector([x[0], 0, 0])
        elif case == 2:
            w = ufl.as_vector([0, x[1], 0])
        elif case == 3:
            w = ufl.as_vector([0, 0, x[2]])
        elif case == 4:
            w = ufl.as_vector([x[1], x[0], 0])
        elif case == 5:
            w = ufl.as_vector([x[2], 0, x[0]])
        elif case == 6:
            w = ufl.as_vector([0, x[2], x[1]])
        else:
            raise ValueError("Invalid case number. Must be between 1 and 6.")
        return w

    @timing.category("homogenization")
    @timing.timeit
    def get_imposed_strains(self, w):
        """
        Calculate strains from imposed displacement domain on dim-6 vectorspace

        Parameters:
        -----------
        w: ufl.as_vector[] dimension (3)
            Imposed strain
        """
        return as_vector([dot(self.tgrad(w), self._t),
                          dot(self.tgrad(w), self._a1),
                          dot(self.tgrad(w), self._a2),
                          0, 0, 0])

    @timing.category("homogenization")
    @timing.timeit
    def locate_Dofs(self, selection_function: callable):
        """
        Locate degrees of freedom from selected logic function
        Modified to be applied only on sub(0) => displacement

        Parameters:
        -----------
        selectionFunction: logicalFunction
            Logical function to locate entities
        """
        nodesLocated = mesh.locate_entities(self.domain, self.domain.topology.dim - 1, selection_function)
        nodesLocatedIndices = np.array(nodesLocated, dtype=np.int32)
        nodesLocatedDofs = fem.locate_dofs_topological(self._V.sub(0), self.domain.topology.dim - 1,
                                                       nodesLocatedIndices)
        return nodesLocatedDofs

    @timing.category("homogenization")
    @timing.timeit
    def define_K_form_boundary(self, markers: int = None):
        """
        Define K_form on element tag by markers

        Parameters:
        ----------
        markers: int
            Markers for boundary conditions, if None then use all boundaries
        """
        self._k_form_boundary = sum([self._Sig[i] * self._Eps[i] * self._dx(markers) for i in [0, 3, 4, 5]]) + (
                self._Sig[1] * self._Eps[1] + self._Sig[2] * self._Eps[2]) * self._dx_shear(markers)

    @timing.category("homogenization")
    @timing.timeit
    def define_L_form(self):
        """
        Define L_form
        """
        self._l_form = -((self._SigImposed[0] * self._EpsImposed[0] * self._dx) +
                         (self._SigImposed[1] * self._EpsImposed[1] * self._dx_shear) +
                         (self._SigImposed[2] * self._EpsImposed[2] * self._dx_shear))

    @timing.category("homogenization")
    @timing.timeit
    def periodic_boundary_condition(self):
        """
        Applying periodic boundary condition on unit cell
        """
        self._mpc = MultiPointConstraint(self._V)

        idx_dict = {}

        tag_categories = {
            "Corner": self.BeamModel.lattice.corner_tags[0],
            "Edge1": self.BeamModel.lattice.edge_tags[0],
            "Edge2": self.BeamModel.lattice.edge_tags[1],
            "Edge3": self.BeamModel.lattice.edge_tags[2],
            "Face1": self.BeamModel.lattice.face_tags[0],
            "Face2": self.BeamModel.lattice.face_tags[1],
            "Face3": self.BeamModel.lattice.face_tags[2],
        }

        for key, tags in tag_categories.items():
            idx_dict[key] = {
                "Master": tags[0],
                "Slave": tags[1:]
            }

        # Add constraints for each category
        for key in idx_dict.keys():
            idx_master = idx_dict[key]["Master"]
            idx_slave = idx_dict[key]["Slave"]
            with common.Timer("Multi constraint"):
                dof_masters = np.tile(fem.locate_dofs_topological(self._V, self.domain.topology.dim - 1,
                                                                  self.BeamModel.facets.find(idx_master)),
                                                                    len(idx_slave))
                dof_slaves = np.block(
                    [fem.locate_dofs_topological(self._V, self.domain.topology.dim - 1, self.BeamModel.facets.find(
                        i)) for i in idx_slave])
                if dof_masters.size > 0:
                    coef = np.ones(dof_slaves.size)
                    owner = np.zeros(dof_slaves.size, dtype=np.int32)
                    offset = np.arange(0, dof_slaves.size + 1)
                    self._mpc.add_constraint(self._V, dof_slaves, dof_masters, coef, owner, offset)
        self._mpc.finalize()

    @timing.category("homogenization")
    @timing.timeit
    def initialize_solver(self):
        """
        Initialize solver for multiple RHS solving
        """
        if self._solver is None:
            with common.Timer("Init Solver"):
                # Assemble the matrix once
                A = dolfinx_mpc.assemble_matrix(fem.form(self._k_form), self._mpc, bcs=self._bcs)
                A.assemble()

                # Create the solver
                self._solver = PETSc.KSP().create(self._COMM)
                self._solver.setOperators(A)
                self._solver.setType(PETSc.KSP.Type.PREONLY)
                self._solver.getPC().setType(PETSc.PC.Type.LU)
                self._solver.setFromOptions()

    @timing.category("homogenization")
    @timing.timeit
    def solve_multiple_linear_problem(self):
        """
        Function to solve multiple linear problem with the same LHS
        """

        # Ensure solver is initialized
        self.initialize_solver()

        b = self.calculate_BTerm()

        # Solve the problem
        self.u = fem.Function(self._V)
        # x_petsc = fem.create_vector(self.u)
        self._solver.solve(b, self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    @timing.category("homogenization")
    @timing.timeit
    def calculate_BTerm(self):
        """
        Calculate the right-hand side vector for the linear problem.
        """
        b = dolfinx_mpc.assemble_vector(fem.form(self._l_form), self._mpc)
        return b

    @timing.category("homogenization")
    @timing.timeit
    def calculate_macro_stress(self, case: int):
        """
        Calculate macro stress on define boundary tags

        Parameter:
        ----------
        case: integer
            Strain case

        Output:
        -------
        MacroStress : matrix[3,3]
            matrix with macroscopic stress

        """
        with common.Timer("Reaction force calculation"):
            self.calculate_generalized_stress(case)

            macroStress = np.zeros((3, 3))
            for tag in self._boundaryTags:
                fi, ri = self.calculate_reaction_force(tag, self._u_tot)
                macroStress += np.kron(np.vstack(fi), ri)
        return macroStress

    @timing.category("homogenization")
    @timing.timeit
    def get_logical_function(self, location: str):
        """
        Return a logical function for selecting parts of the domain boundary

        Parameters:
        -----------
        location: str
            Identifier for the location ("Center" expected here)

        Returns:
        --------
        function: Callable
            Logical function for locating entities
        """
        if location == "Center":
            x_center = np.mean(self.domain.geometry.x, axis=0)

            def center_selector(x):
                tol = 1e-6
                return np.logical_and.reduce([
                    np.isclose(x[0], x_center[0], atol=tol),
                    np.isclose(x[1], x_center[1], atol=tol),
                    np.isclose(x[2], x_center[2], atol=tol)
                ])

            return center_selector

        else:
            raise ValueError(f"Unknown location type: {location}")

    @timing.category("homogenization")
    @timing.timeit
    def apply_dirichlet_for_homogenization(self):
        """
        Apply Dirichlet boundary condition for homogenization
        """
        selectionFunction = self.get_logical_function("Center")
        nodesLocatedDofs = self.locate_Dofs(selectionFunction)
        # Define zero function of dim vector space to apply boundary condition
        u_bc = fem.Function(self._V)
        self._bcs = [fem.dirichletbc(u_bc, nodesLocatedDofs)]

    @timing.category("homogenization")
    @timing.timeit
    def calculate_generalized_stress(self, case: int):
        """
        Calculate generalized stress on a strain case

        Parameters:
        -----------
        case: integer
            Strain case
        """
        M_function = fem.functionspace(self.domain, self._element_mixed)
        Macro = fem.Function(M_function)
        Moment_data = fem.Expression(self.find_imposed_strain(case), M_function.sub(0).element.interpolation_points())
        Macro.sub(0).interpolate(Moment_data)

        self._u_tot = fem.Function(M_function)
        self._u_tot.x.array[:] = self.u.x.array + Macro.x.array
        self.generalizedStress = self.generalized_stress(self._u_tot)

    # =============================================================================
    # SECTION: Solving methods
    # =============================================================================

    @timing.category("homogenization")
    @timing.timeit
    def solve_full_homogenization(self):
        """
        Solve the entire homogenization of the unit cell

        Output:
        -------
        self.homogenizeMatrix: homogenization matrix
        """
        ImposeStrainCase = range(1, 7)  # 6 loading cases
        self.homogenizeMatrix = []
        self.saveDataToExport = []
        self.find_boundary_tags()
        self.define_K_form_boundary(2)
        for loadingCase in ImposeStrainCase:
            with common.Timer("LoadingCase"):
                self.calculate_imposed_stress_strain(loadingCase)
                self.define_L_form()
                self.solve_multiple_linear_problem()

                macroStress = self.calculate_macro_stress(loadingCase)

                self.homogenizeMatrix.append(np.array(
                    [macroStress[0][0], macroStress[1][1], macroStress[2][2], macroStress[1][0], macroStress[2][0],
                     macroStress[2][1]]).T)
                self.saveDataToExport.append(self._u_tot)
        self.homogenizeMatrix = np.column_stack(self.homogenizeMatrix)
        self.convert_to_orthotropic_form()
        self.compute_errors()
        self.homogenizeMatrix = 0.5 * (self.homogenizeMatrix + self.homogenizeMatrix.T)  # Ensure symmetry
        return self.homogenizeMatrix

    # =============================================================================
    # SECTION: Post-processing methods
    # =============================================================================

    @timing.category("homogenization")
    @timing.timeit
    def print_homogenized_matrix(self):
        """
        Print simulation results of 6 loading case
        """
        print("Homogenized matrix: ")
        for row in self.homogenizeMatrix:
            print(" ".join(f"{val:10.3f}" for val in row))

    @timing.category("homogenization")
    @timing.timeit
    def convert_to_orthotropic_form(self):
        """
        Convert homogenize matrix to orthotropic form
        """
        Hinv = np.linalg.inv(self.homogenizeMatrix)
        Ex = 1 / Hinv[0, 0]
        Ey = 1 / Hinv[1, 1]
        Ez = 1 / Hinv[2, 2]
        Gxy = 1 / (2 * Hinv[3, 3])
        Gxz = 1 / (2 * Hinv[4, 4])
        Gyz = 1 / (2 * Hinv[5, 5])
        nuxy = -Hinv[0, 1] * Ey
        nuxz = -Hinv[0, 2] * Ez
        nuyz = -Hinv[1, 2] * Ez

        # Initialize orthotropic matrix
        self.orthotropicMatrix = np.zeros_like(self.homogenizeMatrix)

        # Normal stiffness
        self.orthotropicMatrix[0, 0] = Ex
        self.orthotropicMatrix[1, 1] = Ey
        self.orthotropicMatrix[2, 2] = Ez
        self.orthotropicMatrix[3, 3] = Gxy
        self.orthotropicMatrix[4, 4] = Gxz
        self.orthotropicMatrix[5, 5] = Gyz

        # Poisson effects
        self.orthotropicMatrix[0, 1] = nuxy
        self.orthotropicMatrix[1, 0] = self.orthotropicMatrix[0, 1]
        self.orthotropicMatrix[0, 2] = nuxz
        self.orthotropicMatrix[2, 0] = self.orthotropicMatrix[0, 2]
        self.orthotropicMatrix[1, 2] = nuyz
        self.orthotropicMatrix[2, 1] = self.orthotropicMatrix[1, 2]

    @timing.category("homogenization")
    @timing.timeit
    def get_S_orthotropic(self):
        """
        Return orthotropic matrix
        """
        Hinv = np.linalg.inv(self.homogenizeMatrix)
        Ex = 1 / Hinv[0, 0]
        Ey = 1 / Hinv[1, 1]
        Ez = 1 / Hinv[2, 2]
        Gxy = 1 / (2 * Hinv[3, 3])
        Gxz = 1 / (2 * Hinv[4, 4])
        Gyz = 1 / (2 * Hinv[5, 5])
        nuxy = -Hinv[0, 1] * Ey
        nuxz = -Hinv[0, 2] * Ez
        nuyz = -Hinv[1, 2] * Ez
        matSorthotropic = np.array([
            [1 / Ex, -nuxy / Ex, -nuxz / Ex, 0.0, 0.0, 0.0],
            [-nuxy / Ex, 1 / Ey, -nuyz / Ey, 0.0, 0.0, 0.0],
            [-nuxz / Ex, -nuyz / Ey, 1 / Ez, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1 / Gxy, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1 / Gxz, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1 / Gyz]])
        return matSorthotropic

    def print_orthotropic_form(self):
        """
        Print orthotropic form of homogenization matrix
        """
        print('Ex ', self.orthotropicMatrix[0, 0])
        print('Ey ', self.orthotropicMatrix[1, 1])
        print('Ez ', self.orthotropicMatrix[2, 2])
        print('nuxy ', self.orthotropicMatrix[0, 1])
        print('nuxz ', self.orthotropicMatrix[0, 2])
        print('nuyz ', self.orthotropicMatrix[1, 2])
        print('Gxy ', self.orthotropicMatrix[3, 3])
        print('Gxz ', self.orthotropicMatrix[4, 4])
        print('Gyz ', self.orthotropicMatrix[5, 5])

    @timing.category("homogenization")
    @timing.timeit
    def compute_errors(self):
        """
        Calculate errors
        """
        C = self.homogenizeMatrix
        C_symm = (C + C.T) / 2.0
        self._symmetryError = np.linalg.norm(C_symm - C) / np.linalg.norm(C)

    def print_errors(self):
        """
        Print multiple types of errors in the matrix of simulation results
        """
        print("Symmetry error: ", self._symmetryError)
