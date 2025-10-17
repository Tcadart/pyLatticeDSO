"""
Lattice beam simulation with FenicsX

Author: Thomas Cadart
Date: 2025-02-07
"""
from typing import Tuple
from colorama import Style, Fore

import numpy as np

import ufl
from ufl import (TestFunction, TrialFunction, split, as_vector, dot, grad, diag, Measure,
                 dx, action)
from basix.ufl import element, mixed_element

from dolfinx import fem, common, mesh
from dolfinx.fem.petsc import LinearProblem

class SimulationBase:
    """
    Parent class with all utilities to compute simulation in FenicsX
    """
    def __init__(self, BeamModel, verbose: int = 1):

        self.BeamModel = BeamModel
        self._verbose = verbose
        self._COMM = self.BeamModel.COMM
        self.domain = self.BeamModel.domain
        self._t = self.BeamModel.t
        self._a1 = self.BeamModel.a1
        self._a2 = self.BeamModel.a2

        self._V = None
        self._bcs = []
        self._k_form = None
        self._l_form = None
        self._element_mixed = None
        self._SigGrad = None
        self._dx = None
        self._dx_shear = None
        self._ds = None
        self.moments = None
        self.forces = None
        self._Sig = None
        self._Eps = None
        self._du = None
        self._u_ = None
        self.u = None
        self._k_formGrad = None
        self._boundaryTags = None
        self._K = None
        self._simulation_prepared = False
        self._point_loads = []  # list of (global_dof_index, value) for true nodal point loads




    def prepare_simulation(self):
        """
        Initialize the simulation requisites
        """
        if not self._simulation_prepared:
            self.define_mixed_function_space(('CG', 'CG'))
            self.define_test_trial_function()
            self.calculate_stress_strain()
            self.define_measures()
            self.define_K_form()
            self._simulation_prepared = True

    def define_test_trial_function(self):
        """
        Define Test and Trial Function from function space
        """
        self._u_ = TestFunction(self._V)
        self._du = TrialFunction(self._V)

    def calculate_stress_strain(self):
        """
        Calculate stress and strain
        """
        self._Eps = self.generalized_strains(self._u_)
        self._Sig = self.generalized_stress(self._du)

    def calculate_stress_grad(self):
        """
        Calculate stress gradient
        """
        self._SigGrad = self.generalized_stress_grad(self._du)

    def generalized_stress(self, u):
        """
        Calculate stress from a fem.Function

        Parameters:
        -----------
        u: fem.Function

        """
        return dot(diag(self.BeamModel.material.properties_for_stress), self.generalized_strains(u))

    def generalized_stress_grad(self, u):
        """
        Calculate stress gradient from a fem.Function

        Parameters:
        -----------
        u: fem.Function
        """
        return dot(diag(self.BeamModel.material.properties_for_stress_gradient), self.generalized_strains(u))

    def generalized_strains(self, u):
        """
        Calculate strain from a fem.Function

        Parameters:
        -----------
        u: fem.Function

        """
        (w, theta) = split(u)
        return as_vector([dot(self.tgrad(w), self._t),
                          dot(self.tgrad(w), self._a1) - dot(theta, self._a2),
                          dot(self.tgrad(w), self._a2) + dot(theta, self._a1),
                          dot(self.tgrad(theta), self._t),
                          dot(self.tgrad(theta), self._a1),
                          dot(self.tgrad(theta), self._a2)])


    def calculate_forces(self, u):
        """
        Extract force components
        """
        sig = self.generalized_stress(u)
        self.forces = as_vector([sig[0], sig[1], sig[2]])

    def calculate_moments(self, u):
        """
        Extract moments components
        """
        sig = self.generalized_stress(u)
        self.moments = as_vector([sig[3], sig[4], sig[5]])

    def tgrad(self, u):
        """
        Calculate tangential gradient
        """
        return dot(grad(u), self._t)

    def define_measures(self):
        """
        Define measures with subdomain data
        """
        self._ds = Measure("ds", domain=self.domain, subdomain_data=self.BeamModel.facets)
        self._dx_shear = dx(scheme="default", metadata={"quadrature_scheme": "default", "quadrature_degree": 1},
                            subdomain_data=self.BeamModel.markers)
        self._dx = Measure("dx", domain=self.domain, subdomain_data=self.BeamModel.markers)

    def define_mixed_function_space(self, element_type: Tuple[str, str]):
        """
        Define Mixed function space with ufl library

        Parameters
        ----------
        element_type: Tuple[str, str] => (elementType, elementType)
            Element type for each element of the function space
        """
        element_diplacement = element(element_type[0], self.domain.basix_cell(), 1, shape=(3,))
        element_moment = element(element_type[1], self.domain.basix_cell(), 1, shape=(3,))
        self._element_mixed = mixed_element([element_diplacement, element_moment])
        self._V = fem.functionspace(self.domain, self._element_mixed)


    def define_K_form(self):
        """
        Define K_form
        """
        self._k_form = (sum([self._Sig[i] * self._Eps[i] * self._dx for i in [0, 3, 4, 5]]) +
                        (self._Sig[1] * self._Eps[1] + self._Sig[2] * self._Eps[2]) * self._dx_shear)

    def define_K_form_grad(self):
        """
        Define K_formGrad
        """
        self._k_formGrad = (sum([self._SigGrad[i] * self._Eps[i] * self._dx for i in [0, 3, 4, 5]]) +
                        (self._SigGrad[1] * self._Eps[1] + self._SigGrad[2] * self._Eps[2]) * self._dx_shear)

    def define_L_form_null(self):
        """
        Define L_form as null
        """
        if self._l_form is None:
            self._l_form = self._u_[0] * fem.Constant(self.domain, 0.0) * self._ds

    def apply_force_at_node(self, node_tag: int, forceValue: float, dofIdx: int):
        """
        Apply a concentrated force at a specific node.

        Parameters:
        -----------
        node_tag: int
            Tag identifying the node.
        forceValue: float
            The value of the force.
        dofIdx: int
            The degree of freedom where the force is applied
        """
        self.apply_constraint_at_node(node_tag, forceValue, dofIdx, "Force")


    def apply_displacement_at_node(self, node_tag: int, displacementValue: float, dofIdx: int):
        """
        Applying displacement at a node with a specific tag

        Parameters:
        ------------
        node_tag: int
            Tag identifying the node
        displacementValue: float
            Displacement value to apply at the node
        dofIdx: int
            Between 0 and 5: the degree of freedom where move is apply
        """
        self.apply_constraint_at_node(node_tag, displacementValue, dofIdx, "Displacement")

    # def apply_constraint_at_node(self, node_tag: int, value_constraint: float, dof_idx: int, type_constraint: str):
    #     """
    #     Apply a constraint at a specific node.
    #     """
    #     dictType = {"Displacement", "Force"}
    #     if type_constraint not in dictType:
    #         raise ValueError("Type of constraint must be 'Displacement' or 'Force'")
    #
    #     # Locate the node indices with the given tag
    #     nodeIndices = self.BeamModel.facets.find(node_tag)
    #     if len(nodeIndices) == 0:
    #         raise ValueError("No nodes found with tag " + str(node_tag))
    #
    #     if type_constraint == "Displacement":
    #         # Find the DOFs associated with the node (mixed space)
    #         nodesLocatedDofs = fem.locate_dofs_topological(self._V, self.domain.topology.dim - 1, nodeIndices)
    #         # Keep only the targeted component dof
    #         target_dof = nodesLocatedDofs[dof_idx]
    #         u_bc = fem.Function(self._V)
    #         with u_bc.vector.localForm() as loc:
    #             loc[target_dof] = value_constraint
    #         self._bcs.append(fem.dirichletbc(u_bc, np.array([target_dof], dtype=np.int32)))
    #
    #     elif type_constraint == "Force":
    #         # DOFs at this vertex for the displacement subspace
    #         dofs_node = fem.locate_dofs_topological(self._V.sub(0), self.domain.topology.dim - 1, nodeIndices)
    #
    #         # Indicator on the *mixed* space (so it can be used directly in the UFL form)
    #         indicator = fem.Function(self._V)
    #         with indicator.x.petsc_vec.localForm() as loc:
    #             loc.set(0.0)
    #             for idx in np.atleast_1d(dofs_node):
    #                 loc[idx] = 1.0
    #
    #         (w_, theta_) = ufl.split(self._u_)
    #         (ind_w, ind_theta) = ufl.split(indicator)
    #
    #         # Only the requested component carries the value; others are zero
    #         f_vec = np.zeros(3, dtype=float)
    #         f_vec[dof_idx] = value_constraint
    #
    #         for i in range(3):
    #             if not np.isclose(f_vec[i], 0.0):
    #                 term = f_vec[i] * ind_w[i] * w_[i] * self._dx
    #                 self._l_form = term if self._l_form is None else self._l_form + term
    #
    def apply_constraint_at_node(self, node_tag: int, value_constraint: float, dof_idx: int, type_constraint: str):
        """
        Apply a constraint at a specific node.
        """
        dictType = {"Displacement", "Force"}
        if type_constraint not in dictType:
            raise ValueError("Type of constraint must be 'Displacement' or 'Force'")

        # Locate the node indices with the given tag
        nodeIndices = self.BeamModel.facets.find(node_tag)
        if len(nodeIndices) == 0:
            raise ValueError("No nodes found with tag " + str(node_tag))

        if type_constraint == "Displacement":
            # Map global dof_idx (0..5) to mixed subspace/component
            sub = 0 if dof_idx < 3 else 1
            comp = dof_idx if dof_idx < 3 else dof_idx - 3

            # DOFs for this boundary entity on the chosen subspace/component
            dofs = fem.locate_dofs_topological(self._V.sub(sub).sub(comp),
                                               self.domain.topology.dim - 1, nodeIndices)

            # Dirichlet BC on this scalar component (new dolfinx API)
            C = fem.Constant(self.domain, float(value_constraint))
            self._bcs.append(fem.dirichletbc(C, np.asarray(dofs, dtype=np.int32), self._V.sub(sub).sub(comp)))

        elif type_constraint == "Force":
            # DOFs at this vertex for the displacement subspace
            dofs_node = fem.locate_dofs_topological(self._V.sub(0), self.domain.topology.dim - 1, nodeIndices)

            # Indicator on the *mixed* space (so it can be used directly in the UFL form)
            indicator = fem.Function(self._V)
            with indicator.x.petsc_vec.localForm() as loc:
                loc.set(0.0)
                for idx in np.atleast_1d(dofs_node):
                    loc[idx] = 1.0

            (w_, theta_) = ufl.split(self._u_)
            (ind_w, ind_theta) = ufl.split(indicator)

            f_vec = np.zeros(3, dtype=float)
            f_vec[dof_idx] = value_constraint

            for i in range(3):
                if not np.isclose(f_vec[i], 0.0):
                    term = f_vec[i] * ind_w[i] * w_[i] * self._dx
                    self._l_form = term if self._l_form is None else self._l_form + term


    def apply_all_boundary_condition_on_lattice(self, cell_only=None):
        """
        Apply all boundary conditions on the lattice
        """
        for cell in self.BeamModel.lattice.cells:
            if cell_only is not None and cell_only != cell:
                continue
            for node in cell.points_cell:
                if any(dof == 1 for dof in node.fixed_DOF) or any(force != 0 for force in node.applied_force):
                    for i in range(6):
                        if node.fixed_DOF[i] == 1:
                            self.apply_displacement_at_node(node.tag, node.displacement_vector[i], i)
                        if node.applied_force[i] != 0:
                            self.apply_force_at_node(node.tag, node.applied_force[i], i)

    def apply_all_boundary_condition_on_cell_without_distinction(self, cellToApply):
        """
        Apply all boundary conditions on a specific cell without distinction between fixed DOF and applied force.
        Here we *always* impose the current node.displacement_vector on each boundary DOF of the target cell.
        Uses the **cell-local tag** to avoid None tags and wrong entity mapping.
        """
        for cell in self.BeamModel.lattice.cells:
            if cell is not cellToApply:
                continue

            # Ensure order exists (matches usage elsewhere in your code)
            if getattr(cell, "node_in_order_simulation", None) is None:
                cell.define_node_order_to_simulate()

            for node in cell.node_in_order_simulation:
                # Only boundary nodes are represented in the FenicsX mesh tags
                if node.index_boundary is None:
                    continue

                # Get the FEniCS tag local to this cell
                tag = (node.cell_local_tag.get(cell.index)
                       if isinstance(node.cell_local_tag, dict)
                       else node.cell_local_tag[cell.index])

                if tag is None:
                    continue  # no mapped entity -> skip safely

                # Skip if this entity isn't present in the mesh (robust to trimming/collisions)
                ent = self.BeamModel.facets.find(int(tag))
                if ent.size == 0:
                    continue

                # Impose the current displacement vector component-wise
                for i in range(6):
                    self.apply_displacement_at_node(int(tag), float(node.displacement_vector[i]), i)



    def apply_displacement_at_node_all_DOF(self, node_tag: int, displacementVector: list):
        """
        Applying displacement at a node with a specific tag

        Parameters:
        ------------
        node_tag: int
            Tag identifying the node
        displacementVector: list of float
            Displacement value to apply at the node
        """
        # Locate nodes with the given tag
        nodeIndices = self.BeamModel.facets.find(node_tag)

        if len(nodeIndices) != 0:
            # Find the degrees of freedom associated with these nodes
            nodesLocatedDofs = fem.locate_dofs_topological(self._V, self.domain.topology.dim - 1, nodeIndices)
            # Define the displacement function and set the values
            u_bc = fem.Function(self._V)
            with u_bc.vector.localForm() as loc:
                loc[nodesLocatedDofs] = displacementVector
            # print(u_bc.x.array)
            # Apply the boundary condition
            self._bcs.append(fem.dirichletbc(u_bc, nodesLocatedDofs))



    def find_boundary_tags(self):
        """
        Find all tag with a node on the boundary of the unit cell
        """
        tags = [
            tag for group in (
                    self.BeamModel.lattice.corner_tags +
                    self.BeamModel.lattice.edge_tags +
                    self.BeamModel.lattice.face_tags
            ) for tag in group
        ]

        self._boundaryTags = []
        for tag in tags:
            if len(self.BeamModel.facets.find(tag)) != 0:
                self._boundaryTags.append(tag)
        return self._boundaryTags

    def solve_problem(self):
        """
        Solve the problem with a linear solver

        Uses PETSc KSP solver with LU preconditioner.
        This implementation allows to add true nodal point loads collected in self._point_loads before calling this
        method.
        """
        from dolfinx.fem import petsc as fem_petsc
        from petsc4py import PETSc
        self.define_L_form_null()
        a_form = fem.form(self._k_form)
        l_form = fem.form(self._l_form)

        # Assemble matrix
        A = fem_petsc.assemble_matrix(a_form, bcs=self._bcs)
        A.assemble()

        # Assemble RHS
        b = fem_petsc.create_vector(l_form)
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        fem_petsc.assemble_vector(b, l_form)

        # Apply lifting and BCs to RHS
        fem_petsc.apply_lifting(b, [a_form], bcs=[self._bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, self._bcs)

        # Inject TRUE nodal point loads collected earlier
        if self._point_loads:
            idx = np.array([i for (i, _) in self._point_loads], dtype=np.int32)
            val = np.array([v for (_, v) in self._point_loads], dtype=PETSc.ScalarType)
            b.setValues(idx, val, addv=PETSc.InsertMode.ADD_VALUES)
            b.assemble()

        # Solve with LU
        ksp = PETSc.KSP().create(self._COMM)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)

        self.u = fem.Function(self._V)
        if self._verbose > 0:
            print("Solving linear problem...")
        ksp.solve(b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        if self._verbose > 0:
            print(Fore.GREEN + "Problem solved" + Style.RESET_ALL)

    def calculate_reaction_force(self, node_tag: int, solution: fem.Function = None):
        """
        Calculate reaction force on node_tag on macro coordinates basis
        with strategy 2 : https://bleyerj.github.io/comet-fenicsx/tips/computing_reactions/computing_reactions.html

        Parameters:
        -----------
        node_tag : integer
            The node tag where reaction force is calculated
        solution : fem.Function, optional
            The solution function to use for the calculation. If None, uses self.u.

        Returns:
        --------
        f : np.ndarray
            The reaction force vector (3 components).
        r : np.ndarray
            The position vector of the node.
        """
        if solution is None:
            solution = self.u
        residual = action(self._k_form, solution)
        v_reac = fem.Function(self._V)
        virtual_work_form = fem.form(action(residual, v_reac))

        entity = np.array([self.BeamModel.facets.find(node_tag)[0]], dtype=np.int32)


        C = fem.Constant(self.domain,1.0)
        bcDof = fem.locate_dofs_topological(self._V.sub(0).sub(0), self.domain.topology.dim - 1,entity)
        bcx = fem.dirichletbc(C, np.array(bcDof, dtype=np.int32), self._V.sub(0).sub(0))

        v_local = v_reac.x.array
        bcx.set(v_local)
        v_reac.x.array[:] = v_local
        Rx = fem.assemble_scalar(virtual_work_form)

        v_reac.x.array[:] = 0.0
        bcDof = fem.locate_dofs_topological(self._V.sub(0).sub(1), self.domain.topology.dim - 1,entity)
        bcy = fem.dirichletbc(C, np.array(bcDof, dtype=np.int32), self._V.sub(0).sub(1))
        bcy.set(v_reac.x.array)
        v_reac.x.array[:] = v_local
        Ry = fem.assemble_scalar(virtual_work_form)

        v_reac.x.array[:] = 0.0
        bcDof = fem.locate_dofs_topological(self._V.sub(0).sub(2), self.domain.topology.dim - 1,entity)
        bcz = fem.dirichletbc(C, np.array(bcDof, dtype=np.int32), self._V.sub(0).sub(2))
        bcz.set(v_reac.x.array)
        v_reac.x.array[:] = v_local
        Rz = fem.assemble_scalar(virtual_work_form)
        f = np.array([Rx,Ry,Rz])

        node_idx = self.BeamModel.facets.find(node_tag)[0]
        r = self.domain.geometry.x[node_idx] # Position vector

        return f, r

    def calculate_reaction_force_and_moment_at_position(self, position: np.ndarray, solution: "fem.Function" = None,
                                                        tol: float = 1e-8):
        """
        Same as calculate_reaction_force_and_moment but selects the node by spatial position (x,y,z).
        Avoids locate_dofs_geometrical on subspaces by collapsing subspaces to get dof coords, then
        mapping back to the original subspace indices.
        """
        if solution is None:
            solution = self.u

        px, py, pz = map(float, position)

        # Residual and virtual work form
        residual = action(self._k_form, solution)
        v_reac = fem.Function(self._V)
        virtual_work_form = fem.form(action(residual, v_reac))

        C = fem.Constant(self.domain, 1.0)
        reaction_forces = []

        def _component_dofs_from_position(sub: int, comp: int) -> np.ndarray:
            # Collapse subspace to access dof coordinates
            Vc, collapse_map = self._V.sub(sub).sub(comp).collapse()
            # Coordinates of dofs on collapsed space
            dof_coords = Vc.tabulate_dof_coordinates().reshape(-1, self.domain.geometry.dim)
            mask = (
                    np.isclose(dof_coords[:, 0], px, atol=tol)
                    & np.isclose(dof_coords[:, 1], py, atol=tol)
                    & np.isclose(dof_coords[:, 2], pz, atol=tol)
            )
            idx_collapsed = np.flatnonzero(mask)
            if idx_collapsed.size == 0:
                raise RuntimeError(f"No DOF found for sub={sub}, comp={comp} near point {position} with tol={tol}.")
            # Map back to indices in the (non-collapsed) subspace
            dofs_subspace = [collapse_map[idx] for idx in idx_collapsed]
            return np.asarray(dofs_subspace, dtype=np.int32)

        for i in range(6):
            comp = i % 3
            sub = 0 if i < 3 else 1  # 0: force dofs (u), 1: moment dofs (rotation)

            dofs = _component_dofs_from_position(sub, comp)
            bc = fem.dirichletbc(C, dofs, self._V.sub(sub).sub(comp))

            v_local = v_reac.x.array
            bc.set(v_local)
            v_reac.x.array[:] = v_local
            R = fem.assemble_scalar(virtual_work_form)
            reaction_forces.append(R)

            v_reac.x.array[:] = 0.0  # reset for next component

        return reaction_forces

    def calculate_reaction_force_and_moment(self, node_tag: int):
        """
        Calculate reaction force and moment at a node on the macro coordinate basis
        using strategy 2: https://bleyerj.github.io/comet-fenicsx/tips/computing_reactions/computing_reactions.html

        Parameters:
        -----------
        node_tag : int
            The node tag where the reaction force and moment are calculated.

        Returns:
        --------
        arrayOfReaction : list[float]
            The six components of the reaction (3 forces, 3 moments).
        r : np.ndarray
            The position vector of the node.
        """
        residual = action(self._k_form, self.u)
        v_reac = fem.Function(self._V)
        virtual_work_form = fem.form(action(residual, v_reac))

        entity = np.array([self.BeamModel.facets.find(node_tag)[0]], dtype=np.int32)
        C = fem.Constant(self.domain, 1.0)
        arrayOfReaction = []

        for i in range(6):
            comp = i % 3
            sub = 0 if i < 3 else 1  # 0: force, 1: moment

            dofs = fem.locate_dofs_topological(self._V.sub(sub).sub(comp), self.domain.topology.dim - 1, entity)
            bc = fem.dirichletbc(C, dofs, self._V.sub(sub).sub(comp))

            v_local = v_reac.x.array
            bc.set(v_local)
            v_reac.x.array[:] = v_local
            R = fem.assemble_scalar(virtual_work_form)
            arrayOfReaction.append(R)

            v_reac.x.array[:] = 0.0  # reset for next component

        node_idx = self.BeamModel.facets.find(node_tag)[0]
        r = self.domain.geometry.x[node_idx]

        return arrayOfReaction, r

    def calculate_reaction_force_and_moment_all_boundary_nodes(self, cell, full_nodes: bool = False):
        """
        Calculate reaction force on all boundary nodes
        """
        reactionForces = []
        positions = []
        for node in cell.node_in_order_simulation:
            tag = node.cell_local_tag[cell.index]
            if len(self.BeamModel.facets.find(tag)) != 0:
                force, position = self.calculate_reaction_force_and_moment(tag)
                reactionForces.append(force)
                positions.append(position)
            elif full_nodes:
                reactionForces.append(np.zeros(6))
                positions.append(np.zeros(3))
        return reactionForces, positions

    def calculate_energy_cell(self, displacement: list):
        """
        Calculate the energy in the cell based on the displacement vector.
        """
        nodeInOrder = self.BeamModel.lattice.cells[0].getNodeOrderToSimulate()
        RF, nodes = self.calculate_reaction_force_and_moment_all_boundary_nodes(nodeInOrder)
        dataReactionForce = np.array(RF).flatten()
        displacementVector = np.array(displacement).flatten()
        Energy = 0.5 * np.dot(dataReactionForce, displacementVector)
        return Energy

    def calculate_strain_energy(self):
        """
        Compute the strain energy in the lattice cell.

        Returns:
        --------
        W : float
            Total strain energy stored in the cell.
        """
        residual = 0.5 * action(self._k_form, self.u)
        virtual_work_form = fem.form(action(residual, self.u))
        W = fem.assemble_scalar(virtual_work_form)
        return W

    def calculate_strain_energy_grad(self):
        """
        Compute the strain energy gradient in the lattice cell.

        Returns:
        --------
        WGrad : float
            Total strain energy gradient stored in the cell.
        """
        self.calculate_stress_grad()
        self.define_K_form_grad()

        residual = 0.5 * action(self._k_formGrad, self.u)
        virtual_work_form = fem.form(action(residual, self.u))
        WGrad = fem.assemble_scalar(virtual_work_form)
        return WGrad


    def extract_stress_field(self):
        """
        Extract and interpolate the stress field in the structure after solving the problem.

        Returns:
        --------
        stress_function: fem.Function
            The stress field interpolated into the function space for visualization or post-processing.
        """
        # Ensure the solution exists
        if not hasattr(self, 'u') or self.u is None:
            raise RuntimeError("Solution 'self.u' is not defined. Solve the problem first.")

        # Step 1: Calculate stress from the solved displacement and rotation
        stress_expr = self.generalized_stress(self.u)

        # Step 2: Define a function space for stress
        elem = element("DG", self.domain.ufl_cell(), degree=1)  # Stress has 6 components
        M_function = fem.functionspace(self.domain, elem)

        # Step 3: Create a function for stress and interpolate the calculated stress
        stress_function = fem.Function(M_function)
        stress_data = fem.Expression(stress_expr, M_function.element.interpolation_points())
        stress_function.interpolate(stress_data)

        stress_reshape = stress_function.vector.array.reshape((-1, 6))
        print(f"Stress shape: {stress_reshape}")
        max_stress = np.max(np.abs(stress_reshape), axis=0)
        print(f"Max stress: {max_stress}")

        return stress_function

    def print_timers(self):
        """
        Print all accumulated timers with wall-clock time.
        """
        common.list_timings(self._COMM, [common.TimingType.wall])
