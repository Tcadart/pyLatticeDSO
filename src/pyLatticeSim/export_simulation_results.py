# =============================================================================
# CLASS: exportSimulationResults
#
# DESCRIPTION:
#  This class handles the exportation of simulation results to Paraview-compatible
#  formats (VTU/PVD). It supports exporting displacement, rotation, moments,
#  internal forces, local coordinate systems, and homogenization surfaces.
# =============================================================================

import math
import os
from pathlib import Path

import numpy as np
from basix.ufl import element
import gmsh
import ufl

from .utils import clear_directory, directional_modulus

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

class exportSimulationResults:
    """
    Beam data exportation to Paraview

    Parameters:
    ------------
    simulation_model: object
        Your simulation model, exposing .domain, .u, .BeamModel, etc.

    name_export_file: str
        Folder name (will be created under <project_root>/simulation_results/)
        All exported files will be placed inside this directory.
    """

    def __init__(self, simulation_model, name_export_file: str = "simulation_results"):
        io, _ = _import_dolfinx()

        self.simulation_model = simulation_model

        # Project root and export directory
        project_root = Path(__file__).resolve().parents[2] / "data" / "outputs"
        self.out_dir = (project_root / "simulation_results" / name_export_file).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Master PVD (time series). All VTU pieces will sit alongside it.
        self.pvd_path = self.out_dir / f"{name_export_file}.pvd"
        self._vtk = io.VTKFile(self.simulation_model.domain.comm, str(self.pvd_path), "w")

        self.result_to_export = []
        self.M_function = None
        self.element_mesh = {}

    # =============================================================================
    # Core helpers
    # =============================================================================

    def define_function_space(self):
        """Define the function space for forces and moments."""
        _, fem = _import_dolfinx()
        if self.M_function is None:
            element_ = element("DG", self.simulation_model.domain.basix_cell(), 0, shape=(3,))
            self.M_function = fem.functionspace(self.simulation_model.domain, element_)

    # =============================================================================
    # Exports
    # =============================================================================

    def export_displacement_rotation(self, case: int = 0):
        """
        Export displacement and rotation fields from self.simulation_model.u
        """
        disp = self.simulation_model.u.sub(0).collapse()
        disp.name = f"Displacement_{case}" if case != 0 else "Displacement"
        self.result_to_export.append(disp)

        theta = self.simulation_model.u.sub(1).collapse()
        theta.name = f"Rotation_{case}" if case != 0 else "Rotation"
        self.result_to_export.append(theta)

    def export_moments(self, FE_result, case: int = 0):
        """
        Export moments based on FE_result (typically self.simulation_model.u)
        """
        _, fem = _import_dolfinx()
        self.simulation_model.calculate_moments(FE_result)
        self.define_function_space()

        Moment = fem.Function(self.M_function)
        Moment_data = fem.Expression(self.simulation_model.moments,
                                     self.M_function.element.interpolation_points())
        Moment.interpolate(Moment_data)
        Moment.name = f"Moment_{case}" if case != 0 else "Moment"
        self.result_to_export.append(Moment)

    def export_macro_strain(self, case: int):
        """Export macro strain for a given loading case."""
        _, fem = _import_dolfinx()
        self.define_function_space()
        Macro = fem.Function(self.M_function)
        Macro_data = fem.Expression(self.simulation_model.findImposedStrain(case),
                                    self.M_function.element.interpolation_points())
        Macro.interpolate(Macro_data)
        Macro.name = "Macro"
        self.result_to_export.append(Macro)

    def export_local_coordinates_system(self):
        """Export local coordinate axes (a1, a2, t) on the domain."""
        _, fem = _import_dolfinx()
        self.define_function_space()

        a1_ = fem.Function(self.M_function)
        a1__data = fem.Expression(self.simulation_model.BeamModel.a1,
                                  self.M_function.element.interpolation_points())
        a1_.interpolate(a1__data)
        a1_.name = "a1"
        self.result_to_export.append(a1_)

        a2_ = fem.Function(self.M_function)
        a2__data = fem.Expression(self.simulation_model.BeamModel.a2,
                                  self.M_function.element.interpolation_points())
        a2_.interpolate(a2__data)
        a2_.name = "a2"
        self.result_to_export.append(a2_)

        t_ = fem.Function(self.M_function)
        t__data = fem.Expression(self.simulation_model.BeamModel.t,
                                 self.M_function.element.interpolation_points())
        t_.interpolate(t__data)
        t_.name = "t"
        self.result_to_export.append(t_)

    def export_internal_force(self, u, case: int = 0):
        """Export internal forces."""
        _, fem = _import_dolfinx()
        self.simulation_model.calculate_forces(u)
        self.define_function_space()

        Force = fem.Function(self.M_function)
        Force_data = fem.Expression(self.simulation_model.forces,
                                    self.M_function.element.interpolation_points())
        Force.interpolate(Force_data)
        Force.name = f"Forces_{case}" if case != 0 else "Forces"
        self.result_to_export.append(Force)

    # =============================================================================
    # Write and finalize exports
    # =============================================================================

    def export_finalize(self, time: float = 0.0):
        """Write and close (single-shot)."""
        self._vtk.write_function(self.result_to_export, t=time)
        self.close_file()

    def write_function(self, time: float = 0.0):
        """Append current fields at time t to the PVD series."""
        self._vtk.write_function(self.result_to_export, t=time)
        self.result_to_export = []  # clear after writing (optional)

    def close_file(self):
        """Close the PVD writer."""
        print(f"Saving simulation results to: {self.pvd_path}")
        self._vtk.close()

    # =============================================================================
    # Composite exports
    # =============================================================================

    def full_export(self, case: int):
        """
        Export all data for a given case

        Parameters
        ----------
        case: int
            Loading case index
        """
        self.export_displacement_rotation(case=case)
        self.export_moments(self.simulation_model.u, case=case)
        self.export_macro_strain(case)
        self.export_local_coordinates_system()
        self.write_function(time=float(case))

    def export_data_homogenization(self, homogenization_surface: bool = True):
        """
        Export all data from 6 loading cases in homogenization.
        Each item in saveDataToExport is a solution field.

        Parameters
        ----------
        homogenization_surface: bool
            Whether to export the homogenization surface as well.
        """
        for case, simu_result in enumerate(self.simulation_model.saveDataToExport):
            # displacement/rotation from simu_result
            disp = simu_result.sub(0).collapse()
            disp.name = f"Displacement"
            self.result_to_export.append(disp)

            theta = simu_result.sub(1).collapse()
            theta.name = f"Rotation"
            self.result_to_export.append(theta)

            self.export_moments(simu_result, case=case)
            self.export_internal_force(simu_result, case=case)
            self.export_local_coordinates_system()
            self.write_function(time=float(case))

        # Export homogenization surface
        if homogenization_surface:
            mat = self.simulation_model.get_S_orthotropic()
            if mat is not None:
                self.export_homogenization_surface_paraview(np.asarray(mat),
                                                            filename="homogenization_surface",
                                                            n_theta=90, n_phi=180)

        self.close_file()

    def export_homogenization_surface_paraview(self, mat_S_orthotropic: np.ndarray,
                                           filename: str = "homogenization_surface",
                                           n_theta: int = 90, n_phi: int = 180) -> Path:
        """
        Export the 3D homogenization polar surface (directional stiffness) using Gmsh.
        Writes <filename>.msh into self.out_dir.

        Parameters
        ----------
        mat_S_orthotropic: np.ndarray
            6x6 stiffness matrix in Voigt notation (assumed orthotropic)

        filename: str
            Base name for output files (without extension)

        n_theta: int
            Number of angular samples in theta (0..pi)

        n_phi: int
            Number of angular samples in phi (0..2pi)
        """
        # Angular grid
        thetas = np.linspace(0.0, np.pi, n_theta)       # 0..pi
        phis   = np.linspace(0.0, 2.0*np.pi, n_phi)     # 0..2pi

        # Sample directional surface (expects your existing directional_modulus(theta°, phi°))
        pts = []
        for th in thetas:
            th_deg = float(np.degrees(th))
            for ph in phis:
                ph_deg = float(np.degrees(ph))
                vec = directional_modulus(mat_S_orthotropic, th_deg, ph_deg)
                pts.append(vec)
        points = np.asarray(pts, dtype=float)

        # Rescale to fit inside the lattice box
        lat = getattr(self.simulation_model.BeamModel, "lattice", None)
        if lat is not None:
            lx, ly, lz = float(lat.size_x), float(lat.size_y), float(lat.size_z)
            maxabs = np.maximum(np.max(np.abs(points), axis=0), 1e-12)  # avoid divide-by-zero
            scale = np.array([ (lx * 0.5) / maxabs[0],
                               (ly * 0.5) / maxabs[1],
                               (lz * 0.5) / maxabs[2] ])
            points *= scale

        # Connectivity (triangles)
        tris = []
        for i in range(n_theta - 1):
            for j in range(n_phi - 1):
                n0 = i * n_phi + j
                n1 = n0 + 1
                n2 = n0 + n_phi
                n3 = n2 + 1
                tris.append([n0, n1, n3])
                tris.append([n0, n3, n2])
        tris = np.asarray(tris, dtype=np.int64)

        # Gmsh export
        need_finalize = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            need_finalize = True
        try:
            gmsh.model.add("homogenization_surface")

            # Create a discrete surface entity
            surf_tag = gmsh.model.addDiscreteEntity(2)

            # Nodes: gmsh expects 1-based tags and flat coord array
            node_tags = np.arange(1, points.shape[0] + 1, dtype=np.int64)
            coords = points.reshape(-1).tolist()
            gmsh.model.mesh.addNodes(2, surf_tag, node_tags.tolist(), coords)

            elem_type = 2
            elem_tags = np.arange(1, tris.shape[0] + 1, dtype=np.int32)
            nodes_flat = (tris + 1).astype(np.int32).ravel()
            gmsh.model.mesh.addElements(
                2, surf_tag,
                [elem_type],
                [elem_tags],
                [nodes_flat]
            )

            # Write files next to simulation outputs
            msh_path = (self.out_dir / filename).with_suffix(".msh")
            gmsh.write(str(msh_path))
            print(f"Homogenization surface written to {msh_path}")
            return msh_path
        finally:
            gmsh.model.remove()  # clean current model
            if need_finalize:
                gmsh.finalize()

    # =============================================================================
    # 3D Vizualisation export
    # =============================================================================
    # This section is not up to date with the rest of the code and may require adjustments.

    def export_vizualisation_3D(self, save_directory: str | Path, number_point_ext: int = 8, mesh_radius_int: int = 3):
        """
        Export beam data on 3D cylinders. All files are written inside <out_dir>/<save_directory>.

        Parameters
        ----------
        save_directory: str | Path
            Subdirectory inside self.out_dir where to save the VTU files.

        number_point_ext: int
            Number of points along the external circle of the beam cross-section mesh.

        mesh_radius_int: int
            Number of points along the radius of the beam cross-section mesh.
        """
        io, _ = _import_dolfinx()
        def find_vector_director(dofmap, geometry):
            """Find the director vector of the beam element defined by dofmap."""
            return [geometry.x[dofmap[1]][i] - geometry.x[dofmap[0]][i] for i in range(3)]

        save_dir = self.out_dir / Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        clear_directory(str(save_dir))  # if your util expects str

        nameMesh = self.generate_beam_element(number_point_ext, mesh_radius_int)

        radius = self.simulation_model.BeamModel.radius.x.array
        nodePos = self.simulation_model.domain.geometry.x

        for i, dofmap in enumerate(self.simulation_model.domain.geometry.dofmap):
            if radius[i] > 0.0:
                domainElement, markers, facets = io.gmshio.read_from_msh(nameMesh, self.simulation_model.domain.comm)
                self.change_radius(domainElement, radius[i])
                self.resize_element(domainElement, dofmap, nodePos)
                self.rotate_element(domainElement, find_vector_director(dofmap, self.simulation_model.domain.geometry))
                self.move_element(domainElement, nodePos[dofmap[0]])

                # Save VTU per element
                self.save_VTK_data(save_dir / f"beam_{i}.vtu", domainElement)

        # PVD aggregator in the same directory
        self.create_PVD_file(save_dir)

    def save_VTK_data(self, save_path: str | Path, domainElement):
        """
        Save a single beam element domain to VTU file for Paraview visualization.

        Parameters
        ----------
        save_path: str | Path
            Full path where to save the VTU file.
        domainElement: dolfinx.mesh.Mesh
            The mesh of the beam element to save.
        """
        io, fem = _import_dolfinx()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vtk = io.VTKFile(domainElement.comm, str(save_path), "w")
        ve = ufl.VectorElement("DG", domainElement.ufl_cell(), 0, dim=3)
        function = fem.functionspace(domainElement, ve)
        A = fem.Function(function)
        Adata = fem.Constant(domainElement, (0.5, 0.5, 0.5))
        Aexpr = fem.Expression(Adata, function.element.interpolation_points())
        A.interpolate(Aexpr)
        A.name = "A"
        vtk.write_function(A)
        vtk.close()

    def convert_mesh_for_paraview(self, nameMesh: str | Path):
        """
        Convert a given mesh to Paraview VTU format for visualization.

        Parameters
        ----------
        nameMesh: str | Path
            Full path to the input mesh file in MSH format.
        """
        io, fem = _import_dolfinx()
        nameMesh = Path(nameMesh)
        domain, a, b = io.gmshio.read_from_msh(str(nameMesh), self.simulation_model.domain.comm)
        try_dir = self.out_dir / "Result"
        try_dir.mkdir(parents=True, exist_ok=True)

        vtk = io.VTKFile(self.simulation_model.domain.comm, str(try_dir / "Try.vtu"), "w")
        ve = ufl.VectorElement("DG", domain.ufl_cell(), 0, dim=3)
        function = fem.functionspace(domain, ve)
        A = fem.Function(function)
        Adata = fem.Constant(domain, (0.5, 0.5, 0.5))
        Aexpr = fem.Expression(Adata, function.element.interpolation_points())
        A.interpolate(Aexpr)
        A.name = "A"
        vtk.write_function(A)
        vtk.close()

    @staticmethod
    def create_PVD_file(vtk_directory: str | Path, output_filename: str = "#0_AllElements.pvd"):
        """
        Generate a PVD file to open all beam results in one step inside the given directory.

        Parameters
        ----------
        vtk_directory: str | Path
            Directory where VTU files are located.
        output_filename: str
            Name of the output PVD file.
        """
        vtk_directory = Path(vtk_directory)
        vtk_directory.mkdir(parents=True, exist_ok=True)

        if not output_filename.endswith(".pvd"):
            output_filename += ".pvd"

        pvd_path = vtk_directory / output_filename

        vtk_files = sorted([f for f in os.listdir(vtk_directory) if f.endswith(".vtu")])

        pvdContent = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '    <Collection>'
        ]
        for vtkFile in vtk_files:
            pvdContent.append(f'        <DataSet timestep="0" part="0" file="{vtkFile}"/>')
        pvdContent += ['    </Collection>', '</VTKFile>', '']

        pvd_path.write_text("\n".join(pvdContent))
        print("PVD generated:", pvd_path)

    # =============================================================================
    # 3D Geometry helpers
    # =============================================================================

    def change_radius(self, domainElement, radius: float):
        """ Change the radius of the beam element mesh."""
        domainElement.geometry.x[:, 0] *= radius
        domainElement.geometry.x[:, 1] *= radius

    def move_element(self, domainElement, nodePosition):
        """ Move the beam element mesh to the correct position."""
        domainElement.geometry.x[:] += nodePosition

    def resize_element(self, domainElement, dofmap, nodePos):
        """ Resize the beam element mesh to the correct length."""
        elementLenght = np.sqrt(
            (nodePos[dofmap[1]][0] - nodePos[dofmap[0]][0]) ** 2
            + (nodePos[dofmap[1]][1] - nodePos[dofmap[0]][1]) ** 2
            + (nodePos[dofmap[1]][2] - nodePos[dofmap[0]][2]) ** 2
        )
        domainElement.geometry.x[:, 2] *= elementLenght

    def rotate_element(self, newDirection):
        """ Rotate the beam element mesh to the correct direction."""
        currentDirection = np.array([0, 0, 1])
        newDirection = np.array(newDirection)
        newDirection = newDirection / np.linalg.norm(newDirection)

        axis = np.cross(currentDirection, newDirection)
        axis_length = np.linalg.norm(axis)
        if axis_length == 0:
            return
        axis = axis / axis_length
        angle = np.arccos(np.dot(currentDirection, newDirection))

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        I = np.eye(3)
        R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        domainElement.geometry.x[:, :3] = np.dot(domainElement.geometry.x[:, :3], R.T)

    # =============================================================================
    # Mesh generation helpers
    # =============================================================================

    def find_beam_element_mesh_name(self, number_point_ext: int, mesh_radius_int: int):
        """ Find the mesh name for given parameters and check if it exists."""
        meshName = f"BeamElement_{number_point_ext}{mesh_radius_int}.msh"
        script_dir = Path(__file__).resolve().parent
        directory = script_dir / "Mesh" / "Beam_Element"
        directory.mkdir(parents=True, exist_ok=True)
        filePath = directory / meshName
        return str(filePath), filePath.exists()

    def generate_beam_element(self, number_point_ext: int, mesh_radius_int: int):
        """ Generate the beam element mesh using Gmsh if it does not exist."""
        radius = 1

        def modify_number_point_ext_beam(n: int) -> int:
            while n % 4 != 0:
                n += 1
            return n

        nameMesh, meshExist = self.find_beam_element_mesh_name(number_point_ext, mesh_radius_int)
        if not meshExist:
            gmsh.initialize()
            gmsh.model.add("BeamMesh")
            number_point_ext = modify_number_point_ext_beam(number_point_ext)

            squareLength = radius / 3
            point1 = gmsh.model.occ.addPoint(-squareLength, squareLength, 0)
            point2 = gmsh.model.occ.addPoint(squareLength, squareLength, 0)
            point3 = gmsh.model.occ.addPoint(squareLength, -squareLength, 0)
            point4 = gmsh.model.occ.addPoint(-squareLength, -squareLength, 0)

            line1 = gmsh.model.occ.addLine(point1, point2)
            line2 = gmsh.model.occ.addLine(point2, point3)
            line3 = gmsh.model.occ.addLine(point3, point4)
            line4 = gmsh.model.occ.addLine(point4, point1)

            pointCenter = gmsh.model.occ.addPoint(0, 0, 0)
            angleDegrees = math.radians(45)
            point1C = gmsh.model.occ.addPoint(-radius * math.cos(angleDegrees), radius * math.sin(angleDegrees), 0)
            point2C = gmsh.model.occ.addPoint(radius * math.cos(angleDegrees), radius * math.sin(angleDegrees), 0)
            point3C = gmsh.model.occ.addPoint(radius * math.cos(-angleDegrees), radius * math.sin(-angleDegrees), 0)
            point4C = gmsh.model.occ.addPoint(-radius * math.cos(-angleDegrees), radius * math.sin(-angleDegrees), 0)

            arc_circle1 = gmsh.model.occ.addCircleArc(point1C, pointCenter, point2C)
            arc_circle2 = gmsh.model.occ.addCircleArc(point2C, pointCenter, point3C)
            arc_circle3 = gmsh.model.occ.addCircleArc(point3C, pointCenter, point4C)
            arc_circle4 = gmsh.model.occ.addCircleArc(point4C, pointCenter, point1C)

            lineInt1 = gmsh.model.occ.addLine(point1C, point1)
            lineInt2 = gmsh.model.occ.addLine(point2C, point2)
            lineInt3 = gmsh.model.occ.addLine(point3C, point3)
            lineInt4 = gmsh.model.occ.addLine(point4C, point4)

            gmsh.model.occ.synchronize()
            meshCircleExt = int(number_point_ext / 4 + 1)
            gmsh.model.mesh.setTransfiniteCurve(arc_circle1, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(arc_circle2, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(arc_circle3, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(arc_circle4, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(lineInt1, mesh_radius_int)
            gmsh.model.mesh.setTransfiniteCurve(lineInt2, mesh_radius_int)
            gmsh.model.mesh.setTransfiniteCurve(lineInt3, mesh_radius_int)
            gmsh.model.mesh.setTransfiniteCurve(lineInt4, mesh_radius_int)

            gmsh.model.mesh.setTransfiniteCurve(line1, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(line2, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(line3, meshCircleExt)
            gmsh.model.mesh.setTransfiniteCurve(line4, meshCircleExt)

            loop = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])
            plane_surface = gmsh.model.occ.addPlaneSurface([loop])

            loopCircle1 = gmsh.model.occ.addCurveLoop([lineInt1, arc_circle1, lineInt2, line1])
            loopCircle2 = gmsh.model.occ.addCurveLoop([lineInt2, arc_circle2, lineInt3, line2])
            loopCircle3 = gmsh.model.occ.addCurveLoop([lineInt3, arc_circle3, lineInt4, line3])
            loopCircle4 = gmsh.model.occ.addCurveLoop([lineInt4, arc_circle4, lineInt1, line4])
            plane_surface_arc1 = gmsh.model.occ.addPlaneSurface([loopCircle1])
            plane_surface_arc2 = gmsh.model.occ.addPlaneSurface([loopCircle2])
            plane_surface_arc3 = gmsh.model.occ.addPlaneSurface([loopCircle3])
            plane_surface_arc4 = gmsh.model.occ.addPlaneSurface([loopCircle4])

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.setTransfiniteSurface(plane_surface)
            gmsh.model.mesh.setTransfiniteSurface(plane_surface_arc1)
            gmsh.model.mesh.setTransfiniteSurface(plane_surface_arc2)
            gmsh.model.mesh.setTransfiniteSurface(plane_surface_arc3)
            gmsh.model.mesh.setTransfiniteSurface(plane_surface_arc4)
            dim = 2
            gmsh.model.mesh.setRecombine(dim, plane_surface)
            gmsh.model.mesh.setRecombine(dim, plane_surface_arc1)
            gmsh.model.mesh.setRecombine(dim, plane_surface_arc2)
            gmsh.model.mesh.setRecombine(dim, plane_surface_arc3)
            gmsh.model.mesh.setRecombine(dim, plane_surface_arc4)

            extrudeSurface = [(dim, s) for s in [plane_surface, plane_surface_arc1, plane_surface_arc2, plane_surface_arc3, plane_surface_arc4]]
            gmsh.model.occ.extrude(extrudeSurface, 0, 0, 1, recombine=True, numElements=[1])
            gmsh.model.occ.synchronize()

            dim = 3
            volumeTags = [vol[1] for vol in gmsh.model.occ.getEntities(dim)]
            gmsh.model.addPhysicalGroup(dim, volumeTags, 1)
            gmsh.model.setPhysicalName(dim, 1, "VolumeTag")
            gmsh.model.mesh.generate(dim)

            gmsh.write(nameMesh)
            gmsh.finalize()

        return nameMesh

    # =============================================================================
    # Advanced exports
    # =============================================================================

    def export_reaction_force(self, lattice_data):
        """
        Export reaction force as a CG(1) vector field.
        NOTE: This routine assumes exact node coordinate matches after rounding.

        Parameters
        ----------
        lattice_data: object
            The lattice data structure containing cells and beams with node info.
        """
        io, fem = _import_dolfinx()
        alreadyDone = []
        self.define_function_space()

        functionReaction = fem.FunctionSpace(self.simulation_model.domain, ("CG", 1))
        ReactionForce = fem.Function(functionReaction)

        dof_coordinates = functionReaction.tabulate_dof_coordinates()
        nodePosition = np.round(dof_coordinates, 3)
        triplet_tuples = [tuple(row) for row in nodePosition]
        dictNode = {triplet: idx for idx, triplet in enumerate(triplet_tuples)}
        print(dictNode)
        for cell in lattice_data.cells:
            for beam in cell.beams:
                for node in [beam.point1, beam.point2]:
                    if node.indexBoundary is not None and node.indexBoundary not in alreadyDone:
                        nodeIndices = dictNode.get(tuple(np.array([node.x, node.y, node.z])), None)
                        print(nodeIndices)
                        print(ReactionForce.vector[nodeIndices], node.reactionForceValue[:3])
                        ReactionForce.vector[nodeIndices] = node.reactionForceValue[:3]

        ReactionForce.interpolate(ReactionForce)
        ReactionForce.name = "ReactionForce"
        self.result_to_export.append(ReactionForce)