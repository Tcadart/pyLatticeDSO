import gmsh
import numpy as np
from dolfinx.io.gmshio import model_to_mesh

from .utils_timing import *
timingLG = Timing()

class latticeGeneration:
    """
    Meshing of lattice structures with GMSH from class Lattice

    Parameter:
    ------------
    Lattice: Lattice class object
        Object that contains all information of lattice structures
    """

    def __init__(self, lattice, COMM):
        self._mesh_size = []
        self._tag_point_index = {}
        self._tag_beam_index = {}
        self.geom = None
        self.point = {}
        self.beams = []
        self.gdim = None

        self.lattice = lattice
        self.COMM = COMM

    @timingLG.timeit
    def find_mesh_size(self, mesh_element_lenght: float = 0.05):
        """
        Determine mesh size
        """
        self._mesh_size = mesh_element_lenght * self.lattice.cell_size_x

    @timingLG.timeit
    def mesh_lattice_cells(self, cell_index, mesh_element_lenght:float = 0.05, save_mesh:bool = True):
        """
        Meshing lattice structures with node tags and beam tags

        Parameters:
        -----------

        """
        # Find mesh size
        self.find_mesh_size(mesh_element_lenght)

        self.gdim = 1  # beam geometric dimension
        modelRank = 0

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        if self.COMM.rank == modelRank:
            self.geom = gmsh.model.geo
            self.generate_nodes(cell_index)
            radtagBeam, tagBeam = self.generate_beams(cell_index)
            self.geom.synchronize()
            self.generate_beams_tags()
            self.generate_points_tags()
            # self._print_mesh_tags()
            gmsh.model.mesh.generate(self.gdim)
            if save_mesh:
                gmsh.write("Mesh/" + self.lattice.getName() + ".msh")
            domain, cells, facets = model_to_mesh(model=gmsh.model, comm=self.COMM, rank=modelRank, gdim=3)
            gmsh.finalize()
            return domain, cells, facets, radtagBeam, tagBeam


    @timingLG.timeit
    def generate_nodes(self, cell_index = None):
        """
        Generate nodes in the structures with tags associated

        Parameters:
        -----------
        nodes: list of node object from lattice class
        """
        node_already_added = set()
        for cell in self.lattice.cells:
            if cell_index is None or cell.index == cell_index:
                for node in cell.points_cell:
                    if node.index not in node_already_added:
                        node_already_added.add(node.index)
                        point_id = self.geom.addPoint(node.x, node.y, node.z, meshSize=self._mesh_size)
                        self.point[node.index] = point_id
                        if cell_index is not None:
                            if cell.index not in node.cell_local_tag:
                                tagAdd = None
                            else:
                                tagAdd = node.cell_local_tag[cell.index]
                        else:
                            tagAdd = node.tag
                        if tagAdd is not None:
                            if tagAdd not in self._tag_point_index:
                                self._tag_point_index[tagAdd] = []
                            self._tag_point_index[tagAdd].append(point_id)



    @timingLG.timeit
    def generate_beams(self, cell_index = None):
        """
        Generate beams in the structures with tags associated

        Parameters:
        -----------
        beams: list of beams from lattice class

        Return:
        --------
        tag_beam: dictionary of tags associated with radius
        """
        beam_already_added = set()
        tagBeam = {1:set(), 0:set()}
        radBeam = {}
        idxRadBeam = 0
        for cell in self.lattice.cells:
            if cell_index is None or cell.index == cell_index:
                for beam in cell.beams_cell:
                    if beam.radius > 0:
                        if beam not in beam_already_added:
                            beam_already_added.add(beam)
                            beam_id = self.geom.addLine(self.point[beam.point1.index], self.point[beam.point2.index])
                            self.beams.append(beam_id)
                            if beam.beam_mod:
                                tagBeam[1].add(beam.radius)
                            else:
                                tagBeam[0].add(beam.radius)
                            if beam.radius not in radBeam:
                                radBeam[beam.radius] = idxRadBeam
                                idxRadBeam += 1
                            tag = radBeam[beam.radius]

                            if tag not in self._tag_beam_index:
                                self._tag_beam_index[tag] = []
                            self._tag_beam_index[tag].append(beam_id)
        return radBeam, tagBeam

    @timingLG.timeit
    def generate_beams_tags(self):
        """
        Generate tags for beam elements
        """
        for tag, beamIds in self._tag_beam_index.items():
            if beamIds:
                gmsh.model.addPhysicalGroup(self.gdim, beamIds, tag)
                gmsh.model.setPhysicalName(self.gdim, tag, "tag" + str(tag))

    @timingLG.timeit
    def generate_points_tags(self):
        """
        Generate points tags with normalized procedure
        """
        for tag, pointIds in self._tag_point_index.items():
            if pointIds:
                gmsh.model.addPhysicalGroup(self.gdim - 1, pointIds, tag)
                gmsh.model.setPhysicalName(self.gdim - 1, tag, "tag" + str(tag))

    def _print_mesh_tags(self):
        """
        Debug function to print mesh tags
        Print the physical groups of the mesh, including points and beams.
        """
        for tag in gmsh.model.getPhysicalGroups():
            dim, tag = tag
            if dim == 0:
                ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                print(f"Physical Group {tag} (Points): {ents}")
            if dim == 1:
                tagName = ["Center beam", "Modified beam", "Boundary beam"]
                ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                if tag <= 2:
                    print(f"Physical Group {tag} ({tagName[tag]}): {ents}")
                else:
                    print(f"Physical Group {tag}: {ents}")

    @timingLG.timeit
    def mesh_beam(self, L: float, meshSize:float):
        """
        Mesh a beam of length L
        """
        meshComm = self.COMM
        modelRank = 0
        gmsh.initialize()
        if meshComm.rank == modelRank:
            self.geom = gmsh.model.geo
            P0 = self.geom.addPoint(0, 0, 0, meshSize=meshSize)
            P1 = self.geom.addPoint(L, 0, 0, meshSize=meshSize)
            L0 = self.geom.addLine(P0, P1)
            gmsh.model.addPhysicalGroup(1, [L0])
            self.geom.synchronize()

            gmsh.model.mesh.generate(dim=1)
            gmsh.write("Mesh/Beam.msh")
            gmsh.finalize()