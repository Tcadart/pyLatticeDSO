"""
Script to be used in Grasshopper with GHPython component.
Use save_JSON_to_Grasshopper from pyLattice.utils to export lattice from pyLattice to Grasshopper.
See documentation for more information.
"""

__author__ = "tcadart"
__version__ = "2025.01.16"

import json
import csv
import Rhino.Geometry as rg
import ghpythonlib.components as ghcomp
import ghpythonlib.treehelpers as th

# Lenght and center definition
center = rg.Point3d(0.5, 0.5, 0.5)
size_outer = 2
size_inner = 1


# Two cube generation
half_size = size_outer / 2
min_corner = rg.Point3d(center.X - half_size, center.Y - half_size, center.Z - half_size)
max_corner = rg.Point3d(center.X + half_size, center.Y + half_size, center.Z + half_size)
cube_outer = rg.Box(rg.BoundingBox(min_corner, max_corner)).ToBrep()

half_size = size_inner / 2
min_corner = rg.Point3d(center.X - half_size, center.Y - half_size, center.Z - half_size)
max_corner = rg.Point3d(center.X + half_size, center.Y + half_size, center.Z + size_inner)
cube_inner = rg.Box(rg.BoundingBox(min_corner, max_corner)).ToBrep()

min_corner = rg.Point3d(center.X - half_size, center.Y - half_size, center.Z + half_size)
max_corner = rg.Point3d(center.X + half_size, center.Y + half_size, center.Z + size_inner)
cube_surface_up = rg.Box(rg.BoundingBox(min_corner, max_corner)).ToBrep()
# Solid Difference
result = rg.Brep.CreateBooleanDifference([cube_outer], [cube_inner], 0.01)
final_result = rg.Brep.CreateBooleanUnion([result[0], cube_surface_up], 0.01)

volume_data = []
# List of path to look after JSON file
file_path_list = []

if len(file_path_list) == 0:
    raise ValueError("No path given, add path before running")

print(nameLattice)


lattice = None
for base in file_path_list:
    try:
        filePathAll = "{}{}.json".format(base, nameLattice)
        with open(filePathAll, "r") as f:
            lattice = json.load(f)
            break
    except Exception as e:
        print("Impossible to load from {} : {}".format(base, e))
if lattice is None:
    raise IOError("No file found '{}' in given path.".format(nameLattice))


X = lattice["nodesX"]
Y = lattice["nodesY"]
Z = lattice["nodesZ"]
Radius = lattice["radii"]

# Point generation
points = [rg.Point3d(x, y, z) for x, y, z in zip(X, Y, Z)]

# Generation True/False pattern
Pattern = [i % 2 == 0 for i in range(len(points))]

# Separation point list
list_A = [pt for pt, flag in zip(points, Pattern) if flag]
list_B = [pt for pt, flag in zip(points, Pattern) if not flag]

# Line generation between points
lines = [rg.Line(a, b) for a, b in zip(list_A, list_B)]

# Volume parameters for Dendro
min_radius = min(Radius) if Radius else 0.01
max_radius = max(Radius) if Radius else 0.1

voxel_size = max(min_radius / 2.0, 0.002)  # More small for small structures
iso_value = max(min_radius / 3.0, 0.0005)  # Relatif threshold for small radius

volume_settings = ghcomp.DendroGH.CreateSettings(voxel_size, 1, 0, iso_value)


# Generating volume from lines
volume = ghcomp.DendroGH.CurveToVolume(lines, Radius, volume_settings)

# Conversion to mesh
mesh = ghcomp.DendroGH.VolumetoMesh(volume, volume_settings)

if cutCell:
    mesh_cube = rg.Mesh()
    meshes = rg.Mesh.CreateFromBrep(final_result[0], rg.MeshingParameters.Default)
    if meshes:
        for m in meshes:
            mesh_cube.Append(m)

    # Mesh Boolean Difference
    mesh_difference = rg.Mesh.CreateBooleanDifference([mesh], [mesh_cube])
    mesh = mesh_difference[0]

vol = rg.Mesh.Volume(mesh)
