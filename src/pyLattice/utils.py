"""
Utility functions for plotting and saving lattice structures in 3D, validating inputs, and saving data in JSON
format for Grasshopper compatibility.
"""
import json
import math
import os
import pickle
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.colors as mcolors

if TYPE_CHECKING:
    from pyLattice.lattice import Lattice


def open_lattice_parameters(file_name: str):
    """
    Open a JSON file containing lattice parameters.

    Parameters:
    -----------
    file_name: str
        Name of the JSON file containing lattice parameters.
    """
    project_root = Path(__file__).resolve().parents[2]
    json_path = project_root / "data" / "inputs" / "preset_lattice" / file_name
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix('.json')

    try:
        with open(json_path, 'r') as file:
            lattice_parameters = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_path} does not exist.")
    return lattice_parameters

def _validate_inputs_lattice(cell_size_x, cell_size_y, cell_size_z,
                     num_cells_x, num_cells_y, num_cells_z,
                     geom_types, radii, grad_radius_property, grad_dim_property, grad_mat_property,
                     uncertainty_node, eraser_blocks):
    # Check cell sizes
    assert isinstance(cell_size_x, (int, float)) and cell_size_x > 0, "cell_size_x must be a positive number"
    assert isinstance(cell_size_y, (int, float)) and cell_size_y > 0, "cell_size_y must be a positive number"
    assert isinstance(cell_size_z, (int, float)) and cell_size_z > 0, "cell_size_z must be a positive number"

    # Check number of cells
    assert isinstance(num_cells_x, int) and num_cells_x > 0, "num_cells_x must be a positive integer"
    assert isinstance(num_cells_y, int) and num_cells_y > 0, "num_cells_y must be a positive integer"
    assert isinstance(num_cells_z, int) and num_cells_z > 0, "num_cells_z must be a positive integer"

    # Check lattice type_beam
    assert isinstance(geom_types, list), "Lattice_Type must be a list"
    assert all(isinstance(lt, str) for lt in geom_types), "All elements of Lattice_Type must be strings"

    # Check radii
    assert isinstance(radii, list), "radii must be a list"
    assert all(isinstance(r, float) for r in radii), "All radii values must be floats"
    assert len(radii) == len(geom_types), "The number of radii must be equal to the number of lattice types"

    # Check gradient properties
    if grad_radius_property is not None:
        assert isinstance(grad_radius_property, list), "gradRadiusProperty must be a list"
        assert len(grad_radius_property) == 3, "gradRadiusProperty must be a list of 3 elements"
    if grad_dim_property is not None:
        assert isinstance(grad_dim_property, list), "gradDimProperty must be a list"
        assert len(grad_dim_property) == 3, "gradDimProperty must be a list of 3 elements"
    if grad_mat_property is not None:
        assert len(grad_mat_property) == 2, "gradMatProperty must be a list of 2 elements"
        assert isinstance(grad_mat_property[0], int), "gradMatProperty[0] must be an integer"
        assert isinstance(grad_mat_property[1], int), "gradMatProperty[1] must be an integer"

    # Check optional parameters
    assert isinstance(uncertainty_node, float), "uncertainty_node must be a float"

    if eraser_blocks is not None:
        for erasedPart in eraser_blocks:
            assert len(erasedPart) == 6 and all(
                isinstance(x, float) for x in erasedPart), "eraser_blocks must be a list of 6 floats"

def _validate_inputs_cell(
        pos: list,
        initial_size: list,
        coordinate: list,
        geom_types: list[str],
        radii: list[float],
        grad_radius: list,
        grad_dim: list,
        grad_mat: list,
        uncertainty_node: float,
        _verbose: int,
):
    """Validate inputs for the class constructor."""

    if not isinstance(pos, list) or len(pos) != 3:
        raise TypeError(f"'pos' must be a list of length 3, got {pos}")

    if not isinstance(initial_size, list) or len(initial_size) != 3:
        raise TypeError(f"'initial_size' must be a list of length 3, got {initial_size}")

    if not isinstance(coordinate, list) or len(coordinate) != 3:
        raise TypeError(f"'coordinate' must be a list of length 3, got {coordinate}")

    if not isinstance(geom_types, list) or not all(isinstance(x, str) for x in geom_types):
        raise TypeError(f"'geom_types' must be a list of str, got {geom_types}")

    if not isinstance(radii, list) or not all(isinstance(x, (float, int)) for x in radii):
        raise TypeError(f"'radii' must be a list of float, got {radii}")

    if grad_radius is not None and not isinstance(grad_radius, list):
        raise TypeError(f"'grad_radius' must be a list or None, got {grad_radius}")

    if grad_dim is not None and not isinstance(grad_dim, list):
        raise TypeError(f"'grad_dim' must be a list or None, got {grad_dim}")

    if grad_mat is not None and not isinstance(grad_mat, list):
        raise TypeError(f"'grad_mat' must be a list or None, got {grad_mat}")

    if not isinstance(uncertainty_node, (float, int)):
        raise TypeError(f"'uncertainty_node' must be a float, got {uncertainty_node}")

    if not isinstance(_verbose, int):
        raise TypeError(f"'_verbose' must be an int, got {_verbose}")


def function_penalization_Lzone(radius: float, angle: float) -> float:
    """
    Calculate the penalization length based on radii and angle.

    Parameters:
    -----------
    radius: float
        Radius of the beam.
    angle: float
        Angle in degrees.

    Returns:
    -----------
        float: Length of the penalization zone.
    """
    # Case beam quasi-aligned, avoid division by zero
    if angle > 170:
        return 0.0000001
    if angle == 0.0:
        return 0.0
    return radius / math.tan(math.radians(angle) / 2)



def save_lattice_object(lattice, file_name: str = "LatticeObject") -> None:
    """
    Save ONLY the base `Lattice` state to a pickle, even if `lattice` is an instance
    of a subclass (e.g., LatticeSim/LatticeOpti) carrying non-picklable fields.

    Important: converts internal sets (beams/nodes and per-cell containers) to lists
    before pickling to avoid hashing during unpickling. A marker `_pickle_format`
    is stored so the loader can restore sets later.
    Note: this function does NOT save the full state of subclasses, only the base
    `Lattice` attributes. (TO UPDATE if needed)

    Parameters:
    -----------
    lattice: Lattice
        Lattice object to save.
    file_name: str
        Name of the pickle file to save.
    """
    def _find_base_lattice_cls(obj):
        for cls in obj.__class__.__mro__:
            if cls.__name__ == "Lattice" and getattr(cls, "__module__", "").endswith("pyLattice.lattice"):
                return cls
        return None

    def _extract_base_lattice(obj):
        base_cls = _find_base_lattice_cls(obj)
        if base_cls is not None and obj.__class__ is not base_cls:
            base = object.__new__(base_cls)
            base_attrs = [
                "_verbose",
                "name_lattice",
                "x_min", "y_min", "z_min", "x_max", "y_max", "z_max",
                "cell_size_x", "cell_size_y", "cell_size_z",
                "num_cells_x", "num_cells_y", "num_cells_z",
                "size_x", "size_y", "size_z",
                "radii", "geom_types",
                "grad_dim", "grad_radius", "grad_mat",
                "symmetry_lattice", "uncertainty_node", "eraser_blocks",
                "enable_periodicity", "enable_simulation_properties",
                "_simulation_flag", "_optimization_flag",
                "cells", "beams", "nodes",
                "edge_tags", "face_tags", "corner_tags",
                "lattice_dimension_dict"
            ]
            for a in base_attrs:
                setattr(base, a, getattr(obj, a, None))
        else:
            base = obj

        # per-cell containers (after sets->lists conversion)
        for c in getattr(base, "cells", []) or []:
            # drop pointers back to lattice/parents
            for attr in ("lattice", "parent", "owner", "_lattice_ref"):
                if hasattr(c, attr):
                    setattr(c, attr, None)

            # beams inside the cell
            beams_list = getattr(c, "beams_cell", []) or []
            for b in beams_list:
                for attr in ("cell", "cells", "lattice", "owner", "parent", "_cell_ref", "_lattice_ref"):
                    if hasattr(b, attr):
                        setattr(b, attr, None)

            # points/nodes inside the cell
            pts_list = getattr(c, "points_cell", []) or []
            for p in pts_list:
                for attr in ("cell", "cells", "lattice", "owner", "parent", "incident_beams", "_lattice_ref"):
                    if hasattr(p, attr):
                        setattr(p, attr, None)
                # EXCLUDE connected_beams from pickle (breaks deep cycles)
                if hasattr(p, "connected_beams"):
                    try:
                        p.connected_beams.clear()
                    except Exception:
                        pass
                    p.connected_beams = None

        # top-level nodes/beams as well (if present independently of cells)
        for b in getattr(base, "beams", []) or []:
            for attr in ("cell", "cells", "lattice", "owner", "parent", "_cell_ref", "_lattice_ref"):
                if hasattr(b, attr):
                    setattr(b, attr, None)

        for n in getattr(base, "nodes", []) or []:
            for attr in ("cell", "cells", "lattice", "owner", "parent", "incident_beams", "_lattice_ref"):
                if hasattr(n, attr):
                    setattr(n, attr, None)
            # EXCLUDE connected_beams from pickle (breaks deep cycles)
            if hasattr(n, "connected_beams"):
                try:
                    n.connected_beams.clear()
                except Exception:
                    pass
                n.connected_beams = None

        if isinstance(getattr(base, "beams", None), set):
            base.beams = list(base.beams)
        if isinstance(getattr(base, "nodes", None), set):
            base.nodes = list(base.nodes)
        for c in getattr(base, "cells", []) or []:
            if isinstance(getattr(c, "beams_cell", None), set):
                c.beams_cell = list(c.beams_cell)
            if isinstance(getattr(c, "points_cell", None), set):
                c.points_cell = list(c.points_cell)
        setattr(base, "_pickle_format", "lattice_v2_lists")
        return base

    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "outputs" / "saved_lattice_file" / file_name
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    lattice_to_save = _extract_base_lattice(lattice)

    def _diagnose_pickle_issue(root, max_depth: int = 10, max_paths: int = 100) -> None:
        """
        Heuristic traversal to localize recursion/cycles and unpicklable attributes.
        Instead of bailing out when pickle raises RecursionError at the root,
        this version continues to descend into children to report the *paths* that
        introduce back-references (e.g., '.cells[3].beams_cell[5].lattice').
        """
        import pickle
        from collections import Counter, deque

        seen: set[int] = set()
        stack: list[int] = []
        issues: list[tuple[str, str, str]] = []
        type_hits = Counter()

        def add_issue(kind: str, path: list[str], note: str = ""):
            issues.append((kind, " ".join(path), note))
            if len(issues) >= max_paths:
                raise StopIteration

        def render_path(path: list[str]) -> str:
            # compact " .a ['k'] [2] " formatting
            out = []
            for token in path:
                if token.startswith(".") or token.startswith("["):
                    out.append(token)
                else:
                    out.append(token)
            return "".join(out)

        def children(obj):
            # yield (token, child) pairs
            if isinstance(obj, (list, tuple, set, frozenset)):
                for i, v in enumerate(list(obj)[:200]):
                    yield f"[{i}]", v
            elif isinstance(obj, dict):
                for k, v in list(obj.items())[:200]:
                    # keep keys printable and short
                    kk = repr(k)
                    if len(kk) > 40:
                        kk = kk[:37] + "..."
                    yield f"[{kk}]", v
            else:
                d = getattr(obj, "__dict__", None)
                if isinstance(d, dict):
                    for k, v in list(d.items())[:200]:
                        yield f".{k}", v

        def dfs(obj, depth: int, path: list[str]):
            oid = id(obj)
            type_hits[type(obj).__name__] += 1

            if oid in stack:
                # explicit cycle
                add_issue("CYCLE", [render_path(path)], "(back-reference)")
                return
            if depth > max_depth:
                add_issue("DEPTH_LIMIT", [render_path(path)], "truncated")
                return

            stack.append(oid)
            try:
                try:
                    # Try pickling this node; if it explodes, dive into children instead of stopping.
                    pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                except RecursionError as re:
                    # Record, but *also* dive to find the exact attribute(s)
                    add_issue("RECURSION", [render_path(path)], repr(re))
                    # continue to children to localize
                    pass
                except Exception as ex:
                    # Not directly picklable; we’ll dive
                    add_issue("UNPICKLABLE", [render_path(path)], f"type={type(obj).__name__}: {ex!r}")

                # Heuristics: attributes commonly causing back-refs
                if hasattr(obj, "__dict__"):
                    for bad in ("lattice", "parent", "owner", "_lattice_ref", "_cell_ref", "cells", "incident_beams"):
                        if bad in obj.__dict__:
                            add_issue("SUSPECT_ATTR", [render_path(path + [f'.{bad}'])], f"type={type(obj).__name__}")

                # Traverse children
                for tok, child in children(obj):
                    cid = id(child)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    dfs(child, depth + 1, path + [tok])
            finally:
                stack.pop()

        try:
            dfs(root, 0, [type(root).__name__])
        except StopIteration:
            pass

        # Print summary
        if issues:
            print("\n[Pickle diagnostics] Suspicious paths:")
            for kind, p, note in issues:
                print(f" - {kind}: {p} {note}")
            common = ", ".join(f"{k}:{v}" for k, v in type_hits.most_common(10))
            print(f"[Pickle diagnostics] Top types visited: {common}")
        else:
            print("\n[Pickle diagnostics] No obvious culprit found.")

    try:
        with open(path, "wb") as file:
            pickle.dump(lattice_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
            print("[save_lattice_object] Pickle failed:", repr(e))
            _diagnose_pickle_issue(lattice_to_save, max_depth=10, max_paths=100)
            raise RuntimeError(f"Failed to save lattice pickle to {path}: {e}")

    print(f"Lattice (base state) pickle saved successfully to {path}")


def _prepare_lattice_plot_data(beam, deformedForm: bool = False):
    beamDraw = set()
    lines = []
    index = []
    nodes = []

    if beam.radius != 0.0 and beam not in beamDraw:
        node1 = beam.point1.deformed_coordinates if deformedForm else (beam.point1.x, beam.point1.y, beam.point1.z)
        node2 = beam.point2.deformed_coordinates if deformedForm else (beam.point2.x, beam.point2.y, beam.point2.z)
        lines.append([node1, node2])
        nodes.append(beam.point1)
        nodes.append(beam.point2)
        index.append(beam.point1.index)
        index.append(beam.point2.index)

    return lines, nodes, index


def _get_beam_color(beam, color_palette, beamColor, idxColor, cells, nbRadiusBins):
    beamColor = beamColor.lower()

    def _to_scalar_radius(r):
        arr = np.atleast_1d(r)
        return float(arr[0])

    if beamColor == "material":
        mat = int(getattr(beam, "material", 0))
        colorBeam = color_palette[mat % len(color_palette)]

    elif beamColor == "type":
        t = int(getattr(beam, "type_beam", getattr(beam, "geom_types", 0)))
        colorBeam = color_palette[t % len(color_palette)]

    elif beamColor == "radii":
        r = _to_scalar_radius(getattr(beam, "radius", 0.0))
        if r not in idxColor:
            idxColor.append(r)
        colorBeam = color_palette[idxColor.index(r) % len(color_palette)]

    elif beamColor == "radiusbin":
        # Construire les bords des classes (bin edges) paresseusement dans idxColor
        if not idxColor:
            all_radii = [
                _to_scalar_radius(getattr(b, "radius", 0.0))
                for c in cells for b in c.beams
                if _to_scalar_radius(getattr(b, "radius", 0.0)) > 0.0
            ]
            if not all_radii:
                idxColor = [0.0, 1.0]
            else:
                min_r, max_r = min(all_radii), max(all_radii)
                # nbRadiusBins classes => nbRadiusBins+1 bornes
                idxColor = list(np.linspace(min_r, max_r, nbRadiusBins + 1))

        r = _to_scalar_radius(getattr(beam, "radius", 0.0))
        bin_idx = np.digitize([r], idxColor, right=False)[0] - 1
        bin_idx = max(0, min(len(idxColor) - 2, bin_idx))
        colorBeam = color_palette[bin_idx % len(color_palette)]

    else:
        colorBeam = "blue"

    return colorBeam, idxColor


def get_boundary_condition_color(fixed_DOF: list[bool]) -> str:
    """
    Generate a color based on the fixed DOFs using a bitmask approach.
    """
    # Convert fixed_DOF to a bitmask integer
    bitmask = sum(2**i for i, val in enumerate(fixed_DOF) if val)

    # Create a color palette (reproducible)
    base_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_index = bitmask % len(base_colors)

    return base_colors[color_index]


def visualize_lattice_3D_interactive(lattice, beamColor: str = "Material", voxelViz: bool = False,
                                     deformedForm: bool = False, plotCellIndex: bool = False) -> "go.Figure":
    """
    Visualizes the lattice in 3D using Plotly.

    Parameters:
    -----------
    beamColor: string (default: "Material")
        "Material" -> color by material
        "Type" -> color by type_beam
    voxelViz: boolean (default: False)
        True -> voxel visualization
        False -> beam visualization
    deformedForm: boolean (default: False)
        True -> deformed form
    plotCellIndex: boolean (default: False)
        True -> plot the index of each cell
    """

    color_list = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    fig = go.Figure()

    if not voxelViz:
        beamDraw = set()
        nodeDraw = set()
        node_coords = []
        node_colors = []
        lines_x = []
        lines_y = []
        lines_z = []
        line_colors = []
        node1 = None
        node2 = None

        for cell in lattice.cells:
            for beam in cell.beams:
                if beam not in beamDraw:
                    if deformedForm:
                        node1 = beam.point1.deformed_coordinates
                        node2 = beam.point2.deformed_coordinates
                    else:
                        node1 = (beam.point1.x, beam.point1.y, beam.point1.z)
                        node2 = (beam.point2.x, beam.point2.y, beam.point2.z)

                    # Add the beam to the figure
                    lines_x.extend([node1[0], node2[0], None])
                    lines_y.extend([node1[1], node2[1], None])
                    lines_z.extend([node1[2], node2[2], None])

                    # Determine the color of the beam
                    if beamColor == "Material":
                        colorBeam = color_list[beam.material % len(color_list)]
                    elif beamColor == "Type":
                        colorBeam = color_list[beam.type_beam % len(color_list)]
                    else:
                        colorBeam = 'grey'

                    line_colors.extend([colorBeam, colorBeam, colorBeam])

                    beamDraw.add(beam)

                # Add the nodes to the figure
                for node in [node1, node2]:
                    if node not in nodeDraw:
                        node_coords.append(node)
                        nodeDraw.add(node)
                        # Determine the color of the node
                        node_colors.append('black')

            if plotCellIndex:
                cell_center = cell.center_point
                fig.add_trace(go.Scatter3d(
                    x=[cell_center[0]],
                    y=[cell_center[1]],
                    z=[cell_center[2]],
                    mode='text',
                    text=str(cell.index),
                    textposition="top center",
                    showlegend=False
                ))

        # Add the beams to the figure
        fig.add_trace(go.Scatter3d(
            x=lines_x,
            y=lines_y,
            z=lines_z,
            mode='lines',
            line=dict(color=line_colors, width=5),
            hoverinfo='none',
            showlegend=False
        ))

        # Add the nodes to the figure
        if node_coords:
            node_x, node_y, node_z = zip(*node_coords)
            fig.add_trace(go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers',
                marker=dict(size=4, color=node_colors),
                hoverinfo='none',
                showlegend=False
            ))

    else:
        # Vizualize the lattice as a voxel grid
        for cell in lattice.cells:
            x, y, z = cell.coordinate
            dx, dy, dz = cell.size

            if beamColor == "Material":
                colorCell = color_list[cell.beams[0].material % len(color_list)]
            elif beamColor == "Type":
                colorCell = color_list[int(str(cell.geom_types)[0]) % len(color_list)]
            else:
                colorCell = 'grey'

            # Create the voxel
            fig.add_trace(go.Mesh3d(
                x=[x, x + dx, x + dx, x, x, x + dx, x + dx, x],
                y=[y, y, y + dy, y + dy, y, y, y + dy, y + dy],
                z=[z, z, z, z, z + dz, z + dz, z + dz, z + dz],
                color=colorCell,
                opacity=0.5,
                showlegend=False
            ))

    # Configure the layout
    limMin = min(lattice.x_min, lattice.y_min, lattice.z_min)
    limMax = max(lattice.x_max, lattice.y_max, lattice.z_max)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[limMin, limMax], backgroundcolor='white', showgrid=True, zeroline=True),
            yaxis=dict(title='Y', range=[limMin, limMax], backgroundcolor='white', showgrid=True, zeroline=True),
            zaxis=dict(title='Z', range=[limMin, limMax], backgroundcolor='white', showgrid=True, zeroline=True),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )

    return fig  # Return the figure


def save_JSON_to_Grasshopper(lattice, nameLattice: str = "LatticeObject", multipleParts: int = 1) -> None:
    """
    Save the current lattice object to JSON files for Grasshopper compatibility, separating by cells.

    Parameters:
    -----------
    lattice: Lattice
        Lattice object to save.
    nameLattice: str
        Name of the lattice file to save.
    multipleParts: int, optional (default: 1)
        Number of parts to save.
    """
    folder_path = Path(__file__).resolve().parents[2] / "data" / "outputs" / "saved_lattice_file"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    numberCell = len(lattice.cells)
    cellsPerPart = max(1, numberCell // multipleParts)

    for partIdx in range(multipleParts):
        partName = f"{nameLattice}_part{partIdx + 1}.json" if multipleParts > 1 else f"{nameLattice}.json"
        file_pathJSON = os.path.join(folder_path, partName)

        partNodesX = []
        partNodesY = []
        partNodesZ = []
        partRadius = []

        startIdx = partIdx * cellsPerPart
        endIdx = min((partIdx + 1) * cellsPerPart, numberCell)

        for cell in lattice.cells[startIdx:endIdx]:
            for beam in cell.beams_cell:
                partNodesX.append(beam.point1.x)
                partNodesX.append(beam.point2.x)
                partNodesY.append(beam.point1.y)
                partNodesY.append(beam.point2.y)
                partNodesZ.append(beam.point1.z)
                partNodesZ.append(beam.point2.z)
                partRadius.append(beam.radius)

        obj = {
            "nodesX": partNodesX,
            "nodesY": partNodesY,
            "nodesZ": partNodesZ,
            "radii": partRadius,
            "maxX": lattice.x_max,
            "minX": lattice.x_min,
            "maxY": lattice.y_max,
            "minY": lattice.y_min,
            "maxZ": lattice.z_max,
            "minZ": lattice.z_min,
            "relativeDensity": lattice.get_relative_density()
        }

        with open(file_pathJSON, 'w') as f:
            json.dump(obj, f)

        print(f"Saved lattice part {partIdx + 1} to {file_pathJSON}")


def plot_coordinate_system(ax):
    """
    Plot a 3D coordinate system with arrows representing the X, Y, and Z axes.
    """
    origin = [0, 0, 0]
    axis_length = 0.8

    ax.quiver(*origin, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(*origin, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(*origin, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)

    ax.text(origin[0] + axis_length, origin[1], origin[2], "X", color='r')
    ax.text(origin[0], origin[1] + axis_length, origin[2], "Y", color='g')
    ax.text(origin[0], origin[1], origin[2] + axis_length, "Z", color='b')

def _discard(container, item):
    if container is None:
        return
    if isinstance(container, set):
        container.discard(item)
    else:
        try:
            container.remove(item)
        except ValueError:
            pass

