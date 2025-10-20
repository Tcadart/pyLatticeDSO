"""
Visualization and saving of lattice structures from lattice objects.

Created in 2025-01-16 by Cadart Thomas, University of technology Belfort-Montbéliard.
"""
from typing import Tuple, TYPE_CHECKING
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .cell import Cell

if TYPE_CHECKING:
    from .lattice import Lattice

from .utils import _get_beam_color, _prepare_lattice_plot_data, plot_coordinate_system, get_boundary_condition_color

# Use TkAgg backend for interactive plots, unless MPLBACKEND is already set
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('TkAgg')


class LatticePlotting:
    """
    Class for visualizing lattice structures in 3D.
    """

    def __init__(self, initFig: bool = False):
        if initFig:
            self.init_figure()
        self.fig = None
        self.ax = None
        self.minAxis = None
        self.maxAxis = None
        self.initFig = initFig
        self.axisSet = False

    def init_figure(self):
        """Initialize the 3D figure for plotting."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.initFig = True
        self.ax.set_axis_off()

    def _set_min_max_axis(self, latticeDimDict: dict) -> None:
        limMin = min(latticeDimDict["x_min"], latticeDimDict["y_min"], latticeDimDict["z_min"])
        limMax = max(latticeDimDict["x_max"], latticeDimDict["y_max"], latticeDimDict["z_max"])
        self.minAxis = min(limMin, self.minAxis) if self.minAxis is not None else limMin
        self.maxAxis = max(limMax, self.maxAxis) if self.maxAxis is not None else limMax
        self.axisSet = True

    def visualize_lattice(self, lattice_object, beam_color_type: str = "Material",
                             voxelViz: bool = False, deformedForm: bool = False, file_save_path: str = None,
                             plotCellIndex: bool = False, plotNodeIndex: bool = False, explode_voxel: float = 0.0,
                             plotting: bool = True, nbRadiusBins: int = 5,
                             domain_decomposition_simulation_plotting: bool = False,
                             enable_system_coordinates: bool = True, enable_boundary_conditions: bool = False,
                             camera_position: Tuple[float, float] = None, use_radius_grad_color: bool = False) -> None:
        """
        Visualizes the lattice in 3D using matplotlib.

        Parameters:
        -----------
        cells: list of Cell
            List of cells to visualize.
        latticeDimDict: dict
            Dictionary containing lattice dimension information (x_min, x_max, y_min, y_max, z_min, z_max).
        beamColor: str, optional (default: "Material")
            Color scheme for beams. Options:
            - "Material": Color by material.
            - "Type": Color by type_beam.
            - "radii": Color by radii.
        voxelViz: bool, optional (default: False)
            If True, visualize as voxels; otherwise, use beam visualization.
        deformedForm: bool, optional (default: False)
            If True, use deformed node positions.
        nameSave: str, optional
            If provided, save the plot with this name_lattice.
        plotCellIndex: bool, optional (default: False)
            If True, plot cell indices.
        plotNodeIndex: bool, optional (default: False)
            If True, plot node indices.
        explode_voxel: float, optional (default: 0.0)
            If greater than 0, apply an explosion effect to the voxel visualization.
        plotting: bool, optional (default: True)
            If True, display the plot after creation.
        nbRadiusBins: int, optional (default: 5)
            Number of bins for the histogram of radii distribution.
        enable_system_coordinates: bool, optional (default: True)
            If True, plot the coordinate system axes.
        enable_boundary_conditions: bool, optional (default: False)
            If True, visualize boundary conditions on nodes.
        camera_position: Tuple[float, float], optional
            If provided, set the camera position for the 3D plot as (elevation, azimuth).
        """
        if self.initFig is False:
            self.init_figure()

        def generate_colors(n: int) -> list:
            """Generate a list of `n` distinct colors."""
            base_colors = list(mcolors.TABLEAU_COLORS.values())
            if n <= len(base_colors):
                return base_colors[:n]
            return base_colors + list(mcolors.CSS4_COLORS.values())[:n - len(base_colors)]

        domain_decomposition_plotting = False
        try:
            if domain_decomposition_simulation_plotting and lattice_object.domain_decomposition_solver:
                domain_decomposition_plotting = True
        except AttributeError:
            pass

        cells = lattice_object.cells
        latticeDimDict = lattice_object.lattice_dimension_dict

        # Generate a large color palette to avoid missing colors
        max_elements = max(len(cells), 20)  # Dynamically decide the number of colors
        color_palette = generate_colors(max_elements)
        idxColor = []
        legend_map = {}  # {nb_fixed: color}
        legend_map_beams = {}  # {label: color}
        total_beam_labels = set()
        beam_color_type_lower = beam_color_type.lower()

        if use_radius_grad_color:
            # collect all positive radii to set normalization
            radii_vals = []
            for c in cells:
                for b in c.beams_cell:
                    rv = float(np.atleast_1d(getattr(b, "radius", 0.0))[0])
                    if rv > 0.0:
                        radii_vals.append(rv)
            vmin = min(radii_vals) if radii_vals else 0.0
            vmax = max(radii_vals) if radii_vals else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0  # avoid degenerate norm

            cmap = plt.cm.jet
            margin = -0.08 * (vmax - vmin)  # 5% margin at top and bottom
            norm = mcolors.Normalize(vmin=vmin + margin, vmax=vmax - margin)
            smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


        if not voxelViz and not domain_decomposition_plotting:
            beamDraw = set()
            lines = []
            colors = []
            nodeX, nodeY, nodeZ = [], [], []
            nodeDraw = set()

            for cell in cells:
                for beam in cell.beams_cell:
                    if beam.radius != 0.0 and beam not in beamDraw:
                        if use_radius_grad_color:
                            r_val = float(np.atleast_1d(getattr(beam, "radius", 0.0))[0])
                            color_for_plot = cmap(norm(r_val))
                            label = None  # no categorical legend for gradient
                        else:
                            colorBeam, idxColor = _get_beam_color(
                                beam, color_palette, beam_color_type, idxColor, cells, nbRadiusBins
                            )
                            color_for_plot = colorBeam
                            label = None

                            if beam_color_type_lower == "radii":
                                r_val = float(np.atleast_1d(getattr(beam, "radius", 0.0))[0])
                                label = f"r={r_val:g}"
                            elif beam_color_type_lower == "material":
                                label = f"Material {int(getattr(beam, 'material', 0))}"
                            elif beam_color_type_lower == "type":
                                label = f"Type {int(getattr(beam, 'type_beam', getattr(beam, 'geom_types', 0)))}"
                            elif beam_color_type_lower == "radiusbin":
                                edges = idxColor
                                r_val = float(np.atleast_1d(getattr(beam, "radius", 0.0))[0])
                                bi = np.digitize([r_val], edges, right=False)[0] - 1
                                bi = max(0, min(len(edges) - 2, bi))
                                left, right = edges[bi], edges[bi + 1]
                                right_bracket = ']' if bi == len(edges) - 2 else ')'
                                label = f"[{left:.3g}, {right:.3g}{right_bracket}"

                        if label is not None:
                            total_beam_labels.add(label)
                            if beam_color_type_lower in ("material", "type"):
                                legend_map_beams.setdefault(label, color_for_plot)
                            elif beam_color_type_lower == "radii":
                                if label not in legend_map_beams and len(legend_map_beams) < 10:
                                    legend_map_beams[label] = color_for_plot
                            elif beam_color_type_lower == "radiusbin":
                                legend_map_beams.setdefault(label, color_for_plot)


                        # Add line and node data
                        beam_lines, beam_nodes, beam_indices, = _prepare_lattice_plot_data(beam, deformedForm)
                        lines.extend(beam_lines)
                        colors.extend([color_for_plot] * len(beam_lines))

                        for i, node in enumerate(beam_nodes):
                            if node not in nodeDraw:
                                nodeDraw.add(node)
                                nodeX.append(node.x if not deformedForm else node.deformed_coordinates[0])
                                nodeY.append(node.y if not deformedForm else node.deformed_coordinates[1])
                                nodeZ.append(node.z if not deformedForm else node.deformed_coordinates[2])
                                if plotNodeIndex:
                                    self.ax.text(nodeX[-1], nodeY[-1], nodeZ[-1], str(beam_indices[i]), fontsize=5,
                                                 color='gray')
                                if enable_boundary_conditions:
                                    if any(node.fixed_DOF):
                                        bc_color = get_boundary_condition_color(node.fixed_DOF)
                                        nb_fixed = sum(node.fixed_DOF)
                                        legend_map.setdefault(nb_fixed, bc_color)

                                        self.ax.scatter(nodeX[-1], nodeY[-1], nodeZ[-1], c=bc_color, s=70)
                                        # Apply boundary condition labels
                                        for i, (is_fixed, d_val) in enumerate(zip(node.fixed_DOF, node.displacement_vector)):
                                            if is_fixed and abs(d_val) > 1e-10:
                                                self.ax.text(nodeX[-1] + cell.size[0]/4, nodeY[-1], nodeZ[-1] + cell.size[2]/4,
                                                             f"u{i}={d_val:.2e}", fontsize=10, color=bc_color)
                                        if deformedForm:
                                            # plot initial position of nodes of boundary conditions
                                            self.ax.scatter(node.x, node.y, node.z, facecolor='none', edgecolor = 'k',
                                                            s=70, label="Initial Position")

                        beamDraw.add(beam)

                if plotCellIndex:
                    self.ax.text(cell.center_point[0], cell.center_point[1], cell.center_point[2], str(cell.index),
                                 color='black', fontsize=10)

            # Plot lines and nodes
            line_collection = Line3DCollection(lines, colors=colors, linewidths=2)
            self.ax.add_collection3d(line_collection)
            self.ax.scatter(nodeX, nodeY, nodeZ, c='black', s=5)

        elif voxelViz and not domain_decomposition_plotting:  # Voxel visualization
            for cell in cells:
                x, y, z = cell.coordinate
                dx, dy, dz = cell.size

                if use_radius_grad_color:
                    # use first beam's radius (or any representative) for the cell color
                    rv = float(np.atleast_1d(getattr(cell.beams_cell[0], "radius", 0.0))[0]) if cell.beams_cell else 0.0
                    colorCell = cmap(norm(rv))
                else:
                    beam_color_type = beam_color_type.lower()
                    if beam_color_type == "material":
                        colorCell = color_palette[cell.beams_cell[0].material % len(color_palette)]
                    elif beam_color_type == "type":
                        colorCell = color_palette[cell.geom_types % len(color_palette)]
                    elif beam_color_type == "radii":
                        colorCell = cell.get_RGBcolor_depending_of_radius()
                    elif beam_color_type == "random":
                        rng = np.random.default_rng()
                        colorCell = rng.random(3, )
                    else:
                        colorCell = "green"

                x_offset = explode_voxel * (x - latticeDimDict["x_min"]) / dx
                y_offset = explode_voxel * (y - latticeDimDict["y_min"]) / dy
                z_offset = explode_voxel * (z - latticeDimDict["z_min"]) / dz
                self.ax.bar3d(x + x_offset, y + y_offset, z + z_offset,
                              dx, dy, dz, color=colorCell, alpha=0.5, shade=True, edgecolor='k')
        elif domain_decomposition_plotting:
            beamDraw = set()
            nodeX, nodeY, nodeZ = [], [], []
            nodeDraw = set()
            corners_by_cell = {}

            for cell in cells:
                for beam in cell.beams_cell:
                    if beam.radius != 0.0 and beam not in beamDraw:
                        colorBeam, idxColor = _get_beam_color(beam, color_palette, beam_color_type, idxColor, cells,
                                                              nbRadiusBins)

                        # Add line and node data
                        beam_lines, beam_nodes, beam_indices, = _prepare_lattice_plot_data(beam, deformedForm)

                        for i, node in enumerate(beam_nodes):
                            # collect corner nodes for deformed cell-edge plotting
                            if cell.index not in corners_by_cell:
                                corners_by_cell[cell.index] = {}
                            for code in (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007):
                                if code in getattr(node, "local_tag", []):
                                    corners_by_cell[cell.index][code] = node
                            if node not in nodeDraw:
                                if node.index_boundary is not None:
                                    nodeDraw.add(node)
                                    nodeX.append(node.x if not deformedForm else node.deformed_coordinates[0])
                                    nodeY.append(node.y if not deformedForm else node.deformed_coordinates[1])
                                    nodeZ.append(node.z if not deformedForm else node.deformed_coordinates[2])
                                    if plotNodeIndex:
                                        self.ax.text(nodeX[-1], nodeY[-1], nodeZ[-1], str(beam_indices[i]), fontsize=5,
                                                     color='gray')
                                    if enable_boundary_conditions:
                                        if any(node.fixed_DOF):
                                            bc_color = get_boundary_condition_color(node.fixed_DOF)
                                            nb_fixed = sum(node.fixed_DOF)
                                            legend_map.setdefault(nb_fixed, bc_color)

                                            self.ax.scatter(nodeX[-1], nodeY[-1], nodeZ[-1], c=bc_color, s=70)
                                            # Apply boundary condition labels
                                            for i, (is_fixed, d_val) in enumerate(
                                                    zip(node.fixed_DOF, node.displacement_vector)):
                                                if is_fixed and abs(d_val) > 1e-10:
                                                    self.ax.text(nodeX[-1] + cell.size[0] / 4, nodeY[-1],
                                                                 nodeZ[-1] + cell.size[2] / 4,
                                                                 f"u{i}={d_val:.2e}", fontsize=10, color=bc_color)
                                            if deformedForm:
                                                # plot initial position of nodes of boundary conditions
                                                self.ax.scatter(node.x, node.y, node.z, facecolor='none', edgecolor='k',
                                                                s=70, label="Initial Position")


                if plotCellIndex:
                    self.ax.text(cell.center_point[0], cell.center_point[1], cell.center_point[2], str(cell.index),
                                 color='black', fontsize=10)

            # Plot lines and nodes
            # Build deformed (or undeformed) bounding-box edges from corner-tagged nodes
            def _pcoord(pt):
                return pt.deformed_coordinates if deformedForm else (pt.x, pt.y, pt.z)

            # vertex order matching corner codes:
            # bottom: 1000(x0,y0,z0), 1001(x1,y0,z0), 1003(x1,y1,z0), 1002(x0,y1,z0)
            # top:    1004(x0,y0,z1), 1005(x1,y0,z1), 1007(x1,y1,z1), 1006(x0,y1,z1)
            edge_pairs_codes = [
                (1000, 1001), (1001, 1003), (1003, 1002), (1002, 1000),  # bottom
                (1004, 1005), (1005, 1007), (1007, 1006), (1006, 1004),  # top
                (1000, 1004), (1001, 1005), (1003, 1007), (1002, 1006),  # verticals
            ]

            box_segments = []
            for c in cells:
                corners = corners_by_cell.get(c.index, {})
                # only add edges where both corners were identified
                for a, b in edge_pairs_codes:
                    if a in corners and b in corners:
                        p0 = _pcoord(corners[a])
                        p1 = _pcoord(corners[b])
                        box_segments.append([p0, p1])

            if box_segments:
                self.ax.add_collection3d(
                    Line3DCollection(box_segments, colors='k', linewidths=1.0, linestyles='-')
                )

            self.ax.scatter(nodeX, nodeY, nodeZ, c='black', s=5)


        if self.axisSet is False:
            self._set_min_max_axis(latticeDimDict)



        if enable_system_coordinates:
            plot_coordinate_system(self.ax)

        legend_elements_all = []

        if legend_map_beams:
            def _left_edge(lbl: str) -> float:
                try:
                    return float(lbl.split(",")[0].strip("[").strip())
                except Exception:
                    return 0.0

            if beam_color_type_lower == "radiusbin":
                items = sorted(legend_map_beams.items(), key=lambda kv: _left_edge(kv[0]))
            else:
                items = sorted(legend_map_beams.items(), key=lambda kv: kv[0])

            legend_elements_all.extend(
                [Line2D([0], [0], lw=2, color=col, label=lab) for lab, col in items]
            )

            # Pour 'radii' si >10 catégories, indiquer le surplus
            if beam_color_type_lower == "radii" and len(total_beam_labels) > len(legend_map_beams):
                legend_elements_all.append(
                    Line2D([0], [0], lw=2, linestyle='--',
                           label=f"+{len(total_beam_labels) - len(legend_map_beams)} other")
                )

        if enable_boundary_conditions and legend_map:
            bc_elems = [Patch(facecolor=color, label=f"{n} DOF")
                        for n, color in sorted(legend_map.items())]
            bc_elems.append(
                Line2D([0], [0], marker='o', color='k', label="Initial Position",
                       markerfacecolor='none', markersize=8, linestyle='None')
            )
            legend_elements_all.extend(bc_elems)

        if use_radius_grad_color:
            fig_for_cb = getattr(self, "fig", plt.gcf())
            cbar = fig_for_cb.colorbar(smap, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label("Beam radius")

        if legend_elements_all:
            self.ax.legend(handles=legend_elements_all, title="Legend", loc='upper right')

        if camera_position is not None:
            self.ax.view_init(elev=camera_position[0], azim=camera_position[1])

        # Save or show the plot
        if plotting:
            self.show()
        if file_save_path is not None:
            plt.savefig(file_save_path)



    def visual_cell_zone_blocker(self, lattice, erasedParts: list[tuple]) -> None:
        """
        Visualize the lattice with erased parts

        Parameters:
        -----------
        eraser_blocks: list of tuple
            List of erased parts with (x_start, y_start, z_start, x_dim, y_dim
        """

        # Plot global lattice cube
        x_max = lattice.x_max
        y_max = lattice.y_max
        z_max = lattice.z_max
        vertices_global = [[0, 0, 0], [x_max, 0, 0], [x_max, y_max, 0], [0, y_max, 0],
                           [0, 0, z_max], [x_max, 0, z_max], [x_max, y_max, z_max], [0, y_max, z_max]]
        self.ax.add_collection3d(
            Poly3DCollection([vertices_global], facecolors='grey', linewidths=1, edgecolors='black', alpha=0.3))

        # Plot erased region cube
        for erased in erasedParts:
            x_start, y_start, z_start, x_dim, y_dim, z_dim = erased
            vertices_erased = [[x_start, y_start, z_start], [x_start + x_dim, y_start, z_start],
                               [x_start + x_dim, y_start + y_dim, z_start], [x_start, y_start + y_dim, z_start],
                               [x_start, y_start, z_start + z_dim], [x_start + x_dim, y_start, z_start + z_dim],
                               [x_start + x_dim, y_start + y_dim, z_start + z_dim],
                               [x_start, y_start + y_dim, z_start + z_dim]]
            self.ax.add_collection3d(
                Poly3DCollection([vertices_erased], facecolors='red', linewidths=1, edgecolors='black', alpha=0.6))

        # Set labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([0, x_max])
        self.ax.set_ylim([0, y_max])
        self.ax.set_zlim([0, z_max])

        plt.show()

    def plot_radius_distribution(self, cells: list["Cell"], nbRadiusBins: int = 5):
        """
        Plot the radii distribution of beams in the lattice.

        Parameters:
        -----------
        cells: list of Cell
            List of cells to visualize.
        latticeDimDict: dict
            Dictionary containing lattice dimension information (x_min, x_max, y_min, z_min, z_max).
        nbRadiusBins: int
            Number of bins for the histogram.
        """
        all_radii = []
        all_volumes = []

        for cell in cells:
            radius = cell.radii
            if hasattr(radius, '__len__'):
                all_radii.append(radius)
            else:
                all_radii.append([radius])
            all_volumes.append(cell.volume_each_geom)

        all_radii = np.array(all_radii)
        dimRadius = all_radii.shape[1]
        all_volumes = np.array(all_volumes)
        sumVolume = np.sum(all_volumes, axis=0)
        ratio_volume = sumVolume / np.sum(sumVolume) * 100

        colors = plt.cm.tab10.colors  # Up to 10 colors predefined

        plt.figure(figsize=(7, 5))
        bins = np.linspace(np.min(all_radii), np.max(all_radii), nbRadiusBins + 1)
        bin_width = (bins[1] - bins[0]) / (dimRadius + 1)

        for i in range(dimRadius):
            shifted_bins = bins[:-1] + i * bin_width
            plt.bar(shifted_bins, np.histogram(all_radii[:, i], bins=bins)[0],
                    width=bin_width, align='edge', color=colors[i % len(colors)], edgecolor='black',
                    label=f'Geometry {i}, Ratio Volume: {ratio_volume[i]:.2f}%')

        plt.title('radii Distribution')
        plt.xlabel('radii')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    # at the top of the file, you already have: from matplotlib import pyplot as plt

    def subplot_lattice_geometries(self, lattice: "Lattice", explodeVoxel: float = 0.0):
        """
        Create subplots:
        - One subplot per geometry (radii index) with voxel visualization.
        """
        cells = lattice.cells
        latticeDimDict = lattice.lattice_dimension_dict
        rmin = 0
        rmax = 0.1

        # Determine number of geometries
        dimRadius = len(cells[0].radii) if hasattr(cells[0].radii, '__len__') else 1
        fig, axs = plt.subplots(1, dimRadius, figsize=(5 * dimRadius, 5), subplot_kw={'projection': '3d'})
        axs = [axs] if dimRadius == 1 else axs  # Ensure axs is always iterable

        # ---- Camera settings: view along +Y (use azim=-90 for -Y) ----
        for ax in axs:
            ax.set_axis_off()
            # ax.view_init(elev=0, azim=-90)  # elev≈20° for depth; azim=90° => +Y direction
            try:
                ax.set_proj_type('ortho')  # optional: orthographic projection (matplotlib>=3.2)
            except Exception:
                pass

        for rad in range(dimRadius):
            ax = axs[rad]
            for cell in cells:
                x, y, z = cell.coordinate
                dx, dy, dz = cell.size

                # Get color based on the radii value for current geometry
                radius = cell.radii
                radius_value = radius[rad] if hasattr(radius, '__len__') else radius
                import matplotlib.cm as cm

                # Define the colormap and normalize
                colormap = cm.get_cmap('coolwarm')
                radius_norm = (radius_value - rmin) / (rmax - rmin)
                radius_norm = np.clip(radius_norm, 0.0, 1.0)
                colorCell = colormap(radius_norm)

                x_offset = explodeVoxel * (x - latticeDimDict["x_min"]) / dx
                y_offset = explodeVoxel * (y - latticeDimDict["y_min"]) / dy
                z_offset = explodeVoxel * (z - latticeDimDict["z_min"]) / dz

                ax.bar3d(x + x_offset, y + y_offset, z + z_offset, dx, dy, dz,
                         color=colorCell, alpha=0.5, shade=True, edgecolor='k')

            if self.axisSet is False:
                self._set_min_max_axis(latticeDimDict)

            # ax.set_title(f'Geometry {rad}')
            ax.set_xlim3d(self.minAxis, self.maxAxis)
            ax.set_ylim3d(self.minAxis, self.maxAxis)
            ax.set_zlim3d(self.minAxis, self.maxAxis)
            ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    def show(self):
        """
        Show the 3D plot with axis labels and limits.
        """
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim3d(self.minAxis, self.maxAxis)
        self.ax.set_ylim3d(self.minAxis, self.maxAxis)
        self.ax.set_zlim3d(self.minAxis, self.maxAxis)
        self.ax.set_box_aspect([1, 1, 1])
        plt.show()
