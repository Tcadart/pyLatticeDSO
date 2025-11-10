# =============================================================================
# CLASS: LatticePlotting
#
# Visualization and saving of lattice structures from lattice objects.
# =============================================================================
from typing import Tuple, TYPE_CHECKING

import numpy as np
from colorama import Fore
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from .lattice import Lattice

from .cell import Cell
from .utils import _get_beam_color, _prepare_lattice_plot_data, plot_coordinate_system, get_boundary_condition_color



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
        plt, _, _, _ = self._ensure_matplotlib()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.initFig = True
        self.ax.set_axis_off()

    def _ensure_matplotlib(self):
        """Lazy-import Matplotlib; choose Agg only if no display is available."""
        import importlib
        import matplotlib
        if not self._is_display_available():
            try:
                matplotlib.use('Agg', force=True)  # headless builds (e.g., pdoc, CI)
            except Exception:
                pass
        plt = importlib.import_module('matplotlib.pyplot')
        mcolors = importlib.import_module('matplotlib.colors')
        art3d = importlib.import_module('mpl_toolkits.mplot3d.art3d')
        return plt, mcolors, art3d.Line3DCollection, art3d.Poly3DCollection

    def visualize_lattice(self, lattice_object,
                          beam_color_type: str = "radii",
                          deformed_form: bool = False,
                          file_save_path: str = None,
                          cell_index: bool = False,
                          node_index: bool = False,
                          domain_decomposition_simulation_plotting: bool = False,
                          enable_system_coordinates: bool = True,
                          enable_boundary_conditions: bool = False,
                          camera_position: Tuple[float, float] = None,
                          plotting: bool = True) -> None:
        """
        Visualizes the lattice in 3D using matplotlib.

        Parameters:
        -----------
        lattice_object: Lattice
            The lattice object to visualize.

        beam_color_type: str, optional (default: "radii")
            Color scheme for beams. Options:
            - "radii" or "radius": Color by radii.
            - "material": Color by material.
            - "type": Color by type_beam.

        deformed_form: bool, optional (default: False)
            If True, use deformed node positions.

        file_save_path: str, optional
            If provided, save the plot with this file path.

        cell_index: bool, optional (default: False)
            If True, plot cell indices.

        node_index: bool, optional (default: False)
            If True, plot node indices.

        domain_decomposition_simulation_plotting: bool, optional (default: False)
            If True, indicates that the lattice is part of a domain decomposition simulation.

        enable_system_coordinates: bool, optional (default: True)
            If True, plot the coordinate system axes.

        enable_boundary_conditions: bool, optional (default: False)
            If True, visualize boundary conditions on nodes.

        camera_position: Tuple[float, float], optional
            If provided, set the camera position for the 3D plot as (elevation, azimuth).

        plotting: bool, optional (default: True)
            If True, display the plot after creation.
        """
        if (getattr(lattice_object, "domain_decomposition_solver", None) is True and
                domain_decomposition_simulation_plotting is False):
            print(Fore.YELLOW + "Warning: Plotting domain decomposition lattice without enabling "
                                "domain_decomposition_simulation_plotting flag." + Fore.RESET)


        if not self.initFig:
            self.init_figure()

        ddm_plot = self._is_ddm_plot(lattice_object, domain_decomposition_simulation_plotting)
        cells = lattice_object.cells

        beam_color_type = beam_color_type.lower()
        list_of_valid_types = ("radii", "radius", "material", "type", "radiusbin")
        if beam_color_type not in list_of_valid_types:
            raise ValueError(f"Invalid beam_color_type '{beam_color_type}'. "
                             f"Valid options are: {list_of_valid_types}.")
        if beam_color_type == "radius":
            beam_color_type = "radii"

        cmap, norm, smap, _ = (self._prepare_radius_colormap(cells)
                               if beam_color_type == "radii" else (None, None, None, None))

        legend_map, legend_map_beams, total_beam_labels = ({}, {}, set())

        if ddm_plot:
            self._plot_ddm_mode(
                cells, deformed_form, cell_index, node_index, enable_boundary_conditions
            )
        else:
            legend_map, legend_map_beams, total_beam_labels = self._plot_beams_mode(
                cells, deformed_form, beam_color_type,
                cmap, norm, cell_index, node_index, enable_boundary_conditions
            )

        self._assemble_legends(legend_map_beams, beam_color_type, total_beam_labels, legend_map)
        self._finalize_plot(lattice_object.lattice_dimension_dict, enable_system_coordinates, smap, camera_position,
                            plotting, file_save_path)

    def visualize_lattice_voxels(self,
                                 lattice_object,
                                 beam_color_type: str = "material",
                                 explode_voxel: float = 0.0,
                                 cell_index: bool = False,
                                 enable_system_coordinates: bool = True,
                                 camera_position: Tuple[float, float] | None = None,
                                 file_save_path: str | None = None,
                                 plotting: bool = True) -> None:
        """
        Visualizes the lattice in voxel mode.

        Parameters:
        -----------
        lattice_object: Lattice
            The lattice object to visualize.

        beam_color_type: str, optional (default: "Material")
            Color scheme for voxels. Options:
            - "Material": Color by material.
            - "Type": Color by type_beam.
            - "radii": Color by radii.

        explode_voxel: float, optional (default: 0.0)
            Amount to offset voxels for better visibility.

        cell_index: bool, optional (default: False)
            If True, plot cell indices.

        use_radius_grad_color: bool, optional (default: False)
            If True, color voxels based on a gradient of their radii.

        enable_system_coordinates: bool, optional (default: True)
            If True, plot the coordinate system axes.

        camera_position: Tuple[float, float], optional
            If provided, set the camera position for the 3D plot as (elevation,
            azimuth).

        file_save_path: str, optional
            If provided, save the plot with this file path.

        plotting: bool, optional (default: True)
            If True, display the plot after creation.
        """
        if not self.initFig:
            self.init_figure()

        cells = lattice_object.cells
        latticeDimDict = lattice_object.lattice_dimension_dict
        beam_color_type_lower = beam_color_type.lower()

        cmap, norm, smap, _ = (self._prepare_radius_colormap(cells)
                               if beam_color_type == "radii" else (None, None, None, None))

        # draw voxels
        self._plot_voxel_mode(
            cells=cells,
            latticeDimDict=latticeDimDict,
            beam_color_type_lower=beam_color_type_lower,
            explode_voxel=explode_voxel,
            cmap=cmap,
            norm=norm,
            plotCellIndex=cell_index,
        )

        # finalize
        self._finalize_plot(
            latticeDimDict=lattice_object.lattice_dimension_dict,
            enable_system_coordinates=enable_system_coordinates,
            smap=smap,
            camera_position=camera_position,
            plotting=plotting,
            file_save_path=file_save_path,
        )


    def visual_cell_zone_blocker(self, lattice, erasedParts: list[tuple]) -> None:
        """
        Visualize the lattice with erased parts

        Parameters:
        -----------
        eraser_blocks: list of tuple
            List of erased parts with (x_start, y_start, z_start, x_dim, y_dim
        """
        plt, _mcolors, _Line3DCollection, Poly3DCollection = self._ensure_matplotlib()
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

    def _set_min_max_axis(self, latticeDimDict: dict) -> None:
        """ Set the min and max axis limits based on lattice dimensions."""
        limMin = min(latticeDimDict["x_min"], latticeDimDict["y_min"], latticeDimDict["z_min"])
        limMax = max(latticeDimDict["x_max"], latticeDimDict["y_max"], latticeDimDict["z_max"])
        self.minAxis = min(limMin, self.minAxis) if self.minAxis is not None else limMin
        self.maxAxis = max(limMax, self.maxAxis) if self.maxAxis is not None else limMax
        self.axisSet = True

    @staticmethod
    def _generate_colors(n: int) -> list[str]:
        """Generate a list of distinct colors."""
        from matplotlib import colors as mcolors
        base = list(mcolors.TABLEAU_COLORS.values())
        if n <= len(base):
            return base[:n]
        extra = list(mcolors.CSS4_COLORS.values())
        return (base + extra)[:n]

    @staticmethod
    def _is_ddm_plot(lattice_object, flag: bool) -> bool:
        """Determine if the plot should be in domain decomposition mode."""
        try:
            return bool(flag and getattr(lattice_object, "domain_decomposition_solver", None))
        except Exception:
            return False

    @staticmethod
    def _prepare_radius_colormap(cells: list["Cell"]):
        """Prepare color map normalization for radius-based visualization."""
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        radii_vals = [
            float(np.atleast_1d(getattr(b, "radius", 0.0))[0])
            for c in cells for b in c.beams_cell if getattr(b, "radius", 0.0) > 0.0
        ]
        if not radii_vals:
            return None, None, None, None

        vmin, vmax = min(radii_vals), max(radii_vals)
        if vmax <= vmin:
            vmax = vmin + 1.0

        cmap = plt.cm.jet
        margin = -0.08 * (vmax - vmin)
        norm = mcolors.Normalize(vmin=vmin + margin, vmax=vmax - margin)
        smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return cmap, norm, smap, (vmin, vmax)

    def _assemble_legends(self, legend_map_beams: dict, beam_color_type_lower: str,
                          total_beam_labels: set, legend_map: dict):
        """
        Assemble legends for the plot based on beam colors and boundary conditions.
        """
        legend_elements_all = []

        if legend_map_beams:
            def _left_edge(lbl: str) -> float:
                try:
                    return float(lbl.split(",")[0].strip("[").strip())
                except Exception:
                    return 0.0

            items = (sorted(legend_map_beams.items(), key=lambda kv: _left_edge(kv[0]))
                     if beam_color_type_lower == "radiusbin"
                     else sorted(legend_map_beams.items(), key=lambda kv: kv[0]))
            legend_elements_all.extend(
                [Line2D([0], [0], lw=2, color=col, label=lab) for lab, col in items]
            )
            if beam_color_type_lower == "radii" and len(total_beam_labels) > len(legend_map_beams):
                legend_elements_all.append(
                    Line2D([0], [0], lw=2, linestyle='--',
                           label=f"+{len(total_beam_labels) - len(legend_map_beams)} other")
                )

        if legend_map:
            bc_elems = [Patch(facecolor=color, label=f"{n} DOF")
                        for n, color in sorted(legend_map.items())]
            bc_elems.append(
                Line2D([0], [0], marker='o', color='k', label="Initial Position",
                       markerfacecolor='none', markersize=8, linestyle='None')
            )
            legend_elements_all.extend(bc_elems)

        if legend_elements_all:
            self.ax.legend(handles=legend_elements_all, title="Legend", loc='upper right')

    def _plot_beams_mode(self, cells, deformedForm, beam_color_type_lower, cmap, norm,
                         plotCellIndex, plotNodeIndex, enable_boundary_conditions):
        """
        Plotting function for beams mode.
        """
        plt, mcolors, Line3DCollection, _Poly3DCollection = self._ensure_matplotlib()
        beamDraw = set()
        lines, colors = [], []
        nodeX, nodeY, nodeZ = [], [], []
        nodeDraw = set()

        color_palette = self._generate_colors(max(len(cells), 20))
        idxColor = []
        legend_map = {}
        legend_map_beams = {}
        total_beam_labels = set()

        for cell in cells:
            for beam in cell.beams_cell:
                if beam.radius != 0.0 and beam not in beamDraw:
                    if beam_color_type_lower == "radii" and cmap is not None and norm is not None:
                        r_val = float(np.atleast_1d(getattr(beam, "radius", 0.0))[0])
                        color_for_plot = cmap(norm(r_val))
                        label = None
                    else:
                        colorBeam, idxColor = _get_beam_color(
                            beam, color_palette, beam_color_type_lower, idxColor, cells
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

                    beam_lines, beam_nodes, beam_indices = _prepare_lattice_plot_data(beam, deformedForm)
                    lines.extend(beam_lines)
                    colors.extend([color_for_plot] * len(beam_lines))

                    for i, node in enumerate(beam_nodes):
                        if node not in nodeDraw:
                            nodeDraw.add(node)
                            nodeX.append(node.x if not deformedForm else node.deformed_coordinates[0])
                            nodeY.append(node.y if not deformedForm else node.deformed_coordinates[1])
                            nodeZ.append(node.z if not deformedForm else node.deformed_coordinates[2])
                            if plotNodeIndex:
                                self.ax.text(nodeX[-1], nodeY[-1], nodeZ[-1], str(beam_indices[i]),
                                             fontsize=5, color='gray')
                            if enable_boundary_conditions and any(node.fixed_DOF):
                                bc_color = get_boundary_condition_color(node.fixed_DOF)
                                nb_fixed = sum(node.fixed_DOF)
                                legend_map.setdefault(nb_fixed, bc_color)
                                self.ax.scatter(nodeX[-1], nodeY[-1], nodeZ[-1], c=bc_color, s=70)
                                for j, (is_fixed, d_val) in enumerate(zip(node.fixed_DOF, node.displacement_vector)):
                                    if is_fixed and abs(d_val) > 1e-10:
                                        self.ax.text(nodeX[-1] + cell.size[0] / 4, nodeY[-1],
                                                     nodeZ[-1] + cell.size[2] / 4,
                                                     f"u{j}={d_val:.2e}", fontsize=10, color=bc_color)
                                if deformedForm:
                                    self.ax.scatter(node.x, node.y, node.z, facecolor='none', edgecolor='k',
                                                    s=70, label="Initial Position")
                    beamDraw.add(beam)

            if plotCellIndex:
                self.ax.text(cell.center_point[0], cell.center_point[1], cell.center_point[2],
                             str(cell.index), color='black', fontsize=10)

        line_collection = Line3DCollection(lines, colors=colors, linewidths=2)
        self.ax.add_collection3d(line_collection)
        self.ax.scatter(nodeX, nodeY, nodeZ, c='black', s=5)
        return legend_map, legend_map_beams, total_beam_labels

    def _plot_voxel_mode(self, cells, latticeDimDict, beam_color_type_lower,
                         explode_voxel, cmap, norm, plotCellIndex):
        """
        Plotting function for voxel mode.
        """
        color_palette = self._generate_colors(max(len(cells), 20))
        for cell in cells:
            x, y, z = cell.coordinate
            dx, dy, dz = cell.size

            if beam_color_type_lower == "radii" and cmap is not None and norm is not None:
                rv = float(np.atleast_1d(getattr(cell, "radii", 0.0))[0]) if cell.beams_cell else 0.0
                colorCell = cmap(norm(rv))
            else:
                if beam_color_type_lower == "material":
                    colorCell = color_palette[cell.beam_material % len(color_palette)]
                elif beam_color_type_lower == "type":
                    colorCell = color_palette[cell.geom_types % len(color_palette)]
                elif beam_color_type_lower == "radii":
                    colorCell = cell.get_RGBcolor_depending_of_radius()
                elif beam_color_type_lower == "random":
                    colorCell = np.random.default_rng().random(3, )
                else:
                    colorCell = "green"

            x_offset = explode_voxel * (x - latticeDimDict["x_min"]) / dx
            y_offset = explode_voxel * (y - latticeDimDict["y_min"]) / dy
            z_offset = explode_voxel * (z - latticeDimDict["z_min"]) / dz
            self.ax.bar3d(x + x_offset, y + y_offset, z + z_offset,
                          dx, dy, dz, color=colorCell, alpha=0.5, shade=True, edgecolor='k')

            if plotCellIndex:
                self.ax.text(cell.center_point[0], cell.center_point[1], cell.center_point[2],
                             str(cell.index), color='black', fontsize=10)

    def _plot_ddm_mode(self, cells, deformedForm, plotCellIndex, plotNodeIndex, enable_boundary_conditions):
        """
        Plotting function for domain decomposition mode.
        """
        plt, mcolors, Line3DCollection, _Poly3DCollection = self._ensure_matplotlib()
        beamDraw = set()
        nodeX, nodeY, nodeZ = [], [], []
        nodeDraw = set()
        corners_by_cell = {}
        color_palette = self._generate_colors(max(len(cells), 20))
        idxColor = []

        for cell in cells:
            for beam in cell.beams_cell:
                if beam.radius != 0.0 and beam not in beamDraw:
                    colorBeam, idxColor = _get_beam_color(beam, color_palette, "material", idxColor, cells)
                    beam_lines, beam_nodes, beam_indices = _prepare_lattice_plot_data(beam, deformedForm)

                    for i, node in enumerate(beam_nodes):
                        if cell.index not in corners_by_cell:
                            corners_by_cell[cell.index] = {}
                        for code in (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007):
                            if code in getattr(node, "local_tag", []):
                                corners_by_cell[cell.index][code] = node
                        if node not in nodeDraw and node.index_boundary is not None:
                            nodeDraw.add(node)
                            nodeX.append(node.x if not deformedForm else node.deformed_coordinates[0])
                            nodeY.append(node.y if not deformedForm else node.deformed_coordinates[1])
                            nodeZ.append(node.z if not deformedForm else node.deformed_coordinates[2])
                            if plotNodeIndex:
                                self.ax.text(nodeX[-1], nodeY[-1], nodeZ[-1], str(beam_indices[i]),
                                             fontsize=5, color='gray')
                            if enable_boundary_conditions and any(node.fixed_DOF):
                                bc_color = get_boundary_condition_color(node.fixed_DOF)
                                self.ax.scatter(nodeX[-1], nodeY[-1], nodeZ[-1], c=bc_color, s=70)
                                for j, (is_fixed, d_val) in enumerate(zip(node.fixed_DOF, node.displacement_vector)):
                                    if is_fixed and abs(d_val) > 1e-10:
                                        self.ax.text(nodeX[-1] + cell.size[0] / 4, nodeY[-1],
                                                     nodeZ[-1] + cell.size[2] / 4,
                                                     f"u{j}={d_val:.2e}", fontsize=10, color=bc_color)
                                if deformedForm:
                                    self.ax.scatter(node.x, node.y, node.z, facecolor='none', edgecolor='k',
                                                    s=70, label="Initial Position")
                    beamDraw.add(beam)

            if plotCellIndex:
                self.ax.text(cell.center_point[0], cell.center_point[1], cell.center_point[2],
                             str(cell.index), color='black', fontsize=10)

        def _pcoord(pt):
            return pt.deformed_coordinates if deformedForm else (pt.x, pt.y, pt.z)

        edge_pairs_codes = [
            (1000, 1001), (1001, 1003), (1003, 1002), (1002, 1000),
            (1004, 1005), (1005, 1007), (1007, 1006), (1006, 1004),
            (1000, 1004), (1001, 1005), (1003, 1007), (1002, 1006),
        ]
        box_segments = []
        for c in cells:
            corners = corners_by_cell.get(c.index, {})
            for a, b in edge_pairs_codes:
                if a in corners and b in corners:
                    p0 = _pcoord(corners[a]);
                    p1 = _pcoord(corners[b])
                    box_segments.append([p0, p1])

        if box_segments:
            self.ax.add_collection3d(Line3DCollection(box_segments, colors='k', linewidths=1.0, linestyles='-'))
        self.ax.scatter(nodeX, nodeY, nodeZ, c='black', s=5)

    def _finalize_plot(self, latticeDimDict, enable_system_coordinates, smap, camera_position,
                       plotting: bool, file_save_path: str | None):
        """
        Finalize the plot by setting axes, adding colorbars, and saving/showing the plot.
        """
        plt, _mcolors, _L3D, _P3D = self._ensure_matplotlib()

        if self.axisSet is False:
            self._set_min_max_axis(latticeDimDict)

        if enable_system_coordinates:
            plot_coordinate_system(self.ax)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim3d(self.minAxis, self.maxAxis)
        self.ax.set_ylim3d(self.minAxis, self.maxAxis)
        self.ax.set_zlim3d(self.minAxis, self.maxAxis)
        self.ax.set_box_aspect([1, 1, 1])

        if smap is not None:
            cbar = self.fig.colorbar(smap, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label("Beam radius")

        if camera_position is not None:
            self.ax.view_init(elev=camera_position[0], azim=camera_position[1])

        if file_save_path is not None:
            plt.savefig(file_save_path)

        if plotting:
            self._show()

    @staticmethod
    def plot_radius_distribution(self, lattice_object: "Lattice", nb_radius_bins: int = 5):
        """
        Plot the radii distribution of beams in the lattice.

        Parameters:
        -----------
        lattice_object: Lattice
            The lattice object containing the cells and beams.

        nb_radius_bins: int, optional (default: 5)
            Number of bins to use for the histogram.
        """
        plt, _mcolors, _L3D, _P3D = self._ensure_matplotlib()

        all_radii = []
        all_volumes = []

        for cell in lattice_object.cells:
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
        bins = np.linspace(np.min(all_radii), np.max(all_radii), nb_radius_bins + 1)
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

    def subplot_lattice_hybrid_geometries(self, lattice: "Lattice", explode_voxel: float = 0.0):
        """
        Create subplots for each geometry in a hybrid lattice structure.
        """
        plt, _mcolors, _L3D, _P3D = self._ensure_matplotlib()
        import matplotlib.cm as cm

        cells = lattice.cells
        if len(cells.geom_types) <= 1:
            print("Lattice is not hybrid; only one geometry type found.")
        latticeDimDict = lattice.lattice_dimension_dict
        rmin = 0
        rmax = 0.1

        # Determine number of geometries
        dimRadius = len(cells[0].radii) if hasattr(cells[0].radii, '__len__') else 1
        fig, axs = plt.subplots(1, dimRadius, figsize=(5 * dimRadius, 5), subplot_kw={'projection': '3d'})
        axs = [axs] if dimRadius == 1 else axs

        for ax in axs:
            ax.set_axis_off()
            # ax.view_init(elev=0, azim=-90)  # elev≈20° for depth; azim=90° => +Y direction
            try:
                ax.set_proj_type('ortho')
            except Exception:
                pass

        for rad in range(dimRadius):
            ax = axs[rad]
            for cell in cells:
                x, y, z = cell.coordinate
                dx, dy, dz = cell.size

                radius = cell.radii
                radius_value = radius[rad] if hasattr(radius, '__len__') else radius

                # Define the colormap and normalize
                colormap = cm.get_cmap('coolwarm')
                radius_norm = (radius_value - rmin) / (rmax - rmin)
                radius_norm = np.clip(radius_norm, 0.0, 1.0)
                colorCell = colormap(radius_norm)

                x_offset = explode_voxel * (x - latticeDimDict["x_min"]) / dx
                y_offset = explode_voxel * (y - latticeDimDict["y_min"]) / dy
                z_offset = explode_voxel * (z - latticeDimDict["z_min"]) / dz

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

    def _is_display_available(self) -> bool:
        import os
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def _enable_interactive_backend(self) -> None:
        """
        Try to switch to an interactive backend when possible.
        Safe no-op if already interactive or if GUI toolkits are unavailable.
        """
        import matplotlib
        backend = (matplotlib.get_backend() or "").lower()
        if any(x in backend for x in ("tkagg", "qt5agg", "qtagg", "macosx")):
            return  # already interactive

        # Prefer TkAgg, then Qt5Agg/QtAgg if available
        for candidate, probe in (("TkAgg", "tkinter"), ("Qt5Agg", "PyQt5"), ("QtAgg", "PyQt4")):
            try:
                __import__(probe)
                matplotlib.use(candidate, force=True)
                return
            except Exception:
                continue
        # If we get here, we keep the current (likely non-interactive) backend

    def _show(self):
        """
        Show the figure if an interactive backend is (or can be) enabled.
        Skips in true headless environments.
        """
        plt, _mcolors, _L3D, _P3D = self._ensure_matplotlib()
        import matplotlib

        if not self._is_display_available():
            # No GUI display available (e.g., server/CI) -> don't attempt to show.
            return

        # Try to switch to an interactive backend if we're still on a non-interactive one
        backend = (matplotlib.get_backend() or "").lower()
        if any(x in backend for x in ("agg", "pdf", "svg")):
            self._enable_interactive_backend()
            # rebind plt after backend switch
            from importlib import import_module
            plt = import_module('matplotlib.pyplot')

        plt.show()


