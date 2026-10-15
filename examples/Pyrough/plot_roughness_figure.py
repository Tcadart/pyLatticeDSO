"""
Figure of the roughness study (results of BCC_rough_compression.py, to be run first):
(a) smooth and rough (largest amplitude) BCC cells, the rough surface colored by its deviation from the nominal
    strut surface,
(b) apparent modulus against the relative density, rough samples over the smooth radius sweep,
(c) ratios rough / smooth at the same nominal radius and at the same relative density.

Saved as data/outputs/roughness_compression/roughness_figure.pdf (and .png).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LightSource, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyLatticeDesign.materials import MatProperties
from pyLatticeSim.lattice_sim import LatticeSim

project_root = Path(__file__).resolve().parents[2]
save_dir = project_root / "data" / "outputs" / "roughness_compression"
name_file = "Pyrough/BCC_rough_compression"

with open(save_dir / "results.json", "r") as file:
    results = json.load(file)
lattice = LatticeSim(name_file)
radius = lattice.radii[0]
E_s = float(MatProperties(lattice.material_name).young_modulus)

# Smooth radius sweep and power law fit E*(rho) (quadratic in log-log)
smooth = sorted(results["smooth"].values(), key=lambda e: e["radius"])
rho_smooth = np.array([e["relative_density"] for e in smooth])
E_smooth = np.array([e["E"] for e in smooth]) / E_s
fit_E = np.polyfit(np.log(rho_smooth), np.log(E_smooth), 2)
nominal = next(e for e in smooth if np.isclose(e["radius"], radius))

ratios = sorted({round(e["Sq"] / radius, 6) for e in results["rough"].values()})
rough = {ratio: [e for e in results["rough"].values() if np.isclose(e["Sq"] / radius, ratio)] for ratio in ratios}

# Sequential colors (one hue, light -> dark) for the roughness amplitude
colors = {ratio: plt.cm.Blues(0.4 + 0.6 * i / max(len(ratios) - 1, 1)) for i, ratio in enumerate(ratios)}
gray = "#6b6b6b"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(7.2, 4.6))
grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.45)


# =============================================================================
# (a) Smooth and rough cells, the rough surface colored by its deviation from the nominal strut surface
# =============================================================================
axes_A = np.array([beam.point1.coordinates for beam in lattice.beams], dtype=float)
axes_B = np.array([beam.point2.coordinates for beam in lattice.beams], dtype=float)
mins = np.array([lattice.x_min, lattice.y_min, lattice.z_min], dtype=float)
maxs = np.array([lattice.x_max, lattice.y_max, lattice.z_max], dtype=float)


def deviation_from_nominal(points: np.ndarray) -> np.ndarray:
    """Distance to the closest beam axis minus the nominal radius, NaN on the cut faces of the cell."""
    AB = axes_B - axes_A
    t = np.clip(np.einsum("pbj,bj->pb", points[:, None] - axes_A[None], AB) / np.sum(AB ** 2, axis=1)[None], 0, 1)
    dist = np.linalg.norm(points[:, None] - (axes_A[None] + t[..., None] * AB[None]), axis=2).min(axis=1)
    on_cut = np.any(np.isclose(points, mins[None], atol=1e-3) | np.isclose(points, maxs[None], atol=1e-3), axis=1)
    return np.where(on_cut, np.nan, dist - radius)


light = LightSource(azdeg=225, altdeg=45)
dev_max = 4.0 * largest_sq if (largest_sq := max(ratios) * radius) else 1.0
cmap_dev = plt.cm.YlOrBr


def plot_mesh(ax, path: Path, title: str, colored: bool):
    mesh = trimesh.load(path)
    shading = 0.45 + 0.55 * light.shade_normals(mesh.face_normals, fraction=1.0)[:, None]
    base = np.tile(np.array(to_rgb("#b9c2cc")), (len(mesh.faces), 1))
    if colored:
        deviation = deviation_from_nominal(mesh.triangles_center)
        valid = ~np.isnan(deviation)
        base[valid] = cmap_dev(np.clip(deviation[valid] / dev_max, 0, 1))[:, :3]
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolors=np.clip(base * shading, 0, 1),
                                         edgecolor="none"))
    ax.set_xlim(mins[0], maxs[0]), ax.set_ylim(mins[1], maxs[1]), ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, pad=-4)


largest = ratios[-1]
plot_mesh(fig.add_subplot(grid[0, 0], projection="3d"), save_dir / "meshes" / "BCC_smooth_1.000.stl",
          f"Smooth, r = {radius:g}", colored=False)
ax = fig.add_subplot(grid[0, 1], projection="3d")
plot_mesh(ax, save_dir / "meshes" / f"BCC_rough_{largest:.3f}_seed0.stl", fr"Rough, $S_q / r$ = {largest:g}",
          colored=True)
fig.text(0.04, 0.93, "(a)", fontweight="bold")
colorbar_ax = fig.add_axes([0.68, 0.62, 0.012, 0.22])
colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_dev, norm=plt.Normalize(0, dev_max / radius)),
                        cax=colorbar_ax)
colorbar.ax.set_title(r"$(d - r) / r$", fontsize=8, pad=4)
colorbar.ax.tick_params(labelsize=7)

# =============================================================================
# (b) Apparent modulus against the relative density
# =============================================================================
ax = fig.add_subplot(grid[1, 0:2])
rho_plot = np.linspace(rho_smooth.min(), rho_smooth.max(), 200)
ax.plot(rho_plot, np.exp(np.polyval(fit_E, np.log(rho_plot))), "-", color=gray, linewidth=1.5,
        label="Smooth (radius sweep)")
ax.plot(rho_smooth, E_smooth, "o", color=gray, markersize=4)
ax.plot(nominal["relative_density"], nominal["E"] / E_s, "o", color="k", markersize=7,
        label=f"Smooth, nominal radius r = {radius:g}")
for ratio in ratios:
    ax.plot([e["relative_density"] for e in rough[ratio]], [e["E"] / E_s for e in rough[ratio]], "D",
            color=colors[ratio], markeredgecolor="k", markeredgewidth=0.5, markersize=6,
            label=fr"Rough, $S_q / r$ = {ratio:g}")
ax.set_xlabel(r"Relative density $\rho^*$")
ax.set_ylabel(r"Apparent modulus $E^* / E_s$")
ax.grid(alpha=0.3, linewidth=0.5)
ax.legend(loc="upper left", frameon=False)
ax.text(-0.1, 1.04, "(b)", transform=ax.transAxes, fontweight="bold")

# =============================================================================
# (c) Rough / smooth ratios
# =============================================================================
ax = fig.add_subplot(grid[1, 2])
amplitudes = np.array([0.0] + ratios)


def mean_std(values_per_ratio):
    return (np.array([1.0] + [np.mean(v) for v in values_per_ratio]),
            np.array([0.0] + [np.std(v) for v in values_per_ratio]))


series = {
    r"$E^*$, same radius": ([[e["E"] / nominal["E"] for e in rough[r]] for r in ratios], "o-", "#1f4e79"),
    r"$\rho^*$, same radius": ([[e["relative_density"] / nominal["relative_density"] for e in rough[r]]
                                for r in ratios], "^:", gray),
    r"$E^*$, same density": ([[e["E"] / E_s / np.exp(np.polyval(fit_E, np.log(e["relative_density"])))
                               for e in rough[r]] for r in ratios], "s--", "#c0504d"),
}
for label, (values, style, color) in series.items():
    mean, std = mean_std(values)
    ax.errorbar(amplitudes, mean, yerr=std, fmt=style, color=color, markersize=4, capsize=2, linewidth=1.2,
                label=label)
ax.axhline(1.0, color="k", linewidth=0.6)
ax.set_xlabel(r"RMS roughness $S_q / r$")
ax.set_ylabel("Rough / smooth")
ax.grid(alpha=0.3, linewidth=0.5)
ax.set_ylim(0.9, 2.45)
ax.legend(loc="upper left", frameon=False, fontsize=7)
ax.text(-0.3, 1.04, "(c)", transform=ax.transAxes, fontweight="bold")

fig.savefig(save_dir / "roughness_figure.pdf", bbox_inches="tight")
fig.savefig(save_dir / "roughness_figure.png", dpi=300, bbox_inches="tight")
print(f"Figure saved in {save_dir}")
