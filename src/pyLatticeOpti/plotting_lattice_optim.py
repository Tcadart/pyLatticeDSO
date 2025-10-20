import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Use TkAgg backend for interactive plots, unless MPLBACKEND is already set
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('TkAgg')


class OptimizationPlotter:
    """
    Class to plot the optimization progress of a lattice structure.

    Parameters
    ----------
    lattice : LatticeOpti
        The lattice optimization object.
    enable_field : bool, optional
        Whether to enable the radius field subplot (default: False).
    """
    def __init__(self, lattice, enable_field: bool = False):
        self.lat = lattice
        self.enable_field = enable_field

        if self.enable_field:
            self.fig, (self.ax, self.ax_func) = plt.subplots(
                2, 1, figsize=(9, 10),
                gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True
            )
        else:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(9, 6), constrained_layout=True
            )
            self.ax_func = None  # placeholders for disabled subplot

        self.ax.set_title("Optimization Progress", fontsize=16)
        self.ax.set_xlabel("Iterations", fontsize=12)
        self.ax.set_ylabel("Compliance (normalized)", fontsize=12)
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Relative Density", fontsize=12)
        (self.line_obj,) = self.ax.plot([], [], 'bo-', label="Compliance")
        (self.line_den,) = self.ax2.plot([], [], 'go--', label="Density")
        self.ax.yaxis.label.set_color('blue')
        self.ax.tick_params(axis='y', colors='blue')
        self.ax2.yaxis.label.set_color('green')
        self.ax2.tick_params(axis='y', colors='green')

        # ---------- radius field subplot (optional) ----------
        if self.enable_field:
            xs = [c.center_point[0] for c in self.lat.cells]
            ys = [c.center_point[1] for c in self.lat.cells]
            zs = [c.center_point[2] for c in self.lat.cells]
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            self._z = float(np.mean(zs))
            nx = max(50, min(150, len(set(xs)) * 5))
            ny = max(50, min(150, len(set(ys)) * 5))
            X = np.linspace(xmin, xmax, nx)
            Y = np.linspace(ymin, ymax, ny)
            self.XX, self.YY = np.meshgrid(X, Y)
            self.extent = [xmin, xmax, ymin, ymax]

            self.ax_func.set_title(f"Radius field on plane z={self._z:.3g}", fontsize=14)
            self.ax_func.set_xlabel("x")
            self.ax_func.set_ylabel("y")
            Z0 = np.full_like(self.XX, np.nan, dtype=float)
            self.im = self.ax_func.imshow(Z0, origin="lower", extent=self.extent, aspect="auto")
            self.cb = self.fig.colorbar(self.im, ax=self.ax_func, fraction=0.046, pad=0.04)
            self.cb.set_label("r(x,y,z)")
            self.text_eq = self.ax_func.text(
                0.02, 0.98, "", transform=self.ax_func.transAxes,
                fontsize=12, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
            )
        else:
            self._z = None
            self.XX = self.YY = None
            self.extent = None
            self.im = None
            self.cb = None
            self.text_eq = None
        # -----------------------------------------------------

        self.obj_hist = []
        self.den_hist = []
        plt.ion()
        plt.show(block=False)


    def _format_equation(self, theta):
        info = self.lat.radius_field_info
        if info.get("type") == "linear":
            parts = []
            for i, d in enumerate(info.get("dirs", [])):
                parts.append(f"{float(theta[i]):.3g}\\,{d}")
            parts.append(f"{float(theta[-1]):.3g}")
            body = " + ".join(parts).replace("+ -", "- ")
            return rf"$r(x,y,z)= {body}$"
        if info.get("type") == "poly2":
            pretty = {"x":"x","y":"y","z":"z","x2":"x^2","y2":"y^2","z2":"z^2","xy":"xy","xz":"xz","yz":"yz"}
            parts = []
            for i, t in enumerate(info.get("terms", [])):
                parts.append(f"{float(theta[i]):.3g}\\,{pretty[t]}")
            parts.append(f"{float(theta[-1]):.3g}")
            body = " + ".join(parts).replace("+ -", "- ")
            return rf"$r(x,y,z)= {body}$"
        return r"$r(x,y,z)$"

    def update(self, objective_norm: float, theta):
        """
        Update the plots with new data.

        Parameters
        ----------
        objective_norm : float
            The normalized objective value.
        theta : array_like
            The current parameters for the radius field.
        """

        def _nice_limits(vals, frac=0.1):
            vmin = float(np.nanmin(vals));
            vmax = float(np.nanmax(vals))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return None
            if vmin == vmax:
                span = 0.2 * (abs(vmin) if vmin != 0 else 1.0)
                return vmin - span, vmax + span
            pad = frac * (vmax - vmin)
            return vmin - pad, vmax + pad

        self.obj_hist.append(objective_norm)
        self.den_hist.append(self.lat.get_relative_density())
        it = list(range(len(self.obj_hist)))

        self.line_obj.set_data(it, self.obj_hist)
        self.line_den.set_data(it, self.den_hist)

        ylims_obj = _nice_limits(self.obj_hist, frac=0.1)
        if ylims_obj: self.ax.set_ylim(*ylims_obj)
        ylims_den = _nice_limits(self.den_hist, frac=0.1)
        if ylims_den: self.ax2.set_ylim(*ylims_den)
        self.ax.set_xlim(0, max(5, len(it) - 1))

        # ---- optional field subplot update ----
        if self.enable_field and self.lat.radius_field is not None:
            ZZ = self.lat.radius_field(self.XX, self.YY, self._z, np.asarray(theta, float))
            self.im.set_data(ZZ)
            self.im.set_extent(self.extent)
            self.im.set_clim(self.lat.min_radius, self.lat.max_radius)
            self.cb.update_normal(self.im)
            self.text_eq.set_text(self._format_equation(theta))
        # --------------------------------------

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            self.fig.canvas.draw()

    def finalize(self, block: bool = True):
        """
        Finalize the plotting.

        Parameters
        ----------
        block : bool, optional
            Whether to block the execution when showing the final plot (default: True).
        """
        self.ax.set_title("Optimization Finished", fontsize=16, color="darkgreen")

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            if block:
                plt.show()
        except Exception:
            pass
