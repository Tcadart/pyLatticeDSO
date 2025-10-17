import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_optimization_json(json_path: str | Path) -> dict:
    """Load the optimization summary JSON."""
    p = Path(json_path)
    if not p.exists():
        # Also try default folder if a bare filename is passed
        default = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_data_files" / p.name
        if not default.exists():
            raise FileNotFoundError(f"JSON file not found: {p} or {default}")
        p = default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_convergence(data: dict,
                     use_normalized: bool = True,
                     save_path: str | Path | None = None,
                     show: bool = False,
                     title: str | None = None) -> Path | None:
    """
    Plot objective (left axis) and relative density (right axis) over iterations.

    Parameters
    ----------
    data : dict
        Loaded JSON dict from `save_optimization_json`.
    use_normalized : bool
        If True, plot normalized objective when available; otherwise plot raw objective.
    save_path : str | Path | None
        If provided, save figure to this path. If directory, saves as 'optimization_convergence.png' inside it.
    show : bool
        If True, display the figure (requires a GUI backend).
    title : str | None
        Optional custom title.

    Returns
    -------
    Path | None
        Path to saved figure if saved, else None.
    """
    hist = data.get("history", {})
    iters = hist.get("iteration") or list(range(len(hist.get("objective_norm", []))))
    obj_norm = hist.get("objective_norm", [])
    obj_raw = hist.get("objective", [])
    rho = hist.get("relative_density", [])

    if not iters:
        raise ValueError("No iteration history found in JSON.")

    # Choose objective series
    if use_normalized and obj_norm:
        y_obj_all = obj_norm
        obj_label = "Objective (normalized)"
    else:
        y_obj_all = obj_raw if obj_raw else obj_norm
        obj_label = "Objective"

    # Validate/fix the iteration vector so it is 1,2,3,...,N
    y_obj = y_obj_all[1:] if len(y_obj_all) >= 2 else y_obj_all[:]
    N = len(y_obj)
    if N == 0:
        raise ValueError("No objective values to plot after trimming the last sample.")

    iters_raw = hist.get("iteration", None)

    def _is_consecutive(seq):
        try:
            seq_int = [int(v) for v in seq]
            return all((b - a) == 1 for a, b in zip(seq_int, seq_int[1:]))
        except Exception:
            return False

    if isinstance(iters_raw, list):
        if len(iters_raw) == N + 1 and _is_consecutive(iters_raw):
            iters = [int(v) for v in iters_raw[:-1]]  # drop last to match y_obj
        elif len(iters_raw) == N and _is_consecutive(iters_raw):
            iters = [int(v) for v in iters_raw]
        else:
            iters = list(range(1, N + 1))
    else:
        iters = list(range(1, N + 1))

    # Trim relative density to match objective length (it can also have an extra last value)
    rho = (rho[:N]) if isinstance(rho, list) else []

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()

    # Plot objective (left)
    line_obj, = ax.plot(iters, y_obj, linewidth=2, color="blue", label=obj_label)
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel(obj_label, fontsize=16, color=line_obj.get_color())
    ax.tick_params(axis="y", colors=line_obj.get_color(), labelsize =14)
    ax.tick_params(axis="x", labelsize=14)

    # Plot relative density (right)
    if rho:
        line_rho, = ax2.plot(iters[1:], rho[1:], linestyle="--", linewidth=2, color="green",
                             label="Relative density")
        ax2.set_ylabel(r"Relative density $\rho_{rel}$", fontsize=16, color=line_rho.get_color())
        ax2.tick_params(axis="y", colors=line_rho.get_color(), labelsize =14)

        # Optional: target band/line if present
        rd_meta = data.get("relative_density_constraint", {})
        target = rd_meta.get("target", None)
        tol = rd_meta.get("tolerance", 0.0) or 0.0
        mode = rd_meta.get("mode", None)

        if isinstance(target, (int, float)):
            ax2.axhline(target, linestyle=":", linewidth=1.2, color="gray")
            if mode == "band" and tol > 0:
                ax2.axhspan(target - tol, target + tol, alpha=0.1, color="gray")

    # Title & layout
    if title is None:
        otype = data.get("objective_type", "objective")
        sim = data.get("simulation_type", "")
        title = f"Convergence ({otype}, {sim})".strip(", ")
    # ax.set_title(title)
    # Grid (main x-axis and left y-axis)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.grid(False)  # avoid duplicated grid lines on twin axis
    fig.tight_layout()

    # Save or show
    out_path = None
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.is_dir():
            out_path = save_path / "optimization_convergence.png"
        else:
            out_path = save_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path

from datetime import datetime
from pathlib import Path
import json

def extract_final_results(json_path: str | Path) -> tuple[float | None, float | None]:
    """
    Extract final compliance (non-normalized) and total computation time
    from a saved optimization JSON.

    Parameters
    ----------
    json_path : str | Path
        Path to the optimization summary JSON.

    Returns
    -------
    tuple (final_compliance, computation_time)
        final_compliance : float | None
            Final compliance (non-normalized), or None if missing.
        computation_time : float | None
            Total optimization time in seconds, or None if not computable.
    """
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    final_compliance = data.get("solution", {}).get("final_objective", None)

    timestamps = data.get("history", {}).get("timestamp", [])
    if len(timestamps) >= 2:
        try:
            t0 = datetime.fromisoformat(timestamps[0].replace("Z", ""))
            t1 = datetime.fromisoformat(timestamps[-1].replace("Z", ""))
            computation_time = (t1 - t0).total_seconds()
        except Exception:
            computation_time = None
    else:
        computation_time = None

    return final_compliance, computation_time




def main():
    # json_file = "data/outputs/optimization_data_files/Three_point_bending_optimized_expe.json"
    json_file = "data/outputs/optimization_data_files/Three_point_bending_optimized.json"
    # json_file = "data/outputs/optimization_data_files/Three_point_bending_constant_expe.json"
    # json_file = "data/outputs/optimization_data_files/Cantilever_L_beam_optimized_expe.json"
    # json_file = "data/outputs/optimization_data_files/Cantilever_L_beam_constant_expe.json"
    # json_file = "data/outputs/optimization_data_files/Inversion_mechanism_optimized_expe.json"
    # json_file = "data/outputs/optimization_data_files/Inversion_mechanism_constant_expe.json"
    use_raw = False          # True = objectif non-normalisé, False = objectif normalisé
    save_path = "data/outputs/optimization_data_files/figures/"   # None pour ne pas sauvegarder
    show = True              # True pour afficher la figure

    data = load_optimization_json(json_file)
    out = plot_convergence(
        data,
        use_normalized=not use_raw,
        save_path=save_path,
        show=show
    )
    final_compliance, comp_time = extract_final_results(json_file)
    print(f"Final compliance: {final_compliance}")
    print(f"Total computation time (s): {comp_time}")
    if out:
        print(f"Figure saved to: {out}")


if __name__ == "__main__":
    main()
