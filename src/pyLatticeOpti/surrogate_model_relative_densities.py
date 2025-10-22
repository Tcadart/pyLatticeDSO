# =============================================================================
# MODULE: surrogate_model_relative_densities
#
# DESCRIPTION:
# This module provides functions to compute, save, load, and evaluate
# relative density datasets for lattice structures, as well as to train and
# evaluate Kriging surrogate models using Gaussian Process Regression (GPR).
# =============================================================================

import os
from itertools import product
from pathlib import Path
import pickle

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

try:
    from scipy.spatial import KDTree
except Exception:
    KDTree = None  # Allows importing this module during doc builds without SciPy binary compatibility.


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.interpolate import griddata


# =============================================================================
# SECTION: Dataset computation, saving, and loading
# =============================================================================

def _find_path_to_data(lattice_cell):
    """Determine a default dataset path based on lattice geometry."""
    path_dataset = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "data"
    geom_cell = lattice_cell.cells[0].geom_types
    name_dataset = "RelativeDensities" + "".join(f"_{g}" for g in geom_cell)
    return path_dataset, name_dataset

def compute_relative_densities_dataset(lattice_cell,
                                       step_radius: float = 0.01,
                                       range_radius: tuple = (0.00, 0.11),
                                       save_every: int = 1,
                                       resume: bool = True):
    """
    Compute relative densities for a range of radii in the lattice.
    Periodically saves partial results every `save_every` iterations and can resume if a file exists.
    """
    if lattice_cell.get_number_cells() != 1:
        raise ValueError("The lattice must contain exactly one cell.")

    path_dataset, name_dataset = _find_path_to_data(lattice_cell)

    # Prepare grid and potential resume
    n_geom = len(lattice_cell.cells[0].geom_types)
    valid_combos = _valid_combinations(step_radius, range_radius, n_geom)

    relative_densities = {}
    dataset_file = path_dataset / f"{name_dataset}.pkl"
    if dataset_file.exists() and resume:
        relative_densities = load_dataset(path_dataset, name_dataset)
        print(f"Resuming from {len(relative_densities)} existing entries in {dataset_file}")

        # Use check_missing_entries to report and sanitize
        status = check_missing_entries(
            path_dataset=path_dataset,
            name_dataset=name_dataset,
            step_radius=step_radius,
            range_radius=range_radius,
            n_geom=n_geom,
            threshold=0.001,
        )

        # Remove invalid entries that don't match current grid/threshold
        if status.get("n_invalid", 0) > 0:
            print(f"Removing {status['n_invalid']} invalid entries from {dataset_file}")
            for combo in status["invalid_combinations"]:
                relative_densities.pop(combo, None)
            save_dataset(path_dataset, name_dataset, relative_densities)

        remaining = status["missing_combinations"]
        print(f"{len(remaining)} combinations missing (expected {status['n_expected']}).")
    else:
        remaining = [c for c in valid_combos if c not in relative_densities]

    if not remaining:
        print("Dataset already complete.")
        return

    for i, combo in enumerate(remaining, 1):
        print(f"Computing for radii: {combo}")
        # if i <= 0:
        #     combo = list(combo)
        #     combo[0] += 0.001
        #     lattice_cell.change_beam_radius(combo)
        #     combo[0] -= 0.001
        #     combo = tuple(combo)
        # else:
        lattice_cell.change_beam_radius(list(combo))
        rd = lattice_cell.generate_mesh_lattice_Gmsh(volume_computation=True,
                                                     cut_mesh_at_boundary=True,
                                                     save_STL=False,
                                                     only_relative_density=True)
        if rd is None:
            continue
        relative_densities[combo] = rd
        print(f"Relative density: {rd}")

        if i % max(1, int(save_every)) == 0:
            save_dataset(path_dataset, name_dataset, relative_densities)
            print(f"Progress saved: {len(relative_densities)}/{len(valid_combos)} entries")

    # Final save
    save_dataset(path_dataset, name_dataset, relative_densities)
    print(f"✅ Completed. Total entries: {len(relative_densities)}/{len(valid_combos)}")


def save_dataset(path_dataset, name_dataset, relative_densities_dict):
    """Save the dataset as a pickle file (atomic write)."""
    path_dataset.mkdir(parents=True, exist_ok=True)
    final_path = path_dataset / f"{name_dataset}.pkl"
    tmp_path = path_dataset / f"{name_dataset}.pkl.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(relative_densities_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, final_path)  # atomic on POSIX/Windows
    print(f"Dataset saved as pickle at {final_path}")


def load_dataset(path_dataset, name_dataset,
                 min_vol: float = 0.0, max_vol: float = 0.6,
                 apply_variation_filter: bool = True):
    """
    Load a dataset previously saved as a pickle file and optionally filter it.

    Parameters
    ----------
    path_dataset : Path
        Path to the directory containing the dataset.

    name_dataset : str
        Name of the dataset (without extension).

    min_vol : float, optional
        Minimum volume to keep (None = no lower filter).

    max_vol : float, optional
        Maximum volume to keep (None = no upper filter).

    apply_variation_filter : bool, optional
        If True, apply the remove_large_volume_variations_dict filter.

    Returns
    -------
    dict
        Dictionary of relative densities keyed by radii tuple.
    """
    file_path = path_dataset / f"{name_dataset}.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "rb") as f:
        relative_densities_dict = pickle.load(f)

    print(f"Dataset {name_dataset} loaded from {file_path}")

    # --- Filtrage min/max volume ---
    if min_vol is not None or max_vol is not None:
        filtered_dict = {}
        for radii, vol in relative_densities_dict.items():
            if (min_vol is None or vol >= min_vol) and (max_vol is None or vol <= max_vol):
                filtered_dict[radii] = vol
        print(f"Filtrage volumes : {len(relative_densities_dict)} -> {len(filtered_dict)} entrées")
        relative_densities_dict = filtered_dict

    # --- Filtrage des fortes variations ---
    if apply_variation_filter:
        relative_densities_dict = remove_large_volume_variations_dict(relative_densities_dict)

    return relative_densities_dict


def csv_to_dataset(csv_file: Path):
    """
    Convert a CSV file with columns Radius1, Radius2, Radius3, Volume
    into the dict format { (r1, r2, r3): volume } and save it with save_dataset.
    """
    try:
        import pandas as pd
    except ImportError:
        import subprocess
        try:
            subprocess.check_call(["conda", "install", "-y", "pandas"])
            import pandas as pd
        except Exception:
            raise ImportError("Please install scikit-image to use this function.")
    df = pd.read_csv(csv_file)

    required_cols = {"Radius1", "Radius2", "Radius3", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    relative_densities = {}
    for _, row in df.iterrows():
        key = (round(float(row["Radius1"]),3), round(float(row["Radius2"]),3), round(float(row["Radius3"]),3))
        relative_densities[key] = float(row["Volume"])

    path_dataset = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "data"
    name_dataset = "Test"

    save_dataset(path_dataset, name_dataset, relative_densities)
    return relative_densities

def _valid_combinations(step_radius: float, range_radius: tuple, n_geom: int, threshold: float = 0.001):
    """Generate all valid radius combinations given the grid and a sum threshold."""
    min_radius, max_radius = range_radius
    radius_values = np.arange(min_radius, max_radius + step_radius, step_radius)
    radius_values = np.append(radius_values, 0.001)  # Ensure 0.001 is included
    all_combinations = list(product(radius_values, repeat=n_geom))
    return [c for c in all_combinations if sum(c) > threshold]

def check_missing_entries(path_dataset: Path,
                          name_dataset: str,
                          step_radius: float,
                          range_radius: tuple,
                          n_geom: int,
                          threshold: float = 0.001):
    """
    Check which radius combinations are missing in an existing dataset file (if any).
    Returns a dict with counts and the list of missing combinations.
    """
    expected = set(_valid_combinations(step_radius, range_radius, n_geom, threshold))
    print(len(expected), "expected combinations based on parameters.")
    try:
        data = load_dataset(path_dataset, name_dataset)
        present = set(data.keys())
    except FileNotFoundError:
        present = set()

    missing = sorted(expected - present)
    invalid = sorted(present - expected)

    if invalid:
        print(f"⚠️ Found {len(invalid)} invalid combinations in dataset:")
        for combo in invalid[:20]:
            print("   ", combo)
        if len(invalid) > 20:
            print(f"   ... and {len(invalid)-20} more")

    return {
        "n_expected": len(expected),
        "n_present": len(present),
        "n_missing": len(missing),
        "missing_combinations": missing,
        "n_invalid": len(invalid),
        "invalid_combinations": invalid,
    }

# =============================================================================
# SECTION: Visualization functions
# =============================================================================

def _load_dataset_for_plotting(lattice_cell=None, name_dataset=None):
    """Helper to load dataset for plotting functions."""
    if lattice_cell is not None:
        path_dataset, name_dataset = _find_path_to_data(lattice_cell)
        dataset_file = path_dataset / f"{name_dataset}.pkl"
    else:
        path_dataset = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "data"
        dataset_file = path_dataset / f"{name_dataset}.pkl"
    if not dataset_file.exists():
        raise FileNotFoundError(f"No dataset found at {dataset_file}")

    relative_densities = load_dataset(path_dataset, name_dataset)
    if not relative_densities:
        raise ValueError("Loaded dataset is empty.")

    radii = np.array(list(relative_densities.keys()), dtype=float)    # shape (N, n_geom)
    volumes = np.array(list(relative_densities.values()), dtype=float)  # shape (N,)

    return radii, volumes

def plot_3D_iso_surface(lattice_cell=None, name_dataset=None, n_levels = 10):
    """
    Plot 3D iso-surfaces of volume as a function of three radii using interpolation.

    Parameters
    ----------
    lattice_cell : Lattice
        Lattice object with exactly one cell and three geometry types.

    name_dataset : str
        Name of the dataset to load (without extension).

    n_levels : int, optional
        Number of iso-surface levels to plot (default is 10).
    """
    if lattice_cell is None and name_dataset is None:
        raise ValueError("Either lattice_cell or name_dataset must be provided.")
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        import subprocess
        try:
            # lance conda install dans l'environnement courant
            subprocess.check_call(["conda", "install", "-y", "scikit-image"])
            from skimage.measure import marching_cubes
        except Exception:
            raise ImportError("Please install scikit-image to use this function.")

    # --- Load dataset and build (radii, volumes) arrays ---
    radii, volumes = _load_dataset_for_plotting(lattice_cell, name_dataset)

    if radii.ndim != 2 or radii.shape[1] != 3:
        raise ValueError(f"Expected 3 radii per sample, got shape {radii.shape}")

    # --- Build regular 3D grid over observed radii domain ---
    grid_x, grid_y, grid_z = np.mgrid[
                             np.min(radii[:, 0]):np.max(radii[:, 0]):30j,
                             np.min(radii[:, 1]):np.max(radii[:, 1]):30j,
                             np.min(radii[:, 2]):np.max(radii[:, 2]):30j
                             ]

    # --- Interpolate volumes on the grid ---
    grid_vol = griddata(radii, volumes, (grid_x, grid_y, grid_z), method='linear')

    vmin, vmax = np.nanpercentile(grid_vol, [5, 95])
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Interpolated field is degenerate (constant or invalid).")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import cm, colors

    vmin, vmax = np.nanpercentile(grid_vol, [1, 99])
    levels = np.linspace(vmin, vmax, n_levels)

    dx = float(grid_x[1, 0, 0] - grid_x[0, 0, 0])
    dy = float(grid_y[0, 1, 0] - grid_y[0, 0, 0])
    dz = float(grid_z[0, 0, 1] - grid_z[0, 0, 0])
    origin = np.array([grid_x.min(), grid_y.min(), grid_z.min()], dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    for iso in levels:
        verts, faces, _, _ = marching_cubes(grid_vol, level=iso, spacing=(dx, dy, dz))
        verts += origin

        poly = Poly3DCollection(verts[faces], alpha=0.35, facecolor=cmap(norm(iso)), edgecolor="none")
        ax.add_collection3d(poly)

    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_zlim(grid_z.min(), grid_z.max())
    ax.set_box_aspect((grid_x.ptp(), grid_y.ptp(), grid_z.ptp()))

    ax.set_xlabel("Base geometry 1", fontsize=16)
    ax.set_ylabel("Base geometry 2", fontsize=16)
    ax.set_zlabel("Base geometry 3", fontsize=16)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65)
    cbar.set_label(r"$\rho_{rel}$", fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()

def plot_3D_scatter(lattice_cell=None, name_dataset=None):
    """
    Plot 3D scatter of relative densities as a function of three radii.

    Parameters
    ----------
    lattice_cell : Lattice
        Lattice object with exactly one cell and three geometry types.
    """
    if lattice_cell is None and name_dataset is None:
        raise ValueError("Either lattice_cell or name_dataset must be provided.")
    # --- Load dataset ---
    radii, volumes = _load_dataset_for_plotting(lattice_cell, name_dataset)

    if radii.ndim != 2 or radii.shape[1] != 3:
        raise ValueError(f"Expected 3 radii per sample, got shape {radii.shape}")

    # --- Scatter plot 3D ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(radii[:, 0], radii[:, 1], radii[:, 2],
                    c=volumes, cmap="viridis", s=20, alpha=0.8)

    ax.set_xlabel("Radius 1")
    ax.set_ylabel("Radius 2")
    ax.set_zlabel("Radius 3")
    ax.set_title("3D Scatter of Relative Densities")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Relative density")
    plt.show()

def remove_large_volume_variations_dict(relative_densities: dict,
                                        distance_threshold=0.02,
                                        variation_threshold=0.1):
    """
    Delete entries in the relative_densities dict where neighboring points (in radius space)
    have volume differences exceeding variation_threshold.

    Parameters
    ----------
    relative_densities : dict
        Dictionary with keys as radius tuples and values as relative densities.

    distance_threshold : float
        Maximum distance in radius space to consider points as neighbors.

    variation_threshold : float
        Maximum allowed volume difference between neighboring points.

    Returns
    -------
    dict
        Filtered dictionary with large variations removed.
    """

    # Conversion en arrays pour KDTree
    radii = np.array(list(relative_densities.keys()), dtype=float)
    volumes = np.array(list(relative_densities.values()), dtype=float)

    tree = KDTree(radii)
    to_remove = set()

    for i, radius_set in enumerate(radii):
        indices = tree.query_ball_point(radius_set, distance_threshold)

        for j in indices:
            if i != j:
                volume_diff = abs(volumes[i] - volumes[j])
                if volume_diff > variation_threshold:
                    to_remove.add(i)
                    to_remove.add(j)

    mask = np.array([i not in to_remove for i in range(len(radii))])
    filtered_radii = radii[mask]
    filtered_volumes = volumes[mask]

    # Reconstruction du dictionnaire filtré
    filtered_dict = {tuple(r): v for r, v in zip(filtered_radii, filtered_volumes)}

    print(f"Nombre de points supprimés : {len(to_remove)}")
    print(f"Nombre de points restants : {len(filtered_dict)}")

    return filtered_dict

# =============================================================================
# SECTION: Kriging model training and evaluation
# =============================================================================

def evaluate_kriging_from_pickle(
    dataset_dir: Path,
    name_dataset: str,
    test_size: float = 0.2,
    model_name: str = "kriging_model_",
    normalize_y: bool = True,
    random_state: int = 42,
    min_vol: float = 0.0,
    max_vol: float = 0.6,
    apply_variation_filter: bool = True,
):
    """
    Train and evaluate a Kriging (GPR) model from a relative-density dataset saved as a pickle.

    Parameters
    ----------
    dataset_dir : Path
        Directory that contains '<name_dataset>.pkl' produced by `compute_relative_densities_dataset`.

    name_dataset : str
        Dataset base name (without extension).

    test_size : float, optional
        Fraction of samples used for testing (default 0.2).

    model_name : str | Path, optional
        Relative path (from the project root) where the trained model will be saved.

    normalize_y : bool, optional
        Whether to normalize the target inside the regressor (default True).

    random_state : int, optional
        Random seed for reproducibility.

    min_vol, max_vol : float, optional
        Filtering bounds for volumes (default 0.0–0.6).

    apply_variation_filter : bool, optional
        If True, apply the remove_large_volume_variations_dict filter.

    Returns
    -------
    dict
        Evaluation metrics and paths. Keys: 'MSE', 'RMSE', 'NRMSE', 'MAE', 'R2',
        'n_train', 'n_test', 'model_path', 'kernel_'
    """
    if dataset_dir is None:
        dataset_dir = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "data"

    # --- Load dataset with filters ---
    rel_dens_dict = load_dataset(
        path_dataset=dataset_dir,
        name_dataset=name_dataset,
        min_vol=min_vol,
        max_vol=max_vol,
        apply_variation_filter=apply_variation_filter,
    )

    if not isinstance(rel_dens_dict, dict) or len(rel_dens_dict) == 0:
        raise ValueError("Loaded dataset is empty or not a dictionary.")

    # Convert dict to arrays
    X = np.array(list(rel_dens_dict.keys()), dtype=float)
    y = np.array(list(rel_dens_dict.values()), dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Sanity checks
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("Non-finite values detected in dataset (NaN/Inf).")

    if np.ptp(y) == 0:
        raise ValueError("All relative densities are identical; cannot evaluate NRMSE/R² meaningfully.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=random_state
    )

    # Kernel definition
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=X.shape[1] * [1.0],
                                       length_scale_bounds=(1e-3, 1e3))

    pipe = Pipeline([
        ("x_scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,  # nugget numérique (évite le bruit au bord)
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=random_state,
        ))
    ])

    pipe.fit(X_train, y_train)

    X_test_s = pipe.named_steps["x_scaler"].transform(X_test)
    y_pred, y_std = pipe.named_steps["gpr"].predict(X_test_s, return_std=True)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    nrmse = float(rmse / (np.max(y_test) - np.min(y_test)))
    r2 = r2_score(y_test, y_pred)
    learned_kernel = str(pipe.named_steps["gpr"].kernel_)

    print("✅ Kriging model evaluation")
    print(f"   • Train size = {len(y_train)}, Test size = {len(y_test)}")
    print(f"   • MSE   = {mse:.6e}")
    print(f"   • RMSE  = {rmse:.6e}")
    print(f"   • MAE   = {mae:.6e}")
    print(f"   • NRMSE = {nrmse:.6e}")
    print(f"   • R²    = {r2:.6f}")
    print(f"   • Learned kernel: {learned_kernel}")

    # Save model
    project_root = Path(__file__).resolve().parents[2] / "data" / "outputs" / "relative_densities" / "surrogate_model"
    model_path = (project_root / (model_name + name_dataset.replace("RelativeDensities_", ""))).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipe,
            "kernel": learned_kernel,
            "feature_dim": X.shape[1],
            "metadata": {
                "dataset_path": str(dataset_dir / f"{name_dataset}.pkl"),
                "name_dataset": name_dataset,
                "normalize_y": normalize_y,
                "random_state": random_state,
                "min_vol": min_vol,
                "max_vol": max_vol,
                "variation_filter": apply_variation_filter,
            },
        },
        model_path,
    )
    print(f"💾 Model saved to: {model_path}")

    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": float(mae),
        "R2": float(r2),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model_path": str(model_path),
        "kernel_": str(learned_kernel),
    }

def plot_parity(y_true, y_pred, save_path=None):
    """
    Plot parity plot (true vs predicted values) with R² and RMSE annotation.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values (test set).

    y_pred : array-like
        Predicted values (test set).

    save_path : str or Path, optional
        If provided, saves the figure to this path.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Metrics
    residuals = y_pred - y_true
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter points
    ax.scatter(y_true, y_pred, c="blue", alpha=0.6, edgecolor="k", s=30, label="Test samples")

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")

    ax.set_xlabel(r"Computed $\rho_{rel}$", fontsize=16)
    ax.set_ylabel(r"Predicted $\rho_{rel}$", fontsize=16)
    # ax.set_title("Parity plot for Kriging surrogate", fontsize=14)

    # Annotate metrics
    ax.text(0.05, 0.95,
            f"$R^2$ = {r2:.5f}\nRMSE = {rmse:.2e}\nMAE = {mae:.2e}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=18,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(fontsize=18)
    ax.set_aspect("equal", "box")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Parity plot saved to {save_path}")
    plt.show()

    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def evaluate_saved_kriging(
    dataset_dir: Path | None,
    name_dataset: str,
    model_name: str | None = None,
    use_test_split: bool = True,
    test_size: float = 0.9,
    random_state: int = 42,
    min_vol: float = 0.0,
    max_vol: float = 0.6,
    apply_variation_filter: bool = True,
    save_parity_path: Path | None = None,
):
    """
    Load a previously trained Kriging model and a dataset, then evaluate and (optionally) plot a parity plot.

    Parameters
    ----------
    dataset_dir : Path | None
        Directory containing '<name_dataset>.pkl'. If None, uses the default data path.

    name_dataset : str
        Dataset base name (without extension).

    model_name : str | None
        Relative path (from the project root) to the saved model.

    use_test_split : bool
        If True, evaluate on a train/test split (reproducible with `random_state`).
        If False, evaluate on the whole dataset.

    test_size : float
        Fraction of samples for testing when `use_test_split=True`.

    random_state : int
        Random seed for the split.

    min_vol, max_vol : float
        Volume filtering bounds passed to `load_dataset`.

    apply_variation_filter : bool
        Apply `remove_large_volume_variations_dict` during dataset loading.

    save_parity_path : Path | None
        If provided, saves the parity plot to this path.

    Returns
    -------
    dict
        Metrics and metadata: 'MSE','RMSE','NRMSE','MAE','R2','n_eval','model_kernel','model_path'
    """
    # Resolve default paths
    if dataset_dir is None:
        dataset_dir = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "data"
    model_root = Path(__file__).parents[2] / "data" / "outputs" / "relative_densities" / "surrogate_model"
    model_path = model_root / f"kriging_model_{model_name}"

    save_parity_path = dataset_dir / save_parity_path

    # Load model
    model_obj = joblib.load(model_path)
    gpr = model_obj["model"]
    model_kernel = str(model_obj.get("kernel", getattr(gpr, "kernel_", "unknown")))

    # Load dataset with filters
    rel_dens_dict = load_dataset(
        path_dataset=dataset_dir,
        name_dataset=name_dataset,
        min_vol=min_vol,
        max_vol=max_vol,
        apply_variation_filter=apply_variation_filter,
    )
    if not rel_dens_dict:
        raise ValueError("Loaded dataset is empty after filtering.")

    # Build arrays
    X = np.array(list(rel_dens_dict.keys()), dtype=float)
    y = np.array(list(rel_dens_dict.values()), dtype=float)
    is_1d_keys = (X.shape[1] == 1)
    if is_1d_keys:
        X = X.reshape(-1, 1)

    # Choose eval set
    if use_test_split:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=float(test_size), random_state=random_state
        )
    else:
        X_eval, y_eval = X, y


    # Predict & compute metrics
    y_pred = gpr.predict(X_eval)
    residuals = y_pred - y_eval
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    nrmse = float(rmse / (np.max(y_eval) - np.min(y_eval)))
    r2 = float(1.0 - np.sum(residuals**2) / np.sum((y_eval - np.mean(y_eval))**2))

    # Parity plot (optional)
    try:
        plot_parity(y_eval, y_pred, save_path=save_parity_path)
    except Exception as e:
        print(f"[warn] Parity plot failed: {e}")

    if is_1d_keys:
        fig, ax = plt.subplots()
        ax.scatter(X_eval.ravel(), y_eval, s=16, alpha=0.8, label="Ground truth")

        x_min, x_max = float(np.min(X)), float(np.max(X))
        x_grid = np.linspace(x_min, x_max, 400).reshape(-1, 1)
        y_grid = gpr.predict(x_grid)
        ax.plot(x_grid.ravel(), y_grid, linewidth=1.5, label="Model prediction")

        ax.set_xlabel("Key")
        ax.set_ylabel("Relative density")
        ax.set_title("Relative density vs Key")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

        if save_parity_path is not None:
            save_xy_path = save_parity_path.with_name(save_parity_path.stem + "_xy.png")
            fig.savefig(save_xy_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("✅ Loaded Kriging model evaluation")
    print(f"   • Eval size = {len(y_eval)}")
    print(f"   • MSE   = {mse:.6e}")
    print(f"   • RMSE  = {rmse:.6e}")
    print(f"   • MAE   = {mae:.6e}")
    print(f"   • NRMSE = {nrmse:.6e}")
    print(f"   • R²    = {r2:.6f}")
    print(f"   • Model kernel: {model_kernel}")
    print(f"   • Model path: {model_path}")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "R2": r2,
        "n_eval": int(len(y_eval)),
        "model_kernel": model_kernel,
        "model_path": str(model_path),
    }

def gp_mean_gradient_rbf_pipeline(model, x_row: np.ndarray) -> np.ndarray:
    """
    Exact gradient of the GPR predictive mean wrt inputs for either:
      • Pipeline(StandardScaler -> GaussianProcessRegressor), or
      • bare GaussianProcessRegressor trained in original space.

    Supports kernels:
      - RBF
      - ConstantKernel * RBF
      - (ConstantKernel * RBF) + WhiteKernel
      - RBF + WhiteKernel

    Returns dmu/dx in ORIGINAL (unscaled) space.
    """
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Product, WhiteKernel, Sum

    # --- Extract scaler (optional) and GPR ---
    scaler = None
    gpr = None
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("x_scaler", None) if hasattr(model, "named_steps") else None
        gpr = model.named_steps.get("gpr", None) if hasattr(model, "named_steps") else None
        if gpr is None:
            for _, step in reversed(model.steps):
                if isinstance(step, GaussianProcessRegressor):
                    gpr = step
                    break
        if gpr is None:
            raise ValueError("Pipeline does not contain a GaussianProcessRegressor step.")
    elif isinstance(model, GaussianProcessRegressor):
        gpr = model
    else:
        raise ValueError("Model must be a sklearn Pipeline or a GaussianProcessRegressor.")

    # --- Point in original space -> (optionally) scaled space ---
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    x_s = scaler.transform(x_row).reshape(-1) if scaler is not None else x_row.reshape(-1)

    Xs = gpr.X_train_               # (n_train, d) in the space GPR was fit on
    alpha = gpr.alpha_.reshape(-1)  # (n_train,)

    # --- Extract effective (const, RBF) while ignoring WhiteKernel in cross-cov ---
    def extract_const_rbf(kern):
        # Returns (const_val, rbf_kernel) or None if unsupported
        if isinstance(kern, Sum):
            left = extract_const_rbf(kern.k1)
            right = extract_const_rbf(kern.k2)
            if left is None and right is None:
                return None
            return left if right is None else right
        if isinstance(kern, WhiteKernel):
            return None  # no cross-covariance contribution
        if isinstance(kern, Product):
            a, b = kern.k1, kern.k2
            if isinstance(a, ConstantKernel) and isinstance(b, RBF):
                return float(a.constant_value), b
            if isinstance(b, ConstantKernel) and isinstance(a, RBF):
                return float(b.constant_value), a
            if isinstance(a, ConstantKernel) and isinstance(b, Product):
                sub = extract_const_rbf(b)
                if sub is not None:
                    c, r = sub
                    return float(a.constant_value) * c, r
            if isinstance(b, ConstantKernel) and isinstance(a, Product):
                sub = extract_const_rbf(a)
                if sub is not None:
                    c, r = sub
                    return float(b.constant_value) * c, r
        if isinstance(kern, RBF):
            return 1.0, kern
        return None

    res = extract_const_rbf(gpr.kernel_)
    if res is None:
        raise ValueError("Kernel must reduce to RBF (optionally scaled by ConstantKernel) "
                         "with an optional + WhiteKernel.")
    const_val, rbf = res

    # --- ARD or isotropic length-scales ---
    length_scale = np.asarray(rbf.length_scale, dtype=float)
    if length_scale.ndim == 0:
        length_scale = np.full(Xs.shape[1], float(length_scale))
    ell2 = length_scale**2  # (d,)

    # --- Compute k(x, Xi) and gradient in the space GPR was fit on ---
    diff = Xs - x_s                                # (n_train, d)
    sq_maha = np.sum((diff**2) / ell2, axis=1)     # (n_train,)
    k_vec = const_val * np.exp(-0.5 * sq_maha)     # (n_train,)

    # ∂k_i/∂x_j = k_i * (Xi_j - x_j) / ell_j^2
    dmu_dx_scaled = ((diff / ell2) * k_vec[:, None]).T @ alpha  # (d,)

    # --- Chain rule back to original input space if a StandardScaler was used ---
    if scaler is not None:
        dmu_dx = dmu_dx_scaled / np.asarray(scaler.scale_, dtype=float)
    else:
        dmu_dx = dmu_dx_scaled

    # --- IMPORTANT: undo target normalization if gpr.normalize_y was used ---
    # sklearn stores _y_train_std when normalize_y=True; predictions are de-normalized,
    # so the derivative must be scaled by this std factor.
    y_std = getattr(gpr, "_y_train_std", None)
    if y_std is not None and np.isfinite(y_std).all():
        # y_std is scalar for single-output GPR
        dmu_dx = dmu_dx * float(np.squeeze(y_std))

    return dmu_dx
