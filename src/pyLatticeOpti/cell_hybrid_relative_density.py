from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor

# === Paramètre global : Activer/Désactiver la coupe ===
apply_cut = True  # Mettre False pour voir toute la surface

# === Chemin du fichier CSV ===
csv_file = "saved_lattice_file/volumes_lattice.csv"

# TODO delete this file

def load_and_filter_data(csv_file, min_vol=0, max_vol=0.6):
    """Charge les données et filtre les volumes entre min_vol et max_vol."""
    project_root = Path(__file__).resolve().parent.parent
    csv_file = project_root / csv_file
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    radii, volumes = data[:, :3], data[:, 3]

    # Filtrage des données
    valid_indices = (volumes >= min_vol) & (volumes <= max_vol)
    filtered_radii = radii[valid_indices]
    filtered_volumes = volumes[valid_indices]

    print(f"Nombre de lignes avant filtrage : {data.shape[0]}")
    print(f"Nombre de lignes après filtrage : {filtered_radii.shape[0]}")

    return filtered_radii, filtered_volumes


def project_radii_to_2D(radii):
    """Projette les 3 rayons en 2D via PCA."""
    pca = PCA(n_components=2)
    radii_2D = pca.fit_transform(radii)
    return radii_2D


def interpolate_volume_grid(radii_2D, volumes):
    """Interpoler les volumes sur une grille régulière pour affichage en surface."""
    grid_x, grid_y = np.mgrid[
                     np.min(radii_2D[:, 0]):np.max(radii_2D[:, 0]):100j,
                     np.min(radii_2D[:, 1]):np.max(radii_2D[:, 1]):100j
                     ]

    grid_z = griddata(radii_2D, volumes, (grid_x, grid_y), method='cubic')

    if np.isnan(grid_z).all():
        raise ValueError("Toutes les valeurs interpolées sont NaN. Vérifie les données d'entrée.")

    return grid_x, grid_y, grid_z


def plot_3D_surface(grid_x, grid_y, grid_z, apply_cut=True):
    """Affiche la surface 3D avec ou sans coupe par un plan."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if apply_cut:
        z_cut = np.median(grid_z)  # Coupe à la médiane
        z_cut = min(max(z_cut, np.nanmin(grid_z)), np.nanmax(grid_z))

        mask = grid_z <= z_cut  # On garde uniquement les valeurs sous le plan
        if not np.any(mask):
            raise ValueError("Aucun point sous le plan de coupe. Essaye d'ajuster z_cut.")

        grid_z_cut = np.where(mask, grid_z, np.nan)
        ax.plot_surface(grid_x, grid_y, np.full_like(grid_x, z_cut), color='gray', alpha=0.3)

        # Extraction de la ligne de coupe
        cut_x, cut_y = grid_x[mask], grid_y[mask]
        cut_z = np.full_like(cut_x, z_cut)

        if cut_x.size > 0:
            ax.plot(cut_x, cut_y, cut_z, color='red', linewidth=2, label="Ligne de coupe")
    else:
        grid_z_cut = grid_z

    # Tracé de la surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z_cut, cmap='viridis', edgecolor='none')

    # Ajout de labels et titre
    ax.set_xlabel("Composante 1 (PCA)")
    ax.set_ylabel("Composante 2 (PCA)")
    ax.set_zlabel("Volume")
    ax.set_title(
        "Projection 4D en Surface 3D (Avec Coupe)" if apply_cut else "Projection 4D en Surface 3D (Sans Coupe)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if apply_cut:
        plt.legend()

    plt.show()

def plot_2D_dependence_3D(radii, volumes, fixed_index=2):
    """Affiche une surface 3D du volume en fonction de deux rayons, avec le troisième fixé à 0."""

    # Filtrage des données où le troisième rayon est nul
    mask = radii[:, fixed_index] == 0
    if not np.any(mask):
        raise ValueError(f"Aucune donnée avec Radius[{fixed_index}] = 0")

    filtered_radii = radii[mask, :]
    filtered_volumes = volumes[mask]

    # Définir les deux axes variables
    x_index, y_index = [i for i in range(3) if i != fixed_index]
    x_vals, y_vals = filtered_radii[:, x_index], filtered_radii[:, y_index]

    # Création d'une grille régulière pour l'interpolation
    grid_x, grid_y = np.mgrid[
                     np.min(x_vals):np.max(x_vals):100j,  # 100 points sur X
                     np.min(y_vals):np.max(y_vals):100j  # 100 points sur Y
                     ]

    # Interpolation des volumes sur la grille
    grid_z = griddata((x_vals, y_vals), filtered_volumes, (grid_x, grid_y), method='cubic')

    # Vérification des NaN dans la surface
    if np.isnan(grid_z).all():
        raise ValueError("Toutes les valeurs interpolées sont NaN. Vérifie les données d'entrée.")

    # Création du graphique 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Tracé de la surface 3D
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Ajout d'une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Ajout des labels et titre
    ax.set_xlabel(f"Rayon {x_index + 1}")
    ax.set_ylabel(f"Rayon {y_index + 1}")
    ax.set_zlabel("Volume")
    ax.set_title(
        f"Surface du Volume en fonction de Rayon {x_index + 1} et Rayon {y_index + 1} (Rayon {fixed_index + 1} = 0)")

    plt.show()

def plot_3D_scatter(radii, volumes):
    """Trace un nuage de points 3D où le volume est représenté par une couleur."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extraction des 3 dimensions
    x_vals, y_vals, z_vals = radii[:, 0], radii[:, 1], radii[:, 2]

    # Tracé du scatter 3D avec couleur selon le volume
    sc = ax.scatter(x_vals, y_vals, z_vals, c=volumes, cmap='viridis', edgecolor='k')

    # Ajout des labels et d'une barre de couleur
    ax.set_xlabel("Rayon 1")
    ax.set_ylabel("Rayon 2")
    ax.set_zlabel("Rayon 3")
    ax.set_title("Nuage de points 3D - Volume dépendant des 3 rayons")
    fig.colorbar(sc, ax=ax, label="Volume")

    plt.show()

import plotly.graph_objects as go

def plot_3D_iso_surface(radii, volumes):
    """Affiche une véritable iso-surface 3D avec Plotly."""

    # Création d'une grille régulière
    grid_x, grid_y, grid_z = np.mgrid[
                             np.min(radii[:, 0]):np.max(radii[:, 0]):30j,
                             np.min(radii[:, 1]):np.max(radii[:, 1]):30j,
                             np.min(radii[:, 2]):np.max(radii[:, 2]):30j
                             ]

    # Interpolation des volumes sur la grille 3D
    grid_vol = griddata(radii, volumes, (grid_x, grid_y, grid_z), method='linear')

    # Création du tracé Plotly
    fig = go.Figure(data=go.Isosurface(
        x=grid_x.flatten(), y=grid_y.flatten(), z=grid_z.flatten(),
        value=grid_vol.flatten(),
        opacity=0.3, surface_count=10, colorscale='viridis'
    ))

    fig.update_layout(
        scene=dict(xaxis_title='Rayon 1', yaxis_title='Rayon 2', zaxis_title='Rayon 3'),
        title="Iso-Surface du Volume en 3D"
    )

    fig.show()


def plot_3D_contour_slices(radii, volumes, num_slices=5):
    """Affiche des coupes de volume sous forme de contours 2D dans un plot 3D."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Définir les niveaux de coupe selon Radius3
    z_levels = np.linspace(np.min(radii[:, 2]), np.max(radii[:, 2]), num_slices)

    # Définition des axes X et Y
    x_vals, y_vals, z_vals = radii[:, 0], radii[:, 1], radii[:, 2]

    for z_cut in z_levels:
        mask = np.isclose(z_vals, z_cut, atol=0.01)  # Sélectionne les points autour de ce niveau
        if not np.any(mask):
            continue

        # Sélection des points proches du niveau actuel
        x_plane, y_plane, vol_plane = x_vals[mask], y_vals[mask], volumes[mask]

        # Création de la grille pour l'interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(np.min(x_plane), np.max(x_plane), 50),
            np.linspace(np.min(y_plane), np.max(y_plane), 50)
        )

        # Interpolation des volumes sur la grille 2D
        grid_vol = griddata((x_plane, y_plane), vol_plane, (grid_x, grid_y), method='cubic')

        # Ajout des contours remplis
        ax.contourf(grid_x, grid_y, np.full_like(grid_x, z_cut), grid_vol, levels=10, cmap='viridis', alpha=0.7)

    # Ajout des labels
    ax.set_xlabel("Rayon 1")
    ax.set_ylabel("Rayon 2")
    ax.set_zlabel("Rayon 3")
    ax.set_title("Coupes 2D des volumes en fonction des rayons")
    plt.show()



def plot_3D_slices(radii, volumes, fixed_index=2, num_slices=5):
    """Affiche plusieurs coupes 2D du volume en fixant une valeur de l'un des rayons."""

    # Définir des niveaux fixes pour le rayon sélectionné
    fixed_values = np.linspace(np.min(radii[:, fixed_index]), np.max(radii[:, fixed_index]), num_slices)

    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5), sharex=True, sharey=True)

    # Définition des axes restants
    x_index, y_index = [i for i in range(3) if i != fixed_index]

    for i, fixed_value in enumerate(fixed_values):
        mask = np.isclose(radii[:, fixed_index], fixed_value, atol=0.01)
        if not np.any(mask):
            continue

        x_vals, y_vals = radii[mask, x_index], radii[mask, y_index]
        vol_vals = volumes[mask]

        # Tracé du scatter
        sc = axes[i].scatter(x_vals, y_vals, c=vol_vals, cmap='viridis', edgecolor='k')
        axes[i].set_title(f"{fixed_value:.2f} pour Rayon {fixed_index + 1}")
        axes[i].set_xlabel(f"Rayon {x_index + 1}")
        axes[i].set_ylabel(f"Rayon {y_index + 1}")

    # Barre de couleur
    fig.colorbar(sc, ax=axes, orientation='vertical', label="Volume")
    plt.suptitle(f"Tranches 2D en fonction de Rayon {fixed_index + 1}")
    plt.show()


from scipy.spatial import KDTree


def detect_large_volume_variations(radii, volumes, distance_threshold=0.02, variation_threshold=0.1):
    """Détecte les grandes variations du volume entre rayons proches."""

    # Création d'un arbre KDTree pour trouver les voisins proches
    tree = KDTree(radii)

    high_variation_pairs = []  # Stockage des paires de points avec une variation forte
    variations = []

    for i, radius_set in enumerate(radii):
        # Trouver les indices des points proches
        indices = tree.query_ball_point(radius_set, distance_threshold)

        for j in indices:
            if i != j:  # On évite de comparer un point avec lui-même
                volume_diff = abs(volumes[i] - volumes[j])

                if volume_diff > variation_threshold:
                    high_variation_pairs.append((radii[i], volumes[i], radii[j], volumes[j], volume_diff))
                    variations.append(volume_diff)

    return high_variation_pairs, variations


def remove_large_volume_variations(radii, volumes, distance_threshold=0.02, variation_threshold=0.1):
    """
    Supprime les points où le volume varie fortement par rapport aux voisins proches.

    - distance_threshold : Distance max pour considérer deux points comme voisins.
    - variation_threshold : Différence de volume considérée comme une forte variation.
    """

    tree = KDTree(radii)  # Construction du KDTree pour recherche rapide des voisins
    to_remove = set()  # Indices des points à supprimer

    for i, radius_set in enumerate(radii):
        # Trouver les indices des points proches
        indices = tree.query_ball_point(radius_set, distance_threshold)

        for j in indices:
            if i != j:  # Éviter la comparaison avec soi-même
                volume_diff = abs(volumes[i] - volumes[j])

                # Si la variation est trop grande, marquer les indices à supprimer
                if volume_diff > variation_threshold:
                    to_remove.add(i)
                    to_remove.add(j)

    # Création du nouveau dataset sans les valeurs aberrantes
    mask = np.array([i not in to_remove for i in range(len(radii))])
    filtered_radii = radii[mask]
    filtered_volumes = volumes[mask]

    print(f"Nombre de points supprimés : {len(to_remove)}")
    print(f"Nombre de points restants : {filtered_radii.shape[0]}")

    return filtered_radii, filtered_volumes

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Pour sauvegarder et charger le modèle

def evaluate_kriging_model(radii, volumes, test_size=0.2, save_path="saved_lattice_file/RelativeDensityKrigingModel.pkl"):
    """
    Entraîne un modèle de Kriging et évalue sa précision.

    Paramètres :
    ------------
    radii : np.ndarray
        Matrice des rayons de taille (n_samples, 3).
    volumes : np.ndarray
        Vecteur des volumes de taille (n_samples,).
    test_size : float
        Fraction des données utilisées pour le test (ex: 0.2 = 20%).

    Retourne :
    ----------
    metrics : dict
        Dictionnaire contenant les valeurs des métriques (MSE, RMSE, R²).
    """

    # Séparer les données en train/test
    X_train, X_test, y_train, y_test = train_test_split(radii, volumes, test_size=test_size, random_state=42)

    # Définition du noyau du GPR (Kriging)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    # Entraînement du modèle
    gpr.fit(X_train, y_train)

    # Prédictions sur le test set
    y_pred, std_dev = gpr.predict(X_test, return_std=True)

    # Calcul des métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_test) - np.min(y_test))  # Normalisation
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats
    print(f"✅ Évaluation du modèle Kriging :")
    print(f"   • MSE    = {mse:.6f}")
    print(f"   • RMSE   = {rmse:.6f}")
    print(f"   • NRMSE  = {nrmse:.6f}")
    print(f"   • R²     = {r2:.6f}")

    # Sauvegarde du modèle entraîné
    project_root = Path(__file__).resolve().parent.parent
    save_path = project_root / save_path
    joblib.dump(gpr, save_path)
    print(f"Modèle sauvegardé dans : {save_path}")

    return {"MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R2": r2}



def train_kriging_model(radii, volumes, save_path="saved_lattice_file/RelativeDensityKrigingModel.pkl"):
    """
    Entraîne un modèle de Kriging (Gaussian Process Regression) sur les données de volumes.

    Paramètres :
    ------------
    radii : np.ndarray
        Matrice des rayons de taille (n_samples, 3).
    volumes : np.ndarray
        Vecteur des volumes de taille (n_samples,).
    save_path : str
        Chemin pour sauvegarder le modèle entraîné.

    Retourne :
    ----------
    gpr : GaussianProcessRegressor
        Modèle entraîné.
    """

    # Définition du noyau : produit d'un coefficient de variance (C) et d'un noyau RBF
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    # Création du modèle de Kriging (Gaussian Process)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    # Entraînement du modèle sur les données
    gpr.fit(radii, volumes)

    # Affichage des paramètres appris
    print(f"Modèle entraîné avec noyau optimisé : {gpr.kernel_}")

    # Sauvegarde du modèle entraîné
    joblib.dump(gpr, save_path)
    print(f"Modèle sauvegardé dans : {save_path}")

    return gpr

# --- add this helper (module-level or as a @staticmethod) ---
def _gp_mean_gradient_rbf_pipeline(pipe, x_row: np.ndarray) -> np.ndarray:
    """
    Exact gradient of the GPR predictive mean wrt inputs for a Pipeline(StandardScaler -> GPR)
    with kernel ConstantKernel * RBF (ARD or isotropic). Returns dmu/dx in ORIGINAL (unscaled) space.
    """
    import numpy as np
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.gaussian_process.kernels import Product

    scaler = pipe.named_steps["x_scaler"]
    gpr = pipe.named_steps["gpr"]

    # x in original space -> scaled space
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)
    x_s = scaler.transform(x_row).reshape(-1)  # (d,)

    Xs = gpr.X_train_                     # (n_train, d) in scaled space
    alpha = gpr.alpha_.reshape(-1)        # (n_train,)

    # Extract Constant * RBF
    k = gpr.kernel_
    if isinstance(k, Product):
        k1, k2 = k.k1, k.k2
        if isinstance(k1, ConstantKernel) and isinstance(k2, RBF):
            const_val = float(k1.constant_value)
            rbf = k2
        elif isinstance(k2, ConstantKernel) and isinstance(k1, RBF):
            const_val = float(k2.constant_value)
            rbf = k1
        else:
            raise ValueError("Kernel must be ConstantKernel * RBF for exact gradient.")
    elif isinstance(k, RBF):
        const_val = 1.0
        rbf = k
    else:
        raise ValueError("Kernel must be (ConstantKernel * RBF) or RBF for exact gradient.")

    length_scale = np.asarray(rbf.length_scale, dtype=float)
    if length_scale.ndim == 0:
        length_scale = np.full(Xs.shape[1], float(length_scale))
    ell2 = length_scale**2  # (d,)

    # k(x, Xi) in scaled space
    diff = Xs - x_s  # (n_train, d)
    sq_maha = np.sum((diff**2) / ell2, axis=1)  # (n_train,)
    k_vec = const_val * np.exp(-0.5 * sq_maha)  # (n_train,)

    # ∂k/∂x_j = k * (Xi_j - x_j) / ell_j^2
    # ∂μ/∂x_j = sum_i alpha_i * ∂k_i/∂x_j
    num = (diff / ell2) * k_vec[:, None]        # (n_train, d)
    dmu_dx_scaled = num.T @ alpha               # (d,)

    # Chain rule back to original space: x_s = (x - mean)/scale
    scale = scaler.scale_.astype(float)         # (d,)
    dmu_dx = dmu_dx_scaled / scale              # (d,)
    return dmu_dx


radii, volumes = load_and_filter_data(csv_file)

# === Exécution de la suppression des points aberrants ===
radii, volumes = remove_large_volume_variations(radii, volumes, distance_threshold=0.02,
                                                                  variation_threshold=0.1)

metrics = evaluate_kriging_model(radii, volumes)


# radii_2D = project_radii_to_2D(radii)
# grid_x, grid_y, grid_z = interpolate_volume_grid(radii_2D, volumes)
#
# # Affichage de la surface 3D
# plot_3D_surface(grid_x, grid_y, grid_z, apply_cut=apply_cut)

# Affichage de la dépendance entre 2 rayons et le volume quand le 3e rayon est 0
# plot_2D_dependence_3D(radii, volumes, fixed_index=0)
# plot_3D_scatter(radii, volumes)
# plot_3D_contour_slices(radii, volumes, num_slices=5)
plot_3D_iso_surface(radii, volumes)
# plot_3D_slices(radii, volumes, fixed_index=2, num_slices=5)
