"""
Summary table (Markdown) of optimization_study_three_point_bending.py: DDM + RBF surrogate against direct FEM with
analytic gradient, for every case saved in the study folder. The table is written in comparison_DDM_vs_FEM.md of the
study folder, followed by the observations kept from a previous version of the file (section "## Observations").
Study folder: data/outputs/optimization_study/Three_point_bending by default, or `--input=optimization/<name>`.
"""
import json
import sys
from pathlib import Path

from pyLatticeDesign.utils import open_lattice_parameters

name_file = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--input=")), "optimization/Three_point_bending")
save_dir = Path(__file__).resolve().parents[2] / "data" / "outputs" / "optimization_study" / Path(name_file).name
geometry = open_lattice_parameters(name_file)["geometry"]
cells = geometry["number_of_cells"]

rows = []
for name in ("hybrid", "hybrid_FEM", "graded", "graded_FEM", "unit_cell", "unit_cell_FEM"):
    path = save_dir / f"{name}.json"
    if not path.exists():
        continue
    case = json.loads(path.read_text())
    label = case.get("model", "DDM_RBF")
    solution, timing, compliance = case["solution"], case["time"], case["compliance"]
    density = case["relative_density"]
    method = "FEM direct + gradient analytique" if label.startswith("FEM") else "DDM + RBF"
    verified = f"**{100 * compliance['reduction_FEM']:.2f}**" if "reduction_FEM" in compliance else "-"
    direct = f"{density['final_direct_Gmsh']:.4f}" if "final_direct_Gmsh" in density else "-"
    rows.append(f"| {name} | {method} | {case['n_design_variables']} | {solution['nit']} | {solution['nfev']} | "
                f"{timing['optimization']:.1f} | {timing['optimization'] / solution['nfev']:.1f} | "
                f"{compliance[f'initial_{label}']:.5f} | {compliance[f'final_{label}']:.5f} | "
                f"{100 * compliance[f'reduction_{label}']:.2f} | {compliance.get('initial_FEM', float('nan')):.5f} | "
                f"{compliance.get('final_FEM', float('nan')):.5f} | {verified} | "
                f"{density['final_surrogate']:.4f} | {direct} |")

first_case = json.loads(next(save_dir.glob("*.json")).read_text()) if rows else {}
target = first_case.get("relative_density", {}).get("target")
header = f"""# Flexion 3 points ({" + ".join(geometry["geom_types"])}, {cells["x"]}x{cells["y"]}x{cells["z"]} cellules) : \
DDM + RBF vs FEM direct

Généré par `examples/optimization/optimization_study_summary_table.py --input={name_file}` à partir des fichiers
`<cas>.json` de ce dossier (`optimization_study_three_point_bending.py`).

- **DDM + RBF** : décomposition de domaine, compléments de Schur RBF, pénalisation aux nœuds, gradient adjoint.
- **FEM direct** (`<cas>_FEM`) : solve LU de la structure complète, gradient analytique de la compliance
  (dérivée UFL de l'énergie par rapport au champ de rayon), **sans pénalisation aux nœuds**. Le cas cellule par cellule
  utilise le type `hybrid_size`.
- **Vérification** : chaque design (initial et optimisé) est recalculé avec le même FEM complet pénalisé, ce qui permet
  de comparer les deux méthodes sur une base commune.
- Rayons initiaux {geometry["radii"]}, SLSQP, densité relative cible {target} (krigeage). Densité Gmsh : recalcul direct
  du volume CAO de chaque cellule (`optimization_study_direct_density.py`).

| Cas | Méthode | Variables | Itérations | Évaluations | Temps optim. (s) | Temps / évaluation (s) | C0 (modèle optim.) | \
C final (modèle optim.) | Réduction modèle optim. (%) | C0 (FEM pénalisé) | C final (FEM pénalisé) | \
**Réduction vérifiée (%)** | Densité finale (krigeage) | Densité finale (Gmsh) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

output = save_dir / "comparison_DDM_vs_FEM.md"
observations = ""
if output.exists() and "## Observations" in output.read_text():
    observations = "\n## Observations" + output.read_text().split("## Observations", 1)[1]
output.write_text(header + "\n".join(rows) + "\n" + observations)
print(output.read_text())
