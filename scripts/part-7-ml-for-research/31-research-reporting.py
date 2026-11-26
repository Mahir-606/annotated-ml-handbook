"""The Annotated ML Handbook, Chapter 31: Reproducibility, leakage checklists and writing it up

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-7-ml-for-research/31-research-reporting.ipynb.
Cells are marked "# %%" so you can run them one at a time in VS Code, PyCharm or Spyder.
"""

# %%
# Repository setup (not part of the book's examples): run this cell first.
import os, sys
from pathlib import Path

# Files the examples create (CSV, models, figures) go into ./outputs next to the notebook.
OUT = Path.cwd() if Path.cwd().name == "outputs" else Path.cwd() / "outputs"
OUT.mkdir(exist_ok=True)
os.chdir(OUT)

# insurance.csv is the practice dataset created in Chapter 11 and reused by later chapters.
# If it is not here yet, copy it from the repository's data/ folder (or download it in Colab).
if not Path("insurance.csv").exists():
    import shutil, urllib.request
    for base in OUT.parents:
        if (base / "data" / "insurance.csv").exists():
            shutil.copy(base / "data" / "insurance.csv", "insurance.csv")
            break
    else:
        try:
            urllib.request.urlretrieve("https://raw.githubusercontent.com/Mahir-606/"
                                       "annotated-ml-handbook/main/data/insurance.csv", "insurance.csv")
        except Exception as e:
            print("Could not fetch insurance.csv; run Chapter 11 first.", e)

# The figure style used in the book.
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams.update({
    "figure.figsize": (7.2, 4.0), "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#1f4fff", "#e0245e", "#0f9b62", "#ff6a00",
                                     "#7b3ff2", "#0091b3", "#d99a00", "#d42a2a"]),
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
    "font.size": 10.5, "legend.frameon": False, "image.cmap": "viridis",
})

# Plotly: interactive in Jupyter/Colab; tools/run_all.py stores static PNGs so GitHub can show them.
try:
    import plotly.io as pio
    pio.renderers.default = os.environ.get("PLOTLY_RENDERER") or (
        "colab" if "google.colab" in sys.modules else "notebook_connected")
except ImportError:
    pass


# %%
import json, time, platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CONFIG = {
    "seed": 2026,
    "cv": {"n_splits": 5, "n_repeats": 3},
    "scoring": ["roc_auc", "balanced_accuracy"],
    "models": {"logreg": {"C": 1.0}, "rf": {"n_estimators": 300}},
}

def build(name, params, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **params))
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"unknown model {name!r}")

def main(cfg):
    run_dir = Path("results") / time.strftime("run_%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    cv = RepeatedStratifiedKFold(**cfg["cv"], random_state=cfg["seed"])
    out = {"config": cfg, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                       "sklearn": sklearn.__version__}, "results": {}}
    for name, params in cfg["models"].items():
        r = cross_validate(build(name, params, cfg["seed"]), X, y, cv=cv, scoring=cfg["scoring"], n_jobs=-1)
        out["results"][name] = {m: {"mean": round(float(r[f"test_{m}"].mean()), 4),
                                    "sd": round(float(r[f"test_{m}"].std()), 4)} for m in cfg["scoring"]}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out

result = main(CONFIG)
print(json.dumps(result["results"], indent=2))


# %%
import hashlib
Path("data.csv").write_text("a,b\n1,2\n3,4\n")
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
print("data.csv sha256:", sha256("data.csv")[:16], "...")


# %%
from sklearn.model_selection import permutation_test_score
X, y = load_breast_cancer(return_X_y=True)
score, perm_scores, pvalue = permutation_test_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y,
    scoring="roc_auc", cv=5, n_permutations=100, random_state=0, n_jobs=-1)
print(f"real AUC={score:.3f} | shuffled-label AUCs: mean={perm_scores.mean():.3f}, max={perm_scores.max():.3f} | p={pvalue:.3f}")
