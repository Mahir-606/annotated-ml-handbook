"""The Annotated ML Handbook, Chapter 23: Model selection: cross-validation and hyperparameter tuning

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 23-sklearn-model-selection.py
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


# %% [markdown]
# ## 23.1 k-fold cross-validation · *Beginner*
#
# **cross_val_score and cross_validate**


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,
                                     KFold, StratifiedKFold, RepeatedStratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc")
print(scores.round(4), f"mean={scores.mean():.4f} sd={scores.std():.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
res = cross_validate(model, X_tr, y_tr, cv=cv,
                     scoring=["accuracy", "roc_auc", "f1"],
                     return_train_score=True)
print(pd.DataFrame(res).mean().round(4))


# %% [markdown]
# **Repeated CV for more stable estimates**


# %%
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
s = cross_val_score(model, X_tr, y_tr, cv=rcv, scoring="roc_auc")
print(f"{len(s)} scores: mean={s.mean():.4f}, 2.5–97.5 percentile = {np.percentile(s, 2.5):.4f}–{np.percentile(s, 97.5):.4f}")


# %% [markdown]
# ## 23.2 Choosing the right splitter · *Intermediate*
#
# **Visualising splitters**


# %%
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

n = 60
groups = np.repeat(np.arange(12), 5)
yy = np.r_[np.zeros(40), np.ones(20)]
splitters = [("KFold (shuffled)", KFold(4, shuffle=True, random_state=0)),
             ("StratifiedKFold", StratifiedKFold(4, shuffle=True, random_state=0)),
             ("GroupKFold", GroupKFold(4)),
             ("TimeSeriesSplit", TimeSeriesSplit(4))]
fig, axes = plt.subplots(1, 4, figsize=(14, 2.9), sharey=True)
for ax, (name, sp) in zip(axes, splitters):
    g = groups if name == "GroupKFold" else None
    for i, (tr, va) in enumerate(sp.split(np.zeros((n, 1)), yy, g)):
        row = np.full(n, np.nan); row[tr] = 0; row[va] = 1
        ax.scatter(range(n), [i] * n, c=row, cmap="coolwarm", marker="s", s=22, vmin=0, vmax=1)
    ax.scatter(range(n), [4.2] * n, c=groups, cmap="tab20", marker="s", s=22)
    ax.set(title=name, xlabel="sample index", yticks=[0, 1, 2, 3, 4.2], yticklabels=["1", "2", "3", "4", "group"])
    ax.grid(False)
axes[0].set_ylabel("CV iteration")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 23.2.1 Grouped data: the leakage you don't see · *Intermediate*
#
# **Repeated measurements: KFold is optimistic, GroupKFold is honest**


# %%
from sklearn.ensemble import RandomForestClassifier
rng = np.random.default_rng(0)
n_subj, per = 40, 10
subj = np.repeat(np.arange(n_subj), per)
subj_effect = rng.normal(0, 1, (n_subj, 5))[subj]
label = rng.integers(0, 2, n_subj)[subj]
Xg = subj_effect + rng.normal(0, 0.3, (n_subj * per, 5))
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
naive = cross_val_score(rf, Xg, label, cv=KFold(5, shuffle=True, random_state=0)).mean()
grouped = cross_val_score(rf, Xg, label, cv=GroupKFold(5), groups=subj).mean()
print(f"KFold accuracy:      {naive:.2f}  <- the model recognises subjects it has already seen")
print(f"GroupKFold accuracy: {grouped:.2f}  <- honest: near chance")


# %% [markdown]
# ### 23.2.2 Time series · *Intermediate*
#
# **Forward-chaining validation**


# %%
tscv = TimeSeriesSplit(n_splits=4, test_size=10)
for i, (tr, va) in enumerate(tscv.split(np.arange(60))):
    print(f"fold {i}: train {tr.min()}–{tr.max()}  validate {va.min()}–{va.max()}")


# %% [markdown]
# ## 23.3 Hyperparameter tuning with GridSearchCV · *Intermediate*
#
# **Tuning an SVM inside a pipeline**


# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf"],
}
search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      scoring="roc_auc", n_jobs=-1, return_train_score=True)
search.fit(X_tr, y_tr)
print("best params:", search.best_params_)
print("best CV ROC AUC:", round(search.best_score_, 4))
print("test ROC AUC (once!):", round(search.score(X_te, y_te), 4))
cvres = pd.DataFrame(search.cv_results_)
print(cvres[["param_svc__C", "param_svc__gamma", "mean_test_score", "std_test_score", "rank_test_score"]]
      .sort_values("rank_test_score").head(5).round(4).to_string(index=False))


# %% [markdown]
# **See the landscape**


# %%
pivot = cvres.pivot_table(index="param_svc__gamma", columns="param_svc__C", values="mean_test_score")
pivot.index = pivot.index.astype(str)
fig, ax = plt.subplots(figsize=(6, 3.8))
im = ax.imshow(pivot.values, cmap="viridis")
ax.set(xticks=range(pivot.shape[1]), xticklabels=pivot.columns, yticks=range(pivot.shape[0]), yticklabels=pivot.index,
       xlabel="C", ylabel="gamma", title="Grid search: CV ROC AUC")
for (i, j), v in np.ndenumerate(pivot.values):
    ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v < pivot.values.max() - 0.01 else "black", fontsize=9)
ax.grid(False)
fig.colorbar(im)
plt.show()


# %% [markdown]
# ### 23.3.1 Searching over models and preprocessing too · *Advanced*
#
# **One search, several model families**


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
grid = [
    {"clf": [LogisticRegression(max_iter=2000)], "clf__C": [0.01, 0.1, 1, 10]},
    {"clf": [RandomForestClassifier(random_state=0)], "clf__n_estimators": [200], "clf__max_depth": [None, 5],
     "scale": ["passthrough"]},
]
gs = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
print(gs.best_params_["clf"], round(gs.best_score_, 4))


# %% [markdown]
# ## 23.4 RandomizedSearchCV and successive halving · *Intermediate*
#
# **Random search over a gradient-boosting model**


# %%
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

dist = {
    "learning_rate": loguniform(1e-3, 0.5),
    "max_leaf_nodes": randint(8, 64),
    "min_samples_leaf": randint(5, 60),
    "l2_regularization": loguniform(1e-4, 10),
}
rs = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0), dist, n_iter=25,
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc",
                        random_state=0, n_jobs=-1)
rs.fit(X_tr, y_tr)
print({k: round(float(v), 4) if isinstance(v, float) else int(v) for k, v in rs.best_params_.items()})
print("best CV ROC AUC:", round(rs.best_score_, 4))


# %% [markdown]
# ## 23.5 Learning curves and validation curves · *Intermediate*
#
# **Diagnose bias vs variance**


# %%
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
LearningCurveDisplay.from_estimator(make_pipeline(StandardScaler(), SVC()), X_tr, y_tr,
                                    train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring="accuracy",
                                    ax=axes[0], score_name="accuracy", n_jobs=-1)
axes[0].set_title("Learning curve (SVC)")
ValidationCurveDisplay.from_estimator(make_pipeline(StandardScaler(), SVC()), X_tr, y_tr,
                                      param_name="svc__C", param_range=np.logspace(-3, 3, 13), cv=5,
                                      scoring="accuracy", ax=axes[1], score_name="accuracy", n_jobs=-1)
axes[1].set(xscale="log", title="Validation curve over C")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 23.6 Nested cross-validation: an unbiased estimate after tuning · *Advanced*
#
# **Inner loop tunes, outer loop evaluates**


# %%
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
tuned_svc = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                         {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
                         cv=inner, scoring="roc_auc", n_jobs=-1)
nested = cross_val_score(tuned_svc, X, y, cv=outer, scoring="roc_auc")
tuned_svc.fit(X, y)
print(f"non-nested best_score_: {tuned_svc.best_score_:.4f}")
print(f"nested CV estimate:     {nested.mean():.4f} ± {nested.std():.4f}")


# %% [markdown]
# ## Try it yourself
#
# 1. Tune `n_neighbors` (1–30) and `weights` (`"uniform"`, `"distance"`) for kNN on the breast-cancer data with `GridSearchCV`.
# 2. Use `cross_validate(..., return_estimator=True)` to inspect how much logistic-regression coefficients vary across folds.
# 3. Compare 5-fold and 10-fold CV estimates and their standard deviations for the same model.
#
# Worked solutions: [solutions-parts-5-8.ipynb](../../solutions/solutions-parts-5-8.ipynb), section *Chapter 23*.
