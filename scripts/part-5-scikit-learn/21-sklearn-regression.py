"""The Annotated ML Handbook, Chapter 21: Regression: predicting numbers

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-5-scikit-learn/21-sklearn-regression.ipynb.
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

ins = pd.read_csv("insurance.csv")
X, y = ins.drop(columns="charges"), ins["charges"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols, cat_cols = ["age", "bmi", "children"], ["sex", "smoker", "region"]
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])
print(X_tr.shape, X_te.shape)


# %%
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate(model, name):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {"model": name,
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": root_mean_squared_error(y_te, pred),
            "R2": r2_score(y_te, pred),
            "MAPE%": 100 * mean_absolute_percentage_error(y_te, pred)}


# %%
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor

results = [
    evaluate(Pipeline([("pre", pre), ("m", DummyRegressor())]), "Mean baseline"),
    evaluate(Pipeline([("pre", pre), ("m", LinearRegression())]), "Linear"),
    evaluate(Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]), "Ridge"),
    evaluate(Pipeline([("pre", pre), ("m", Lasso(alpha=10.0))]), "Lasso"),
    evaluate(Pipeline([("pre", pre), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5))]), "ElasticNet"),
]
print(pd.DataFrame(results).round(3).to_string(index=False))


# %%
lin = Pipeline([("pre", pre), ("m", LinearRegression())]).fit(X_tr, y_tr)
coefs = pd.Series(lin.named_steps["m"].coef_,
                  index=lin.named_steps["pre"].get_feature_names_out()).sort_values()
print(coefs.round(0))
print("intercept:", round(lin.named_steps["m"].intercept_, 0))


# %%
alphas = np.logspace(0, 4.3, 40)
Xt_tr = pre.fit_transform(X_tr)
names = pre.get_feature_names_out()
paths = np.array([Lasso(alpha=a, max_iter=20000).fit(Xt_tr, y_tr).coef_ for a in alphas])
fig, ax = plt.subplots(figsize=(8, 4))
for j, name in enumerate(names):
    lw = 2.4 if name in ("cat__smoker_yes", "num__age", "num__bmi") else 1
    ax.plot(alphas, paths[:, j], lw=lw, label=name.split("__")[1])
ax.set(xscale="log", xlabel="alpha (penalty strength)", ylabel="coefficient", title="Lasso regularisation path")
ax.legend(fontsize=8, ncol=2, loc="upper right")
plt.show()


# %%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

results += [
    evaluate(Pipeline([("pre", pre), ("m", KNeighborsRegressor(n_neighbors=10))]), "kNN (k=10)"),
    evaluate(Pipeline([("pre", pre), ("m", SVR(C=20000, epsilon=500))]), "SVR (RBF)"),
    evaluate(Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=5, random_state=0))]), "Tree (depth 5)"),
    evaluate(Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))]), "Random forest"),
    evaluate(Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=0))]), "Gradient boosting"),
    evaluate(Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=0))]), "HistGradientBoosting"),
]
table = pd.DataFrame(results).sort_values("RMSE").round(3)
print(table.to_string(index=False))


# %%
fig, ax = plt.subplots(figsize=(8, 4))
t = table.sort_values("RMSE", ascending=False)
colors = ["#bbb" if m == "Mean baseline" else "#0f9b62" for m in t["model"]]
bars = ax.barh(t["model"], t["RMSE"], color=colors)
ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
ax.set(xlabel="test RMSE (charges)", title="Model comparison on the insurance data")
ax.grid(axis="y", visible=False)
plt.show()


# %%
from sklearn.tree import export_text
tree = Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=2, random_state=0))]).fit(X_tr, y_tr)
print(export_text(tree.named_steps["m"], feature_names=list(pre.get_feature_names_out())))


# %%
from sklearn.metrics import PredictionErrorDisplay

rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))]).fit(X_tr, y_tr)
pred = rf.predict(X_te)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
PredictionErrorDisplay.from_predictions(y_te, pred, kind="actual_vs_predicted", ax=axes[0],
                                        scatter_kwargs={"s": 12, "alpha": 0.6})
axes[0].set_title("Actual vs predicted")
PredictionErrorDisplay.from_predictions(y_te, pred, kind="residual_vs_predicted", ax=axes[1],
                                        scatter_kwargs={"s": 12, "alpha": 0.6})
axes[1].set_title("Residuals vs predicted")
plt.tight_layout()
plt.show()


# %%
rng = np.random.default_rng(0)
y_true_arr, pred_arr = y_te.to_numpy(), pred
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
    boot.append(root_mean_squared_error(y_true_arr[idx], pred_arr[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"RMSE = {root_mean_squared_error(y_true_arr, pred_arr):.0f}  (95% CI {lo:.0f} to {hi:.0f})")
