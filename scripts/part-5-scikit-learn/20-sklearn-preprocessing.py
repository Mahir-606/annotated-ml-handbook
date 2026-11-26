"""The Annotated ML Handbook, Chapter 20: Preprocessing, feature engineering and pipelines

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-5-scikit-learn/20-sklearn-preprocessing.ipynb.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = np.array([[1.0, 200.0], [2.0, 300.0], [3.0, 250.0], [4.0, 5000.0]])
for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
    Xs = scaler.fit_transform(X)
    print(f"{type(scaler).__name__:<15}", Xs[:, 1].round(2))
sc = StandardScaler().fit(X)
print("learned mean_:", sc.mean_, " scale_:", sc.scale_.round(1))
print(sc.inverse_transform(sc.transform(X))[0])


# %%
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.DataFrame({"region": ["north", "south", "east", "south"],
                   "size": ["S", "L", "M", "M"]})
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
print(ohe.fit_transform(df[["region"]]))
print(ohe.get_feature_names_out())
print(ohe.transform(pd.DataFrame({"region": ["west"]})))
ord_enc = OrdinalEncoder(categories=[["S", "M", "L"]])
print(ord_enc.fit_transform(df[["size"]]).ravel())


# %%
from sklearn.impute import SimpleImputer, KNNImputer

Xm = np.array([[1.0, 10.0], [np.nan, 12.0], [3.0, np.nan], [4.0, 16.0]])
print(SimpleImputer(strategy="median").fit_transform(Xm))
print(SimpleImputer(strategy="mean", add_indicator=True).fit_transform(Xm))
print(KNNImputer(n_neighbors=2).fit_transform(Xm).round(2))
print(SimpleImputer(strategy="most_frequent").fit_transform(np.array([["a"], ["b"], ["a"], [np.nan]], dtype=object)).ravel())


# %%
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_tr, y_tr)
print("test accuracy:", round(pipe.score(X_te, y_te), 4))
print(pipe.named_steps["scale"].mean_[:3].round(2))
quick = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print([name for name, _ in quick.steps])


# %%
from sklearn.compose import ColumnTransformer, make_column_selector

ins = pd.read_csv("insurance.csv")
X = ins.drop(columns="charges")
y = ins["charges"]
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols),
], remainder="drop")

Xt = pre.fit_transform(X)
print(Xt.shape)
print(pre.get_feature_names_out())


# %%
from sklearn.linear_model import Ridge
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
model = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
model.fit(X_tr, y_tr)
print("test R2:", round(model.score(X_te, y_te), 4))
new_customer = pd.DataFrame([{"age": 45, "sex": "female", "bmi": np.nan, "children": 2,
                              "smoker": "no", "region": "southwest"}])
print("predicted charges:", model.predict(new_customer).round(0))


# %%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X_noise = rng.normal(size=(100, 5000))
y_noise = rng.integers(0, 2, 100)

X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise, cv=5).mean()

right_pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
right = cross_val_score(right_pipe, X_noise, y_noise, cv=5).mean()
print(f"selection outside CV: {wrong:.2f}   <- looks impressive, entirely fake")
print(f"selection inside CV : {right:.2f}   <- honest: no signal")


# %%
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, KBinsDiscretizer, SplineTransformer

log_tf = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
print(log_tf.fit_transform(np.array([[0.0], [9.0], [99.0]])).ravel().round(3))
poly = PolynomialFeatures(degree=2, include_bias=False)
print(poly.fit_transform([[2, 3]]), poly.get_feature_names_out(["a", "b"]))
bins = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf")
print(bins.fit_transform(np.array([[1], [2], [3], [10], [20], [30]])).ravel())
spl = SplineTransformer(n_knots=4, degree=3)
print(spl.fit_transform(np.linspace(0, 1, 5).reshape(-1, 1)).shape)


# %%
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatures(BaseEstimator, TransformerMixin):
    """Add the ratio of two columns as a new column."""
    def __init__(self, num="bmi", den="age"):
        self.num, self.den = num, den
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[f"{self.num}_per_{self.den}"] = X[self.num] / X[self.den]
        return X

print(RatioFeatures().fit_transform(ins[["age", "bmi"]].head(3)).round(3))


# %%
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print(list(pipe.get_params().keys())[-6:])
pipe.set_params(logisticregression__C=0.1)
print(pipe.named_steps["logisticregression"].C)
print(pipe)
