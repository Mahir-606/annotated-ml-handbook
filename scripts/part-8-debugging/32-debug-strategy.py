"""The Annotated ML Handbook, Chapter 32: How to debug: reading tracebacks and a systematic method

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-8-debugging/32-debug-strategy.ipynb.
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
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import numpy as np
from sklearn.linear_model import LinearRegression

def fit_model(features, target):
    model = LinearRegression()
    model.fit(features, target)
    return model

heights = np.array([150, 160, 170, 180])
weights = np.array([50, 58, 69, 80])
fit_model(heights, weights)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %%
import pandas as pd
def inspect(name, obj):
    """Print the facts that explain 90% of data bugs."""
    info = f"{name}: type={type(obj).__name__}"
    if hasattr(obj, "shape"): info += f" shape={obj.shape}"
    if hasattr(obj, "dtype"): info += f" dtype={obj.dtype}"
    elif hasattr(obj, "dtypes"): info += f" dtypes={dict(obj.dtypes.astype(str))}"
    if hasattr(obj, "isna"): info += f" n_missing={int(obj.isna().sum().sum())}"
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == "f": info += f" n_nan={int(np.isnan(obj).sum())}"
    print(info)

df = pd.DataFrame({"x": [1.0, None, 3.0], "g": ["a", "b", "a"]})
inspect("heights", heights)
inspect("df", df)
inspect("df[['x']]", df[["x"]])
inspect("df['x']", df["x"])


# %%
lr, epoch, loss = 0.01, 3, float("nan")
print(f"{lr=} {epoch=} {loss=}")
assert not np.isnan(loss) or epoch > 0, "loss is NaN at epoch 0"


# %%
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        np.array([1.0, 2.0]) / np.array([0.0, 1.0])
    except RuntimeWarning as w:
        print("caught:", w)


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[1.0], [2.0], [np.nan]])
y = np.array([0, 1, 0])
LogisticRegression().fit(X, y)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %%
def validate(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"age": "int64", "bmi": "float64", "smoker": "str"}
    missing_cols = set(expected) - set(df.columns)
    if missing_cols:
        raise KeyError(f"missing columns: {sorted(missing_cols)}")
    for col, dt in expected.items():
        if str(df[col].dtype) != dt:
            raise TypeError(f"{col}: expected {dt}, got {df[col].dtype}")
    if not df["age"].between(0, 120).all():
        raise ValueError(f"age out of range: {df.loc[~df['age'].between(0, 120), 'age'].tolist()[:5]}")
    return df

ok = validate(pd.read_csv("insurance.csv"))
print("validated", ok.shape)
