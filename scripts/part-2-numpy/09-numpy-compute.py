"""The Annotated ML Handbook, Chapter 9: Computing with NumPy: vectorization, broadcasting, statistics, linear algebra

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 09-numpy-compute.py
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
# ## 9.1 Universal functions (ufuncs) · *Beginner*
#
# **Element-wise maths**


# %%
import numpy as np
x = np.array([1.0, 4.0, 9.0, 16.0])
print(x + 1, x * 2, x ** 0.5)
print(np.sqrt(x), np.log(x), np.exp([0, 1]))
print(np.round(np.log10(x), 3))
a, b = np.array([1, 5, 3]), np.array([4, 2, 3])
print(a + b, a * b, a > b, np.maximum(a, b))
print(np.clip(np.array([-2, 0.5, 3]), 0, 1))
print(np.abs([-3, 3]), np.sign([-3, 0, 3]))


# %% [markdown]
# ## 9.2 Broadcasting · *Intermediate*
#
# **Broadcasting in action**


# %%
X = np.array([[1, 2, 3],
              [4, 5, 6]])
print(X + 10)
row = np.array([10, 20, 30])
print(X + row)
col = np.array([[100], [200]])
print(X + col)
mu = X.mean(axis=0)
sd = X.std(axis=0)
print((X - mu) / sd)


# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 2.9))
def grid(ax, x0, vals, color, faded=None, shape=None):
    rows, cols = vals.shape
    for i in range(rows):
        for j in range(cols):
            alpha = 0.35 if (faded is not None and faded[i, j]) else 1
            ax.add_patch(plt.Rectangle((x0 + j, rows - 1 - i), 0.92, 0.92, color=color, alpha=alpha))
            ax.text(x0 + j + 0.46, rows - 1 - i + 0.46, vals[i, j], ha="center", va="center", color="white", fontweight="bold", fontsize=11)
X = np.array([[1, 2, 3], [4, 5, 6]])
for ax, other, label in [(axes[0], np.array([[10, 20, 30]]), "(2,3) + (3,)"), (axes[1], np.array([[100], [200]]), "(2,3) + (2,1)")]:
    ax.set_xlim(-0.3, 13.5); ax.set_ylim(-0.6, 2.4); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title(label, loc="left", family="monospace", fontweight="bold")
    grid(ax, 0, X, "#7b3ff2")
    ax.text(3.4, 0.95, "+", fontsize=20, ha="center", va="center")
    big = np.broadcast_to(other, (2, 3))
    faded = np.ones((2, 3), bool)
    if other.shape == (1, 3): faded[0, :] = False
    else: faded[:, 0] = False
    grid(ax, 4.0, big, "#e0245e", faded)
    ax.text(7.5, 0.95, "=", fontsize=20, ha="center", va="center")
    grid(ax, 8.2, X + big, "#0f9b62")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Incompatible shapes**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
X = np.ones((2, 3))
v = np.array([1, 2])
X + v
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 9.2.1 Broadcasting power move: pairwise distances · *Advanced*
#
# **All pairwise Euclidean distances, no loops**


# %%
P = np.array([[0, 0], [3, 4], [6, 8]])
diff = P[:, None, :] - P[None, :, :]
D = np.sqrt((diff ** 2).sum(axis=-1))
print(D)


# %% [markdown]
# ## 9.3 Statistics and aggregation · *Beginner*
#
# **Descriptive statistics**


# %%
rng = np.random.default_rng(seed=0)
x = rng.normal(loc=50, scale=10, size=1000)
print(f"mean={x.mean():.2f}  median={np.median(x):.2f}  std={x.std(ddof=1):.2f}")
print(f"min={x.min():.1f} max={x.max():.1f} range={np.ptp(x):.1f}")
print("quartiles:", np.percentile(x, [25, 50, 75]).round(2))
print("cumsum:", np.cumsum([1, 2, 3, 4]))
y = 2 * x + rng.normal(0, 5, size=1000)
print("corr matrix:\n", np.corrcoef(x, y).round(3))


# %% [markdown]
# ## 9.4 Sorting, searching, unique values · *Beginner*
#
# **Order and counts**


# %%
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.sort(a))
print(np.argsort(a))
print(a[np.argsort(a)[::-1][:3]])
vals, counts = np.unique(a, return_counts=True)
print(vals, counts)
print(np.searchsorted([10, 20, 30], 25))
print(np.argmax(a), np.argmin(a))


# %% [markdown]
# ## 9.5 Missing values: NaN · *Intermediate*
#
# **Detecting and handling NaN**


# %%
d = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
print(d.mean())
print(np.nanmean(d), np.nansum(d), np.nanmax(d))
print(np.isnan(d), np.isnan(d).sum())
print(d[~np.isnan(d)])
print(np.nan_to_num(d, nan=0.0))
print(np.nan == np.nan)


# %% [markdown]
# ## 9.6 Random numbers · *Beginner*
#
# **The Generator API (use this, not np.random.seed)**


# %%
rng = np.random.default_rng(42)
print(rng.random(3))
print(rng.integers(0, 10, size=5))
print(rng.normal(0, 1, size=(2, 3)).round(2))
print(rng.choice(["A", "B", "C"], size=5, p=[0.6, 0.3, 0.1]))
idx = rng.permutation(10)
print(idx)


# %% [markdown]
# ## 9.7 Linear algebra · *Intermediate*
#
# **Matrix operations ML is built on**


# %%
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
b = np.array([3.0, 5.0])
print(A @ b)
print(A @ A)
print(np.linalg.det(A))
x = np.linalg.solve(A, b)
print(x, np.allclose(A @ x, b))
vals, vecs = np.linalg.eigh(A)
print(vals.round(4))
print(np.linalg.norm([3, 4]))
U, S, Vt = np.linalg.svd(np.arange(6).reshape(2, 3).astype(float))
print(S.round(4))


# %% [markdown]
# **Inner dimensions must agree**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
A = np.ones((3, 2))
B = np.ones((3, 2))
A @ B
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## Try it yourself
#
# 1. Min-max scale each column of a random `(100, 4)` matrix to [0, 1] using broadcasting.
# 2. Using `np.unique(..., return_counts=True)`, report the class proportions of `rng.choice([0, 1], 500, p=[.8, .2])`.
# 3. Solve the system 3x + 2y = 7, x − y = −1 with `np.linalg.solve`.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 9*.
