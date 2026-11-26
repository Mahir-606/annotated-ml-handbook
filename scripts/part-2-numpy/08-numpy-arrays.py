"""The Annotated ML Handbook, Chapter 8: NumPy arrays: shape, dtype and indexing

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-2-numpy/08-numpy-arrays.ipynb.
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
import time

values = list(range(1_000_000))
arr = np.arange(1_000_000)

t0 = time.perf_counter()
squared_list = [v * v for v in values]
t1 = time.perf_counter()
squared_arr = arr * arr
t2 = time.perf_counter()
print(f"list: {(t1 - t0) * 1000:.1f} ms | array: {(t2 - t1) * 1000:.1f} ms | speed-up ~{(t1 - t0) / (t2 - t1):.0f}x")
print(squared_list[:5], squared_arr[:5])


# %%
a = np.array([1, 2, 3])
M = np.array([[1, 2, 3], [4, 5, 6]])
print(a, M, sep="\n")
print(np.zeros((2, 3)))
print(np.ones(4), np.full((2, 2), 7))
print(np.eye(3))
print(np.arange(0, 10, 2.5))
print(np.linspace(0, 1, 5))
print(np.empty(3).shape)


# %%
X = np.array([[1.5, 2.0, 3.1],
              [4.0, 5.2, 6.3]])
print("shape:", X.shape)
print("ndim :", X.ndim)
print("size :", X.size)
print("dtype:", X.dtype)
print(np.array([1, 2]).dtype, np.array([True, False]).dtype, np.array(["a", "bc"]).dtype)


# %%
x = np.array([1, 2, 3], dtype=np.float32)
print(x.dtype, x.nbytes, "bytes")
y = np.array([1.7, -2.9, 3.5]).astype(int)
print(y)
mixed = np.array([1, 2.5, "3"])
print(mixed, mixed.dtype)
small = np.array([250, 251], dtype=np.uint8)
print(small + 10)


# %%
a = np.arange(12)
B = a.reshape(3, 4)
print(B)
print(a.reshape(2, -1).shape)
print(B.T.shape)
print(B.ravel())
v = np.array([1, 2, 3])
print(v.shape, v[:, np.newaxis].shape, v[np.newaxis, :].shape)
print(v.reshape(-1, 1).shape)


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
np.arange(10).reshape(3, 4)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %%
A = np.arange(1, 13).reshape(3, 4)
print(A)
print(A[0, 2])
print(A[1])
print(A[:, 1])
print(A[0:2, 1:3])
print(A[-1, ::-1])
print(A[:, [0, 3]])


# %%
temps = np.array([18.5, 22.1, 30.4, 15.0, 27.8, 33.2])
mask = temps > 25
print(mask)
print(temps[mask])
print(temps[(temps > 20) & (temps < 30)])
print(mask.sum(), mask.mean())
temps[temps < 16] = 16
print(temps)
print(np.where(temps > 25, "hot", "ok"))
print(np.flatnonzero(temps > 25))


# %%
a = np.arange(6)
s = a[1:4]
s[0] = 100
print(a)
c = a[1:4].copy()
c[0] = -1
print(a, c)
f = a[[0, 1]]
f[0] = 999
print(a[0], np.shares_memory(a, s), np.shares_memory(a, f))


# %%
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
print(np.concatenate([x, y], axis=0))
print(np.vstack([x, y]))
print(np.hstack([x, x]))
print(np.column_stack([[1, 2, 3], [4, 5, 6]]))
print(np.stack([x, x]).shape)
top, bottom = np.split(np.arange(8), 2)
print(top, bottom)


# %%
scores = np.array([[70, 80, 90],
                   [60, 65, 70],
                   [95, 85, 75]])
print(scores.sum())
print(scores.mean(axis=0))
print(scores.mean(axis=1))
print(scores.max(axis=1), scores.argmax(axis=1))
print(scores.mean(axis=0, keepdims=True).shape)


# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
data = np.array([[70, 80, 90], [60, 65, 70], [95, 85, 75]])
for ax, axis, title, col in [(axes[0], 0, "scores.mean(axis=0)", "#7b3ff2"), (axes[1], 1, "scores.mean(axis=1)", "#e0245e")]:
    ax.set_xlim(-0.6, 4.2 if axis == 1 else 2.6); ax.set_ylim(-1.7 if axis == 0 else -0.6, 2.6); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", family="monospace")
    means = data.mean(axis=axis)
    for i in range(3):
        for j in range(3):
            k = j if axis == 0 else i
            ax.add_patch(plt.Rectangle((j - 0.45, 2 - i - 0.45), 0.9, 0.9, color=col, alpha=0.10 + 0.12 * k, ec="white"))
            ax.text(j, 2 - i, data[i, j], ha="center", va="center", fontsize=12)
    if axis == 0:
        for j in range(3):
            ax.annotate("", xy=(j, -0.75), xytext=(j, -0.5), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.add_patch(plt.Rectangle((j - 0.45, -1.55), 0.9, 0.75, color=col))
            ax.text(j, -1.18, f"{means[j]:.1f}", ha="center", va="center", color="white", fontweight="bold")
    else:
        for i in range(3):
            ax.annotate("", xy=(2.95, 2 - i), xytext=(2.5, 2 - i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.add_patch(plt.Rectangle((3.0, 2 - i - 0.38), 1.0, 0.76, color=col))
            ax.text(3.5, 2 - i, f"{means[i]:.1f}", ha="center", va="center", color="white", fontweight="bold")
plt.tight_layout(); plt.show()
