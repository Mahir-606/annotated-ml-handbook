"""The Annotated ML Handbook, Chapter 3: Control flow: decisions and loops

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-1-python/03-python-control.ipynb.
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
p_value = 0.032
if p_value < 0.01:
    verdict = "strong evidence"
elif p_value < 0.05:
    verdict = "moderate evidence"
else:
    verdict = "insufficient evidence"
print(verdict)
label = "pass" if p_value < 0.05 else "fail"
print(label)


# %%
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit.upper())
for i in range(3):
    print("iteration", i)
for i, fruit in enumerate(fruits, start=1):
    print(i, fruit)
prices = [1.2, 0.5, 3.0]
for fruit, price in zip(fruits, prices):
    print(f"{fruit}: ${price:.2f}")


# %%
x = 10.0
steps = 0
while abs(x * x - 2) > 1e-10:
    x = (x + 2 / x) / 2
    steps += 1
print(f"sqrt(2) ~ {x:.10f} after {steps} steps")


# %%
for n in [3, 8, -1, 12, 5]:
    if n < 0:
        print("negative value found, stopping")
        break
    if n > 10:
        continue
    print("processing", n)
for n in [2, 4, 6]:
    if n % 2 == 1:
        break
else:
    print("all numbers were even")


# %%
def describe(task):
    match task:
        case "regression" | "forecasting":
            return "predict a number"
        case "classification":
            return "predict a category"
        case {"type": "clustering", "k": k}:
            return f"group into {k} clusters"
        case _:
            return "unknown task"
print(describe("forecasting"))
print(describe({"type": "clustering", "k": 4}))
print(describe("translation"))


# %%
r = range(10**12)
print(r[5], len(r))
z = zip([1, 2], ["a", "b"])
print(z)
print(list(z))
print(list(z))
