"""The Annotated ML Handbook, Chapter 1: Values, variables and types

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-1-python/01-python-basics.ipynb.
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
age = 29
height_m = 1.74
name = "Ada"
is_student = True
age = age + 1
print(name, age, height_m, is_student)
print(type(age), type(height_m), type(name), type(is_student))


# %%
a, b = 17, 5
print(a + b, a - b, a * b)
print(a / b)
print(a // b)
print(a % b)
print(a ** 2)
print(-7 // 2, -7 % 2)
print(round(3.14159, 2), abs(-4), max(3, 9, 2), min(3, 9, 2))


# %%
import math
x = 0.1 + 0.2
print(x)
print(x == 0.3)
print(math.isclose(x, 0.3))
print(math.sqrt(16), math.log(math.e), math.pi)
print(float("inf") > 1e308, math.isnan(float("nan")))


# %%
s = "Machine Learning"
print(s[0], s[-1])
print(s[0:7])
print(s[8:], s[:7], s[::-1])
print(s.lower(), s.upper())
print(s.replace("Machine", "Deep"))
print(s.split(" "))
print("-".join(["a", "b", "c"]))
print("  padded  ".strip() + "|")
print("Learn" in s, s.startswith("Mach"), s.find("L"), len(s))


# %%
model, acc, n = "RandomForest", 0.934567, 12500
print(f"{model} accuracy: {acc}")
print(f"{model} accuracy: {acc:.2f}")
print(f"{model} accuracy: {acc:.1%}")
print(f"Samples: {n:,}")
print(f"|{model:<14}|{acc:>8.3f}|")
print(f"{acc=}")
print(f"{1e-5:.2e}")


# %%
x = 7
print(x > 5, x == 7, x != 7)
print(3 < x <= 10)
print(x > 5 and x % 2 == 0)
print(x > 5 or x % 2 == 0)
print(not x > 5)
print(bool(0), bool(""), bool([]), bool(None))
print(bool(42), bool("hi"), bool([0]))


# %%
print(int("42") + 1)
print(float("3.5") * 2)
print(str(3.5) + " kg")
print(int(3.99))
print(int(float("3.7")))


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
age = 30
print("Age: " + age)
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %%
result = None
if result is None:
    print("No result yet")
print(type(None))
