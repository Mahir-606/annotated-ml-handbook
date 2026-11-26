"""The Annotated ML Handbook, Chapter 6: Errors, files and modules

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 06-python-errors-files.py
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
# ## 6.1 Exceptions and try/except · *Beginner*
#
# **Catching specific exceptions**


# %%
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero; returning None")
        return None
    except TypeError as err:
        print("Bad types:", err)
        return None
    else:
        return result
    finally:
        print("done:", a, b)

print(safe_divide(10, 4))
print(safe_divide(1, 0))
print(safe_divide("1", 2))


# %% [markdown]
# ### 6.1.1 Raising your own exceptions · *Intermediate*
#
# **Validating inputs**


# %%
class DataValidationError(ValueError):
    pass

def check_probabilities(ps):
    if not ps:
        raise DataValidationError("empty list of probabilities")
    bad = [p for p in ps if not 0 <= p <= 1]
    if bad:
        raise DataValidationError(f"{len(bad)} value(s) outside [0, 1]: {bad[:3]}")
    return True

try:
    check_probabilities([0.2, 1.4, -0.1])
except DataValidationError as e:
    print(type(e).__name__, "->", e)


# %% [markdown]
# **assert stops the program when an assumption fails**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
n_train, n_labels = 120, 118
assert n_train == n_labels, f"features ({n_train}) and labels ({n_labels}) differ"
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 6.2 Reading a traceback · *Beginner*
#
# **A traceback through two functions**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
def load_scores(record):
    return record["scores"]

def average(record):
    s = load_scores(record)
    return sum(s) / len(s)

average({"score": [1, 2, 3]})
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 6.3 Working with files · *Beginner*
#
# **Writing and reading text files**


# %%
with open("notes.txt", "w", encoding="utf-8") as f:
    f.write("epoch,loss\n")
    for epoch, loss in enumerate([0.9, 0.6, 0.45], 1):
        f.write(f"{epoch},{loss}\n")

with open("notes.txt", encoding="utf-8") as f:
    header = f.readline().strip()
    rows = [line.strip().split(",") for line in f]
print(header, rows)
with open("notes.txt", "a", encoding="utf-8") as f:
    f.write("4,0.40\n")


# %% [markdown]
# ### 6.3.1 Paths with pathlib · *Beginner*
#
# **Building, checking and listing paths**


# %%
from pathlib import Path

root = Path("project")
data_dir = root / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)
(data_dir / "a.csv").write_text("x\n1\n")
(data_dir / "b.csv").write_text("x\n2\n")
print(sorted(p.name for p in data_dir.glob("*.csv")))
p = data_dir / "a.csv"
print(p.exists(), p.suffix, p.stem, p.parent)


# %% [markdown]
# ### 6.3.2 JSON and CSV · *Beginner*
#
# **JSON for configs and results, CSV for tables**


# %%
import json, csv

results = {"model": "rf", "acc": 0.91, "params": {"n_estimators": 300}}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("results.json") as f:
    loaded = json.load(f)
print(loaded["params"]["n_estimators"])

rows = [{"id": 1, "group": "A"}, {"id": 2, "group": "B"}]
with open("groups.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "group"])
    writer.writeheader()
    writer.writerows(rows)
with open("groups.csv", newline="") as f:
    print(list(csv.DictReader(f)))


# %% [markdown]
# ## 6.4 Modules, packages and imports · *Beginner*
#
# **Import styles**


# %%
import math
import numpy as np
from statistics import mean, median
from collections import Counter as C
print(math.floor(2.7), np.pi > 3, mean([1, 2, 3]), median([3, 1, 2]), C("aab"))


# %% [markdown]
# **utils.py: a reusable module**
#
# *Shown for reference, not run automatically (it waits for input, needs your own files, or would change your system). Copy it into a cell to try it.*
#
# ```python
# """Helper functions for the project."""
#
# def zscore(xs):
#     m = sum(xs) / len(xs)
#     sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
#     return [(x - m) / sd for x in xs]
#
# if __name__ == "__main__":
#     print(zscore([1, 2, 3]))
# ```
#
# **main.py: using it**
#
# *Shown for reference, not run automatically (it waits for input, needs your own files, or would change your system). Copy it into a cell to try it.*
#
# ```python
# from utils import zscore
# print(zscore([10, 20, 30]))
# ```
#
# ## 6.5 Logging instead of print · *Intermediate*
#
# **Basic logging setup**


# %%
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("train")
log.info("starting run with lr=%s", 0.001)
log.warning("validation loss increased")
log.debug("not shown: DEBUG is below the INFO level")


# %% [markdown]
# ## Try it yourself
#
# 1. Write `read_numbers(path)` that returns a list of floats from a text file, skipping lines that can't be converted (catch `ValueError`).
# 2. Save a dict of hyperparameters to `config.json` and load it back.
# 3. Use `Path.glob("**/*.csv")` to count CSV files in a folder tree.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 6*.
