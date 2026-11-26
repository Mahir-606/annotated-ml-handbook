"""The Annotated ML Handbook, Chapter 4: Functions: reusable, testable logic

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-1-python/04-python-functions.ipynb.
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
def bmi(weight_kg, height_m):
    """Return body-mass index (kg / m^2)."""
    value = weight_kg / height_m ** 2
    return round(value, 1)

print(bmi(70, 1.75))
print(bmi(height_m=1.6, weight_kg=55))
print(bmi.__doc__)


# %%
def mean_and_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5

m, s = mean_and_std([2, 4, 4, 4, 5, 5, 7, 9])
print(f"mean={m:.3f}, sd={s:.3f}")


# %%
def train(model, epochs=10, lr=0.001, *, verbose=False):
    return f"{model}: epochs={epochs}, lr={lr}, verbose={verbose}"

print(train("mlp"))
print(train("mlp", 50, verbose=True))

def summarize(*values, **options):
    print("values :", values)
    print("options:", options)

summarize(1, 2, 3, unit="cm", digits=2)
nums = [4, 5]
cfg = {"unit": "kg"}
summarize(*nums, **cfg)


# %%
def add_bad(item, history=[]):
    history.append(item)
    return history

print(add_bad("a"))
print(add_bad("b"))

def add_good(item, history=None):
    if history is None:
        history = []
    history.append(item)
    return history

print(add_good("a"), add_good("b"))


# %%
threshold = 0.5

def classify(p):
    label = 1 if p >= threshold else 0
    return label

def bump():
    threshold = 0.9
    return threshold

print(classify(0.7), bump(), threshold)


# %%
square = lambda x: x ** 2
print(square(7))
people = [("Ana", 31), ("Bo", 25), ("Cy", 40)]
print(sorted(people, key=lambda p: p[1]))
print(list(map(lambda x: x * 10, [1, 2, 3])))
print(list(filter(lambda x: x > 1, [0, 1, 2, 3])))

def apply_twice(f, x):
    return f(f(x))
print(apply_twice(square, 3))


# %%
import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed * 1000:.2f} ms")
        return result
    return wrapper

@timed
def slow_sum(n):
    return sum(i * i for i in range(n))

total = slow_sum(200_000)
print(total)


# %%
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(fib(80))
print(fib.cache_info())


# %%
def batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]

for b in batches(list(range(10)), 4):
    print(b)
squares = (x * x for x in range(5))
print(sum(squares))


# %%
def normalize(values: list[float], eps: float = 1e-12) -> list[float]:
    """Scale values to the range [0, 1].

    Parameters
    ----------
    values : list of float
        Raw measurements.
    eps : float, default 1e-12
        Guards against division by zero when all values are equal.

    Returns
    -------
    list of float
    """
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo + eps) for v in values]

print(normalize([10.0, 15.0, 20.0]))
print(normalize.__annotations__)
