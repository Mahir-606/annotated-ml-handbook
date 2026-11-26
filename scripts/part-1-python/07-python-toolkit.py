"""The Annotated ML Handbook, Chapter 7: The Pythonic toolkit

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-1-python/07-python-toolkit.ipynb.
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
from collections import Counter, defaultdict, namedtuple, deque

labels = ["cat", "dog", "cat", "bird", "cat", "dog"]
counts = Counter(labels)
print(counts, counts.most_common(1))
print(counts["fish"])

by_group = defaultdict(list)
for name, group in [("ana", "A"), ("bo", "B"), ("cy", "A")]:
    by_group[group].append(name)
print(dict(by_group))

Point = namedtuple("Point", ["x", "y"])
p = Point(2, 5)
print(p.x, p[1], p)

window = deque(maxlen=3)
for loss in [0.9, 0.7, 0.6, 0.55, 0.54]:
    window.append(loss)
print(list(window), sum(window) / len(window))


# %%
from itertools import product, combinations, permutations, chain, islice, accumulate

grid = {"lr": [0.1, 0.01], "depth": [3, 5]}
for lr, depth in product(grid["lr"], grid["depth"]):
    print(f"lr={lr}, depth={depth}")
print(list(combinations(["A", "B", "C"], 2)))
print(len(list(permutations(range(4)))))
print(list(chain([1, 2], (3, 4), {5})))
print(list(islice(range(100), 5, 10)))
print(list(accumulate([1, 2, 3, 4])))


# %%
from functools import partial, reduce

def scale(x, factor, offset=0):
    return x * factor + offset

to_percent = partial(scale, factor=100)
print(to_percent(0.257), to_percent(0.5, offset=1))
print(reduce(lambda a, b: a * b, [1, 2, 3, 4, 5]))


# %%
import time
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f} s")

with timer("sum of squares"):
    total = sum(i * i for i in range(300_000))


# %%
import random

random.seed(42)
print(random.random(), random.randint(1, 6))
items = list(range(10))
random.shuffle(items)
print(items, random.choice(items), random.sample(items, 3))
random.seed(42)
print(random.random())


# %%
from datetime import datetime, timedelta, date

d = datetime(2026, 3, 14, 9, 30)
print(d.isoformat(), d.strftime("%d %b %Y"))
print(d + timedelta(days=30))
parsed = datetime.strptime("2026-01-05", "%Y-%m-%d")
print((d - parsed).days, date(2026, 3, 14).weekday())


# %%
import re

text = "Sample S-104 measured 12.5 mg on 2026-02-01; sample S-221 measured 7 mg."
ids = re.findall(r"S-\d+", text)
doses = re.findall(r"(\d+(?:\.\d+)?) mg", text)
date_m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
print(ids, [float(x) for x in doses], date_m.groups())
print(re.sub(r"\s+", " ", "too    many   spaces"))


# %%
import timeit

loop_time = timeit.timeit("s = 0\nfor i in range(1000): s += i", number=2000)
builtin_time = timeit.timeit("sum(range(1000))", number=2000)
print(f"loop {loop_time:.3f}s vs sum() {builtin_time:.3f}s -> {loop_time / builtin_time:.1f}x faster")
