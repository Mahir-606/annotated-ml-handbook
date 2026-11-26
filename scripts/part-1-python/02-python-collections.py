"""The Annotated ML Handbook, Chapter 2: Data structures: lists, tuples, dictionaries, sets

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 02-python-collections.py
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
# ## 2.1 Lists · *Beginner*
#
# **Creating, indexing and modifying a list**


# %%
scores = [88, 92, 79, 95]
print(scores[0], scores[-1])
scores[2] = 81
scores.append(70)
scores.extend([99, 60])
scores.insert(1, 100)
print(scores)
last = scores.pop()
scores.remove(100)
print(last, scores)
print(len(scores), sum(scores), max(scores), 95 in scores)


# %% [markdown]
# ### 2.1.1 Slicing · *Beginner*
#
# **Slices**


# %%
nums = list(range(10))
print(nums[2:5])
print(nums[:3], nums[-3:])
print(nums[::2])
print(nums[::-1])
nums[1:3] = [10, 20]
print(nums)


# %% [markdown]
# ### 2.1.2 Sorting · *Beginner*
#
# **sorted() vs .sort()**


# %%
names = ["mia", "Ola", "zed", "ben"]
print(sorted(names))
print(sorted(names, key=str.lower))
print(sorted(names, key=len, reverse=True))
names.sort()
print(names)


# %% [markdown]
# ## 2.2 Tuples · *Beginner*
#
# **Tuples and unpacking**


# %%
point = (3.0, 4.0)
x, y = point
print(x, y, len(point))
single = (5,)
print(type(single), type((5)))
a, b = 1, 2
a, b = b, a
print(a, b)
first, *rest = [10, 20, 30, 40]
print(first, rest)


# %% [markdown]
# **Tuples cannot be changed**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
shape = (28, 28)
shape[0] = 32
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ## 2.3 Dictionaries · *Beginner*
#
# **Building and querying a dictionary**


# %%
params = {"model": "svm", "C": 1.0, "kernel": "rbf"}
print(params["C"])
params["gamma"] = 0.1
params["C"] = 10.0
print(params.get("degree"))
print(params.get("degree", 3))
print("kernel" in params)
del params["gamma"]
for key, value in params.items():
    print(f"{key:>7} -> {value}")
print(list(params.keys()), list(params.values()))


# %% [markdown]
# **The KeyError**
#
# ⚠️ *This cell raises an error **on purpose**, so you can see the real message. The book explains the cause and the fix.*


# %%
# This example raises an error ON PURPOSE (the book explains the cause and the fix).
# It runs inside exec() so the script prints the error and carries on.
_example = r'''
params = {"model": "svm"}
print(params["learning_rate"])
'''
try:
    exec(_example)
except Exception as _e:
    print(f"[expected error] {type(_e).__name__}: {_e}")


# %% [markdown]
# ### 2.3.1 Nested structures · *Intermediate*
#
# **A nested record**


# %%
experiment = {
    "name": "baseline",
    "metrics": {"acc": 0.91, "f1": 0.88},
    "seeds": [0, 1, 2],
}
print(experiment["metrics"]["f1"])
experiment["seeds"].append(3)
print(experiment["seeds"])
merged = {**experiment["metrics"], "auc": 0.95}
print(merged)
print(experiment["metrics"] | {"acc": 0.93})


# %% [markdown]
# ## 2.4 Sets · *Beginner*
#
# **Uniqueness and set algebra**


# %%
labels = ["cat", "dog", "cat", "bird", "dog"]
unique = set(labels)
print(unique, len(unique))
a = {1, 2, 3, 4}
b = {3, 4, 5}
print(a | b, a & b, a - b, a ^ b)
print(3 in a)
train_ids, test_ids = {1, 2, 3}, {3, 4}
print("Overlap:", train_ids & test_ids)


# %% [markdown]
# ## 2.5 Comprehensions · *Beginner*
#
# **List, dict and set comprehensions**


# %%
nums = [1, 2, 3, 4, 5, 6]
squares = [n ** 2 for n in nums]
evens = [n for n in nums if n % 2 == 0]
labels = ["even" if n % 2 == 0 else "odd" for n in nums]
print(squares, evens, labels, sep="\n")
word_len = {w: len(w) for w in ["numpy", "pandas", "keras"]}
print(word_len)
first_letters = {w[0] for w in ["apple", "avocado", "banana"]}
print(first_letters)
pairs = [(i, j) for i in range(3) for j in range(2)]
print(pairs)


# %% [markdown]
# ## 2.6 Mutability, references and copies · *Intermediate*
#
# **Two names, one list**


# %%
a = [1, 2, 3]
b = a
b.append(4)
print(a)
print(a is b)
c = a.copy()
c.append(5)
print(a, c, a is c)


# %% [markdown]
# **Shallow vs deep copies of nested data**


# %%
import copy
grid = [[0, 0], [0, 0]]
shallow = grid.copy()
deep = copy.deepcopy(grid)
grid[0][0] = 99
print(shallow[0][0], deep[0][0])
bad = [[0] * 2] * 2
bad[0][0] = 1
print(bad)
good = [[0] * 2 for _ in range(2)]
good[0][0] = 1
print(good)


# %% [markdown]
# ## Try it yourself
#
# 1. From `words = ["Data", "science", "IS", "fun", "data"]`, build the set of lowercase unique words.
# 2. Build a dict mapping each number 1–10 to `"fizz"` if divisible by 3, else the number itself.
# 3. Explain why `t = ([1, 2], 3); t[0].append(9)` works even though tuples are immutable.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 2*.
