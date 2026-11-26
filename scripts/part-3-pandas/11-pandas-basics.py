"""The Annotated ML Handbook, Chapter 11: pandas foundations: Series, DataFrames and selection

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 11-pandas-basics.py
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
# ## 11.1 Series and DataFrame · *Beginner*
#
# **Building both by hand**


# %%
import numpy as np
import pandas as pd

s = pd.Series([4.2, 3.9, 5.1], index=["a", "b", "c"], name="yield")
print(s)
print(s["b"], s.mean())

df = pd.DataFrame({
    "plot": ["P1", "P2", "P3", "P4"],
    "fertilizer": ["A", "B", "A", "B"],
    "rain_mm": [120, 95, 140, 80],
    "yield_t": [4.2, 3.9, 5.1, 3.4],
})
print(df)
print(df.shape, list(df.columns), df.index)


# %% [markdown]
# ## 11.2 The practice dataset · *Beginner*
#
# **Create and save the practice dataset**


# %%
rng = np.random.default_rng(2026)
n = 1000
age = rng.integers(18, 65, n)
sex = rng.choice(["female", "male"], n)
bmi = np.round(rng.normal(30, 6, n).clip(16, 53), 1)
children = rng.choice([0, 1, 2, 3, 4], n, p=[.43, .24, .18, .12, .03])
smoker = rng.choice(["yes", "no"], n, p=[.2, .8])
region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)
charges = (2500 + 260 * age + 320 * (bmi - 30).clip(0) + 450 * children
           + np.where(smoker == "yes", 23000, 0)
           + rng.gamma(2.0, 1500, n)).round(2)
ins = pd.DataFrame({"age": age, "sex": sex, "bmi": bmi, "children": children,
                    "smoker": smoker, "region": region, "charges": charges})
ins.loc[rng.choice(n, 40, replace=False), "bmi"] = np.nan
ins.to_csv("insurance.csv", index=False)
print("saved", ins.shape)


# %% [markdown]
# ## 11.3 Reading and writing data · *Beginner*
#
# **read_csv and friends**


# %%
df = pd.read_csv("insurance.csv")
print(df.head())


# %% [markdown]
# ## 11.4 Inspecting a DataFrame · *Beginner*
#
# **First look**


# %%
print(df.shape)
df.info()


# %% [markdown]
# **Summary statistics**


# %%
print(df.describe().round(2))
print(df.describe(include="str"))


# %% [markdown]
# **Counts and missing values**


# %%
print(df["region"].value_counts())
print(df["smoker"].value_counts(normalize=True))
print(df.isna().sum())
print(df.nunique())


# %% [markdown]
# ## 11.5 Selecting data · *Beginner*
#
# ### 11.5.1 Columns
#
# **One column, several columns**


# %%
ages = df["age"]
print(type(ages).__name__, ages.shape)
sub = df[["age", "bmi", "charges"]]
print(type(sub).__name__, sub.shape)
print(df.age.mean())


# %% [markdown]
# ### 11.5.2 Rows: `.loc` (labels) and `.iloc` (positions)
#
# **loc vs iloc**


# %%
print(df.loc[0, "charges"])
print(df.loc[0:2, ["age", "smoker"]])
print(df.iloc[0:2, 0:3])
print(df.iloc[-1])


# %% [markdown]
# ### 11.5.3 Filtering with conditions
#
# **Boolean filtering**


# %%
smokers = df[df["smoker"] == "yes"]
print(len(smokers))
older_smokers = df[(df["age"] >= 50) & (df["smoker"] == "yes")]
print(len(older_smokers))
south = df[df["region"].isin(["southeast", "southwest"])]
print(south["region"].unique())
print(df.loc[df["bmi"] > 45, ["age", "bmi"]].head(3))
print(df.query("age < 25 and children >= 2").shape)
limit = 40
print(df.query("bmi > @limit").shape)


# %% [markdown]
# ## 11.6 Adding, changing and removing columns · *Beginner*
#
# **Creating new columns**


# %%
df = pd.read_csv("insurance.csv")
df["charges_k"] = df["charges"] / 1000
df["is_smoker"] = (df["smoker"] == "yes").astype(int)
df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 64],
                         labels=["18-30", "31-45", "46-64"])
df = df.assign(log_charges=np.log(df["charges"]))
df.loc[df["children"] >= 3, "big_family"] = True
df = df.rename(columns={"bmi": "BMI"})
df = df.drop(columns=["big_family"])
print(df.head(3))


# %% [markdown]
# ### 11.6.1 Sorting
#
# **sort_values and nlargest**


# %%
print(df.sort_values("charges", ascending=False).head(3)[["age", "smoker", "charges"]])
print(df.sort_values(["smoker", "age"], ascending=[True, False]).head(3)[["smoker", "age"]])
print(df.nlargest(3, "BMI")[["BMI", "age"]])


# %% [markdown]
# ## 11.7 Copy-on-Write and the "chained assignment" trap · *Intermediate*
#
# **This does NOT change df**


# %%
df = pd.read_csv("insurance.csv")
df[df["age"] > 60]["bmi"] = 0
print((df["bmi"] == 0).sum())


# %% [markdown]
# **The correct way: one .loc call**


# %%
df.loc[df["age"] > 60, "bmi"] = 0
print((df["bmi"] == 0).sum())
df = pd.read_csv("insurance.csv")
young = df[df["age"] < 30].copy()
young["bmi"] = young["bmi"].round()


# %% [markdown]
# ## 11.8 The index · *Intermediate*
#
# **Setting and resetting the index**


# %%
idx = df.set_index("region")
print(idx.loc["southwest"].shape)
print(idx.reset_index().columns.tolist()[:3])
r = df[df["age"] > 60]
print(r.index[:5].tolist())
print(r.reset_index(drop=True).index[:5].tolist())


# %% [markdown]
# ## 11.9 From pandas to NumPy and back · *Beginner*
#
# **Getting arrays out for modelling**


# %%
X = df[["age", "bmi", "children"]].to_numpy()
y = df["charges"].to_numpy()
print(X.shape, X.dtype, y[:3])
back = pd.DataFrame(X[:2], columns=["age", "bmi", "children"])
print(back)


# %% [markdown]
# ## Try it yourself
#
# 1. How many female smokers in the southeast are over 40?
# 2. Add a column `bmi_class` with "under" (< 18.5), "normal" (< 25), "over" (< 30), "obese" (≥ 30) using `pd.cut`.
# 3. Show the 5 youngest customers with 4 children, sorted by charges.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 11*.
