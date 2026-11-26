"""The Annotated ML Handbook, Chapter 12: Cleaning data with pandas

Companion script by Mahir Absar Khan. Cells are marked with "# %%" so you can run them one
at a time in VS Code, PyCharm or Spyder, or run the whole file: python 12-pandas-cleaning.py
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
# ## 12.1 A deliberately messy dataset · *Beginner*
#
# **Read messy CSV text directly**


# %%
import io
import numpy as np
import pandas as pd

raw = """id,Name ,age,income,city,signup_date,score
1, Alice ,34,"52,000",new york,2025-01-15,88
2,bob,,61000,New York,2025/02/03,92
3,Carol,29,?,Boston,15-03-2025,
4,dave,41,48000,boston ,2025-04-22,75
4,dave,41,48000,boston ,2025-04-22,75
5,Eve,230,70000,Chicago,2025-05-30,81
6,frank,38,,CHICAGO,2025-06-11,-1
"""
df = pd.read_csv(io.StringIO(raw), na_values=["?"])
print(df)
print(df.dtypes)


# %% [markdown]
# ## 12.2 Step 1: tidy column names · *Beginner*
#
# **Normalise headers**


# %%
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(df.columns.tolist())


# %% [markdown]
# ## 12.3 Step 2: duplicates · *Beginner*
#
# **Find and drop duplicate rows**


# %%
print(df.duplicated().sum())
print(df[df.duplicated(keep=False)])
df = df.drop_duplicates().reset_index(drop=True)
print(df["id"].is_unique, len(df))


# %% [markdown]
# ## 12.4 Step 3: fix types · *Beginner*
#
# **Text to numbers**


# %%
df["income"] = pd.to_numeric(df["income"].str.replace(",", ""), errors="coerce")
print(df["income"].tolist(), df["income"].dtype)
df["score"] = df["score"].replace(-1, np.nan)
print(df["score"].tolist())


# %% [markdown]
# **Mixed date formats: the tempting shortcut is wrong**


# %%
guess = pd.to_datetime(df["signup_date"], format="mixed", dayfirst=True)
print(guess.dt.strftime("%Y-%m-%d").tolist())


# %% [markdown]
# **Parse each known format explicitly**


# %%
d = df["signup_date"]
parsed = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
parsed = parsed.fillna(pd.to_datetime(d, format="%Y/%m/%d", errors="coerce"))
parsed = parsed.fillna(pd.to_datetime(d, format="%d-%m-%Y", errors="coerce"))
print(parsed.isna().sum(), "unparsed")
df["signup_date"] = parsed
print(df["signup_date"].dt.strftime("%Y-%m-%d").tolist())
print(df["signup_date"].dt.month.tolist(), df["signup_date"].dt.day_name().tolist()[:3])


# %% [markdown]
# ## 12.5 Step 4: clean text and categories · *Beginner*
#
# **The .str accessor**


# %%
df["name"] = df["name"].str.strip().str.title()
df["city"] = df["city"].str.strip().str.title()
print(df[["name", "city"]])
print(df["city"].value_counts())
print(df["name"].str.contains("a", case=False).tolist())
print(df["name"].str.len().tolist(), df["name"].str[0].tolist())


# %% [markdown]
# **Recoding values**


# %%
s = pd.Series(["M", "male", "F", "female", "f", "unknown"])
mapping = {"m": "male", "male": "male", "f": "female", "female": "female"}
print(s.str.lower().map(mapping).tolist())
print(s.replace({"M": "male", "F": "female"}).tolist())


# %% [markdown]
# ## 12.6 Step 5: missing values · *Beginner*
#
# **Measure missingness**


# %%
print(df.isna().sum())
print(df.isna().mean().round(2))
print(df[df.isna().any(axis=1)])


# %% [markdown]
# **Drop or fill**


# %%
print(df.dropna().shape)
print(df.dropna(subset=["age"]).shape)
filled = df.copy()
filled["age"] = filled["age"].fillna(filled["age"].median())
filled["income"] = filled["income"].fillna(filled.groupby("city")["income"].transform("median"))
filled["score_missing"] = filled["score"].isna().astype(int)
filled["score"] = filled["score"].fillna(filled["score"].mean())
print(filled[["age", "income", "score", "score_missing"]])


# %% [markdown]
# **Ordered data: forward fill and interpolation**


# %%
ts = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
print(ts.ffill().tolist())
print(ts.interpolate().tolist())


# %% [markdown]
# ## 12.7 Step 6: outliers and impossible values · *Intermediate*
#
# **Domain rules and the IQR rule**


# %%
print(df[(df["age"] < 0) | (df["age"] > 110)])
df.loc[df["age"] > 110, "age"] = np.nan
x = pd.Series([10, 12, 11, 13, 12, 95, 11, 10])
q1, q3 = x.quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"bounds=({lo}, {hi}) outliers={x[(x < lo) | (x > hi)].tolist()}")
print(x.clip(lower=lo, upper=hi).tolist())


# %% [markdown]
# ## 12.8 Transforming columns with map, apply and vectorized ops · *Intermediate*
#
# **Fastest to slowest**


# %%
ins = pd.read_csv("insurance.csv")
ins["bmi_sq"] = ins["bmi"] ** 2
ins["obese"] = np.where(ins["bmi"] >= 30, "yes", "no")
ins["risk"] = np.select(
    [ins["smoker"].eq("yes") & (ins["bmi"] >= 30), ins["smoker"].eq("yes")],
    ["high", "medium"], default="low")
ins["region_code"] = ins["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})
ins["desc"] = ins.apply(lambda r: f"{r['age']}y {r['sex']}", axis=1)
print(ins[["bmi", "obese", "smoker", "risk", "region_code", "desc"]].head())
print(ins["risk"].value_counts())


# %% [markdown]
# ## 12.9 Categorical dtype · *Intermediate*
#
# **Memory-efficient, ordered categories**


# %%
ins["region"] = ins["region"].astype("category")
print(ins["region"].cat.categories.tolist(), ins["region"].cat.codes[:5].tolist())
size = pd.Categorical(["M", "S", "L", "M"], categories=["S", "M", "L"], ordered=True)
print(size.sort_values().tolist(), (size > "S").tolist())
dummies = pd.get_dummies(ins["region"], prefix="reg", dtype=int)
print(dummies.head(3))


# %% [markdown]
# ## 12.10 A reusable cleaning function · *Intermediate*
#
# **clean_customers()**


# %%
def parse_dates(s, formats=("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y")):
    out = pd.Series(pd.NaT, index=s.index)
    for fmt in formats:
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out

def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw customer table."""
    return (raw
            .rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
            .drop_duplicates()
            .assign(
                name=lambda d: d["name"].str.strip().str.title(),
                city=lambda d: d["city"].str.strip().str.title(),
                income=lambda d: pd.to_numeric(d["income"].astype("str").str.replace(",", ""), errors="coerce"),
                age=lambda d: d["age"].where(d["age"].between(0, 110)),
                score=lambda d: d["score"].replace(-1, np.nan),
                signup_date=lambda d: parse_dates(d["signup_date"]),
            )
            .reset_index(drop=True))

clean = clean_customers(pd.read_csv(io.StringIO(raw), na_values=["?"]))
print(clean)
print(clean.dtypes)


# %% [markdown]
# ## Try it yourself
#
# 1. In `insurance.csv`, fill missing BMI with the median BMI of the same sex and smoker group (`groupby(...).transform("median")`).
# 2. Count how many charges are outliers by the IQR rule. Are they mostly smokers?
# 3. Write a check function that raises an error if any age is outside 18–64 or any charge is negative.
#
# Worked solutions: [solutions-parts-1-4.ipynb](../../solutions/solutions-parts-1-4.ipynb), section *Chapter 12*.
