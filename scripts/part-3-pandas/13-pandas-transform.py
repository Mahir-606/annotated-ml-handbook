"""The Annotated ML Handbook, Chapter 13: Grouping, reshaping, joining and time series

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-3-pandas/13-pandas-transform.ipynb.
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
import pandas as pd

ins = pd.read_csv("insurance.csv")
print(ins.groupby("smoker")["charges"].mean().round(0))
print(ins.groupby(["region", "smoker"])["charges"].median().round(0).head(6))
print(ins.groupby("region").size())


# %%
summary = ins.groupby("smoker").agg(
    n=("charges", "size"),
    mean_charges=("charges", "mean"),
    sd_charges=("charges", "std"),
    median_bmi=("bmi", "median"),
    max_age=("age", "max"),
).round(1)
print(summary)
print(ins.groupby("sex")[["age", "bmi"]].agg(["mean", "min", "max"]).round(1))


# %%
ins["region_mean"] = ins.groupby("region")["charges"].transform("mean")
ins["vs_region"] = ins["charges"] / ins["region_mean"]
ins["rank_in_region"] = ins.groupby("region")["charges"].rank(ascending=False)
ins["bmi_z_by_sex"] = ins.groupby("sex")["bmi"].transform(lambda s: (s - s.mean()) / s.std())
print(ins[["region", "charges", "region_mean", "vs_region", "rank_in_region"]].head().round(2))
big = ins.groupby("children").filter(lambda g: len(g) > 100)
print(sorted(big["children"].unique()))


# %%
pt = ins.pivot_table(values="charges", index="region", columns="smoker",
                     aggfunc="mean", margins=True).round(0)
print(pt)
print(pd.crosstab(ins["region"], ins["smoker"]))
print(pd.crosstab(ins["sex"], ins["smoker"], normalize="index").round(3))


# %%
wide = pd.DataFrame({"subject": ["s1", "s2"],
                     "week1": [5.1, 6.0], "week2": [5.5, 6.2], "week3": [5.9, 6.1]})
long = wide.melt(id_vars="subject", var_name="week", value_name="score")
print(long)
back = long.pivot(index="subject", columns="week", values="score")
print(back)
print(back.stack().head(3))


# %%
a = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
b = pd.DataFrame({"id": [3], "x": [30]})
print(pd.concat([a, b], ignore_index=True))
print(pd.concat([a, pd.DataFrame({"y": [7, 8]})], axis=1))


# %%
patients = pd.DataFrame({"pid": [1, 2, 3, 4], "age": [34, 51, 29, 62]})
visits = pd.DataFrame({"pid": [1, 1, 2, 5], "bp": [120, 118, 135, 128]})
print(pd.merge(patients, visits, on="pid", how="inner"))
print(pd.merge(patients, visits, on="pid", how="left"))
print(pd.merge(patients, visits, on="pid", how="outer", indicator=True))


# %%
check = pd.merge(patients, visits, on="pid", how="left", validate="one_to_many")
print(len(patients), "->", len(check), "rows after merge")


# %%
rng = np.random.default_rng(1)
dates = pd.date_range("2025-01-01", periods=120, freq="D")
sales = pd.Series(100 + np.arange(120) * 0.8 + rng.normal(0, 8, 120)
                  + 15 * np.sin(np.arange(120) / 7 * 2 * np.pi), index=dates, name="sales")
print(sales.head(3))
print(sales.loc["2025-02"].shape)
monthly = sales.resample("ME").agg(["mean", "sum"]).round(1)
print(monthly)


# %%
import matplotlib.pyplot as plt
feat = sales.to_frame()
feat["roll7"] = sales.rolling(window=7).mean()
feat["lag1"] = sales.shift(1)
feat["diff1"] = sales.diff()
feat["pct"] = sales.pct_change()
print(feat.iloc[6:10].round(2))
ax = feat[["sales", "roll7"]].plot(figsize=(8, 3.4), title="Daily sales and 7-day rolling mean")
ax.set_ylabel("units")
plt.show()


# %%
report = (pd.read_csv("insurance.csv")
          .dropna(subset=["bmi"])
          .assign(obese=lambda d: d["bmi"] >= 30,
                  age_band=lambda d: pd.cut(d["age"], [17, 35, 50, 64]))
          .groupby(["age_band", "smoker"], observed=True)
          .agg(n=("charges", "size"), median_charges=("charges", "median"),
               obese_rate=("obese", "mean"))
          .round(2)
          .reset_index())
print(report)
