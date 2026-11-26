"""The Annotated ML Handbook, Chapter 16: The Seaborn chart gallery

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-4-visualization/16-viz-seaborn-gallery.ipynb.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
sns.set_theme(style="whitegrid", palette="deep")
ins = pd.read_csv("insurance.csv")
iris = load_iris(as_frame=True).frame
iris["species"] = iris["target"].map(dict(enumerate(load_iris().target_names)))
print(ins.shape, iris.shape)


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(data=ins, x="bmi", bins=30, kde=True, ax=axes[0])
axes[0].set_title("histplot + KDE")
sns.kdeplot(data=ins, x="charges", hue="smoker", fill=True, common_norm=False, ax=axes[1])
axes[1].set_title("kdeplot by group")
plt.tight_layout()
plt.show()


# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
sns.ecdfplot(data=ins, x="age", hue="smoker", ax=ax)
sns.rugplot(data=ins.sample(150, random_state=0), x="age", hue="smoker", height=0.05, ax=ax, legend=False)
ax.set_title("ecdfplot + rugplot")
plt.show()


# %%
g = sns.displot(data=ins, x="charges", col="smoker", row="sex", bins=25,
                height=2.6, aspect=1.6, color="#0f9b62")
g.set_titles("{row_name} | smoker = {col_name}")
plt.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=ins, x="age", y="charges", hue="smoker", style="sex", size="children",
                sizes=(10, 80), alpha=0.6, ax=axes[0])
axes[0].set_title("scatterplot")
axes[0].legend(fontsize=7, loc="upper left", ncol=2)
sns.lineplot(data=ins, x="age", y="charges", hue="smoker", errorbar=("ci", 95), ax=axes[1])
axes[1].set_title("lineplot (mean ± 95% CI)")
plt.tight_layout()
plt.show()


# %%
g = sns.relplot(data=ins, x="bmi", y="charges", hue="smoker", col="region", col_wrap=2,
                height=2.6, aspect=1.4, s=12, alpha=0.6)
plt.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.regplot(data=iris, x="petal length (cm)", y="petal width (cm)", ax=axes[0],
            scatter_kws={"s": 14, "alpha": 0.6})
axes[0].set_title("regplot")
sns.residplot(data=iris, x="petal length (cm)", y="petal width (cm)", ax=axes[1],
              scatter_kws={"s": 14, "alpha": 0.6})
axes[1].set_title("residplot")
plt.tight_layout()
plt.show()


# %%
g = sns.lmplot(data=iris, x="sepal length (cm)", y="sepal width (cm)", hue="species",
               height=3.6, aspect=1.5, scatter_kws={"s": 14})
plt.show()


# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
kw = dict(data=iris, x="species", y="petal length (cm)")
sns.stripplot(**kw, ax=axes[0], size=3, alpha=0.7); axes[0].set_title("stripplot")
sns.swarmplot(**kw, ax=axes[1], size=2); axes[1].set_title("swarmplot")
sns.boxplot(**kw, ax=axes[2]); axes[2].set_title("boxplot")
sns.violinplot(**kw, ax=axes[3], inner="quart"); axes[3].set_title("violinplot")
sns.boxenplot(**kw, ax=axes[4]); axes[4].set_title("boxenplot")
for a in axes: a.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()


# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
sns.barplot(data=ins, x="region", y="charges", hue="smoker", ax=axes[0])
axes[0].set_title("barplot (mean ± 95% CI)"); axes[0].tick_params(axis="x", rotation=20)
sns.pointplot(data=ins, x="children", y="charges", hue="smoker", dodge=0.3, ax=axes[1])
axes[1].set_title("pointplot")
sns.countplot(data=ins, x="region", order=ins["region"].value_counts().index, ax=axes[2])
axes[2].set_title("countplot"); axes[2].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()


# %%
g = sns.catplot(data=ins, x="smoker", y="bmi", col="sex", kind="box",
                height=3.2, aspect=1.1)
plt.show()


# %%
corr = iris.drop(columns=["target", "species"]).corr()
fig, ax = plt.subplots(figsize=(6, 4.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("heatmap")
plt.show()


# %%
sub = iris.sample(60, random_state=1)
lut = dict(zip(sub["species"].unique(), ["#1f4fff", "#e0245e", "#0f9b62"]))
g = sns.clustermap(sub.drop(columns=["target", "species"]), standard_scale=1,
                   row_colors=sub["species"].map(lut), cmap="mako", figsize=(6, 6),
                   yticklabels=False)
plt.show()


# %%
g = sns.pairplot(iris.drop(columns="target"), hue="species", corner=True, height=1.9,
                 plot_kws={"s": 12, "alpha": 0.7})
plt.show()


# %%
for kind in ["scatter", "hex", "kde"]:
    g = sns.jointplot(data=ins, x="age", y="bmi", kind=kind, height=3.4,
                      color="#7b3ff2")
    g.figure.suptitle(f"jointplot(kind='{kind}')", y=1.02)
plt.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, ctx in zip(axes, ["paper", "talk"]):
    with sns.plotting_context(ctx):
        sns.boxplot(data=ins, x="region", y="bmi", hue="smoker", palette="colorblind", ax=ax)
        ax.set_title(f"context = '{ctx}'")
        ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
print(sns.color_palette("colorblind").as_hex()[:4])


# %%
fig, ax = plt.subplots(figsize=(4, 3))
sns.histplot(data=ins, x="bmi", ax=ax)
fig.savefig("bmi_hist.png", dpi=300, bbox_inches="tight")
plt.close(fig)
g = sns.relplot(data=ins, x="age", y="charges", hue="smoker", height=3)
g.savefig("age_charges.pdf")
plt.close(g.figure)
print("saved bmi_hist.png and age_charges.pdf")
