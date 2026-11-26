"""The Annotated ML Handbook, Chapter 18: Figures for papers, theses and assignments

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-4-visualization/18-viz-export.ipynb.
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
import matplotlib.pyplot as plt
from pathlib import Path
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
x = np.linspace(0, 10, 50)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(x, np.sin(x), label="model", lw=1.2)
ax.scatter(x, np.sin(x) + rng.normal(0, 0.2, 50), s=6, label="data", color="k")
ax.set(xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.legend(frameon=False)
plt.show()


# %%
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(x, np.cos(x), lw=1.2)
ax.set(xlabel="Time (s)", ylabel="Amplitude (mV)")
fig.savefig("fig_a.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            transparent=False)
fig.savefig("fig_a.pdf", bbox_inches="tight")
plt.close(fig)
from PIL import Image
with Image.open("fig_a.png") as im:
    w, h = im.size
print(f"PNG: {w} x {h} px  ->  {w / 300:.2f} x {h / 300:.2f} in at 300 dpi")


# %%
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.bar(["A", "B", "C"], [3, 5, 2])
ax.set_ylabel("Count")
fig.savefig("fig_fonts.pdf", bbox_inches="tight")
fig.savefig("fig_fonts.svg", bbox_inches="tight")
plt.close(fig)
print("PDF has Type 3 fonts:", b"/Type3" in Path("fig_fonts.pdf").read_bytes())


# %%
big = rng.normal(size=(100_000, 2))
for raster in (False, True):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.scatter(big[:, 0], big[:, 1], s=1, alpha=0.3, rasterized=raster)
    ax.set(xlabel="x", ylabel="y")
    name = f"scatter_raster_{raster}.pdf"
    fig.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(name, Path(name).stat().st_size // 1024, "KB")


# %%
okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), layout="constrained")
for i, c in enumerate(okabe_ito[:3]):
    axes[0].plot(x, np.sin(x + i), color=c, lw=1.2, label=f"group {i + 1}")
axes[0].set(xlabel="Time (s)", ylabel="Signal (a.u.)")
axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
axes[1].hist([rng.normal(m, 1, 300) for m in (0, 1.5)], bins=25, color=okabe_ito[3:5],
             label=["control", "treated"])
axes[1].set(xlabel="Score", ylabel="Count")
axes[1].legend(frameon=False)
for ax, letter in zip(axes, "ab"):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.savefig("figure2.pdf"); fig.savefig("figure2.png", dpi=300)
plt.show()


# %%
def save_figure(fig, name, folder="figures", formats=("pdf", "png"), dpi=300):
    """Save `fig` in several formats into `folder` and return the file paths."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(p)
    return paths

fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot([1, 2, 3], [2, 4, 3], marker="o")
for p in save_figure(fig, "fig3", formats=("pdf", "png", "svg", "tiff")):
    print(p, p.stat().st_size // 1024, "KB")
plt.close(fig)


# %%
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, c in enumerate(okabe_ito[:4]):
    ax.plot(x, x * (i + 1) / 4, color=c, lw=2, ls=["-", "--", ":", "-."][i], label=f"g{i + 1}")
ax.legend(frameon=False)
fig.savefig("colour_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
img = Image.open("colour_check.png")
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(img); axes[0].set_title("colour")
axes[1].imshow(img.convert("L"), cmap="gray"); axes[1].set_title("grayscale")
for a in axes: a.axis("off")
plt.show()
