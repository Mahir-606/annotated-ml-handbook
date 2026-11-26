# Figures for papers, theses and assignments

A chart that looks fine on screen can be blurry in print, unreadable at column width, or rejected by a journal's submission system. This chapter is a practical guide to getting figures out of Python as PNG, PDF, SVG or TIFF files, at the right size and resolution, and into Word, Google Docs, PowerPoint or LaTeX.

## Raster vs vector: which format? {B}

:::def
**Raster image**: a grid of coloured pixels (PNG, JPEG, TIFF). It has a fixed resolution and becomes blurry or blocky when enlarged. **Vector image**: a description of shapes, lines and text (PDF, SVG, EPS). It stays perfectly sharp at any size and its text remains selectable.

**DPI (dots per inch)**: how many pixels are printed per inch of paper. A 3.5-inch-wide figure at 300 dpi is 1,050 pixels wide.
:::

| Format | Type | Best for | Avoid for |
|---|---|---|---|
| **PDF** | vector | LaTeX papers and theses, journal submissions, printing | very dense scatter plots (huge files): rasterise the points (see 18.4) |
| **SVG** | vector | editing in Inkscape or Illustrator, Word 365, websites | some older journal systems |
| **EPS** | vector | journals that explicitly ask for it | transparency (EPS doesn't support it) |
| **PNG** | raster, lossless | Word, Google Docs, PowerPoint, websites, photos of plots | enlarging after export |
| **TIFF** | raster, lossless | journals that require TIFF for submission | everyday use (very large files) |
| **JPEG** | raster, lossy | photographs only | charts: compression blurs lines and text |

:::tip The two-file habit
Save every final figure twice: a **PDF** (vector, for LaTeX, printing and later editing) and a **300-dpi PNG** (for Word, slides and quick sharing). The function in section 18.6 does both in one call.
:::

## Size first, then resolution {B}

The most common mistake is making a figure at screen size and shrinking it into the paper, which makes the fonts tiny. Instead, **create the figure at its final printed size** and choose font sizes for that size.

| Target | Width | Typical font size |
|---|---|---|
| Journal, single column | 3.3–3.5 in (85–90 mm) | 7–9 pt |
| Journal, double column (full width) | 6.7–7.2 in (170–183 mm) | 7–9 pt |
| A4 thesis or assignment, full text width | about 6.3 in (160 mm) | 9–11 pt |
| Slide (16:9) | 10–13 in | 14–20 pt |
| Poster panel | 10–20 in | 20+ pt |

Always check your journal's author guidelines; the numbers above are common defaults, not rules.

```py title="Create a figure at its final size" caption="Designed at 3.5 x 2.6 inches with 8-pt fonts: exactly what a reader will see in a single journal column."
import numpy as np                                             #> Arrays.
import matplotlib.pyplot as plt                                #> Plotting.
from pathlib import Path                                       #> File checks.
rng = np.random.default_rng(0)                                 #> Seeded generator.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,   #> Font sizes for print.
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})   #> Ticks and legend slightly smaller.
x = np.linspace(0, 10, 50)                                     #> Data.
fig, ax = plt.subplots(figsize=(3.5, 2.6))                     #> Width x height in inches, the final printed size.
ax.plot(x, np.sin(x), label="model", lw=1.2)                   #> Thin lines suit small figures.
ax.scatter(x, np.sin(x) + rng.normal(0, 0.2, 50), s=6, label="data", color="k")   #> Small markers.
ax.set(xlabel="Time (s)", ylabel="Amplitude (mV)")             #> Axis labels with units.
ax.legend(frameon=False)                                       #> Legend without a box.
plt.show()                                                     #> Display.
```

## Saving with `savefig`: every option that matters {B}

```py title="The savefig options explained"
fig, ax = plt.subplots(figsize=(3.5, 2.6))                     #> A figure at final size.
ax.plot(x, np.cos(x), lw=1.2)                                  #> Any content.
ax.set(xlabel="Time (s)", ylabel="Amplitude (mV)")             #> Labels.
fig.savefig("fig_a.png",                                       #> The file name; the extension picks the format.
            dpi=300,                                           #> Resolution for raster formats (ignored by vector formats for lines and text).
            bbox_inches="tight",                               #> Crop away surrounding white space...
            pad_inches=0.02,                                   #> ...but keep a tiny margin.
            facecolor="white",                                 #> Background colour (avoids a transparent background in dark-mode viewers).
            transparent=False)                                 #> True gives a transparent background, useful for posters on coloured backgrounds.
fig.savefig("fig_a.pdf", bbox_inches="tight")                  #> Vector version.
plt.close(fig)                                                 #> Close it, since we only wanted the files.
from PIL import Image                                          #> Pillow reads images (installed with Matplotlib).
with Image.open("fig_a.png") as im:                            #> Open the PNG (the with-block closes the file afterwards).
    w, h = im.size                                             #> Its pixel dimensions.
print(f"PNG: {w} x {h} px  ->  {w / 300:.2f} x {h / 300:.2f} in at 300 dpi")   #> Pixels / dpi = printed inches (slightly under 3.5 because of the crop).
```

### Fonts that journals and Word accept {I}

```py title="Embed fonts as TrueType in PDF and keep text editable in SVG"
plt.rcParams["pdf.fonttype"] = 42                              #> Embed TrueType (Type 42) fonts: many journals reject Type 3 fonts.
plt.rcParams["ps.fonttype"] = 42                               #> The same for EPS/PS output.
plt.rcParams["svg.fonttype"] = "none"                          #> Keep SVG text as real text (editable) instead of converting it to paths.
plt.rcParams["font.family"] = "DejaVu Sans"                    #> A font that is always available; use "Arial" or "Times New Roman" if installed and required.
fig, ax = plt.subplots(figsize=(3.5, 2.4))                     #> New figure.
ax.bar(["A", "B", "C"], [3, 5, 2])                             #> Content.
ax.set_ylabel("Count")                                         #> Label.
fig.savefig("fig_fonts.pdf", bbox_inches="tight")              #> Fonts embedded as TrueType.
fig.savefig("fig_fonts.svg", bbox_inches="tight")              #> Text stays editable.
plt.close(fig)                                                 #> Close.
print("PDF has Type 3 fonts:", b"/Type3" in Path("fig_fonts.pdf").read_bytes())   #> Check: should print False.
```

:::warn Required fonts
If a journal requires Arial or Helvetica, set `plt.rcParams["font.family"] = "Arial"`. If Matplotlib can't find the font it falls back to another one and warns `findfont: Font family 'Arial' not found`. Install the font on your system, then delete Matplotlib's font cache (the folder shown by `matplotlib.get_cachedir()`) and restart Python.
:::

## Huge scatter plots: rasterise the points, keep the text sharp {I}

A vector PDF with 100,000 points stores every point separately, which gives huge, slow files. Rasterising only the data layer keeps axes and text as vectors.

```py title="rasterized=True"
big = rng.normal(size=(100_000, 2))                            #> 100,000 points.
for raster in (False, True):                                   #> Save twice to compare.
    fig, ax = plt.subplots(figsize=(3.5, 3))                   #> New figure.
    ax.scatter(big[:, 0], big[:, 1], s=1, alpha=0.3, rasterized=raster)   #> rasterized=True turns only this layer into pixels.
    ax.set(xlabel="x", ylabel="y")                             #> Labels stay vector text.
    name = f"scatter_raster_{raster}.pdf"                      #> File name records the setting.
    fig.savefig(name, dpi=300, bbox_inches="tight")            #> dpi sets the resolution of the rasterised layer.
    plt.close(fig)                                             #> Close.
    print(name, Path(name).stat().st_size // 1024, "KB")       #> Compare file sizes.
```

## Multi-panel figures with panel letters {I}

Journals usually want related plots combined into one figure with panels labelled **a**, **b**, **c**.

```py title="A publication-style two-panel figure" caption="A full-width (7-inch) figure with panels a and b, a colour-blind-safe palette, and consistent fonts."
okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]   #> A widely used colour-blind-safe palette.
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), layout="constrained")   #> Full width; constrained layout avoids overlaps.
for i, c in enumerate(okabe_ito[:3]):                          #> Three groups.
    axes[0].plot(x, np.sin(x + i), color=c, lw=1.2, label=f"group {i + 1}")   #> One line each.
axes[0].set(xlabel="Time (s)", ylabel="Signal (a.u.)")         #> a.u. = arbitrary units.
axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))   #> Legend in one row above the panel.
axes[1].hist([rng.normal(m, 1, 300) for m in (0, 1.5)], bins=25, color=okabe_ito[3:5],   #> Two histograms side by side.
             label=["control", "treated"])                     #> Labels.
axes[1].set(xlabel="Score", ylabel="Count")                    #> Labels.
axes[1].legend(frameon=False)                                  #> Legend.
for ax, letter in zip(axes, "ab"):                             #> Pair each panel with a letter.
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")   #> Placed in axes coordinates, top-left outside the plot.
fig.savefig("figure2.pdf"); fig.savefig("figure2.png", dpi=300)   #> Both formats.
plt.show()                                                     #> Display.
```

## One function to export everything {B}

```py title="save_figure(): consistent export for a whole project"
def save_figure(fig, name, folder="figures", formats=("pdf", "png"), dpi=300):
    """Save `fig` in several formats into `folder` and return the file paths."""
    out = Path(folder)                                         #> The output folder as a Path.
    out.mkdir(parents=True, exist_ok=True)                     #> Create it if needed.
    paths = []                                                 #> Collect the saved paths.
    for fmt in formats:                                        #> One file per format.
        p = out / f"{name}.{fmt}"                              #> e.g. figures/fig3.pdf
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")   #> The standard options.
        paths.append(p)                                        #> Remember it.
    return paths                                               #> Hand the list back.

fig, ax = plt.subplots(figsize=(3.5, 2.4))                     #> Any figure.
ax.plot([1, 2, 3], [2, 4, 3], marker="o")                      #> Content.
for p in save_figure(fig, "fig3", formats=("pdf", "png", "svg", "tiff")):   #> TIFF too, for journals that require it.
    print(p, p.stat().st_size // 1024, "KB")                   #> Report each file.
plt.close(fig)                                                 #> Close.
```

## Exporting Seaborn and Plotly figures {B}

| Library | Code |
|---|---|
| Matplotlib | `fig.savefig("f.pdf")`, `fig.savefig("f.png", dpi=300)` |
| Seaborn axes-level (`histplot`, `boxplot`, …) | draw with `ax=ax`, then `fig.savefig(...)` |
| Seaborn figure-level (`relplot`, `displot`, `catplot`, `pairplot`, `jointplot`) | `g = sns.relplot(...)`, then `g.savefig("f.pdf")` |
| Plotly | `fig.write_image("f.png", width=700, height=450, scale=3)`, `fig.write_image("f.pdf")`, `fig.write_html("f.html")` |
| pandas `.plot()` | `ax = df.plot(...)`, then `ax.figure.savefig("f.png", dpi=300)` |

For Plotly, the pixel size of a PNG is `width × scale`. For a 7-inch figure at 300 dpi you need 2,100 pixels: `width=700, scale=3`.

## Putting figures into documents {B}

| Destination | Use | How |
|---|---|---|
| **Microsoft Word** | PNG at 300 dpi (any version), or SVG (Word 365) | Insert → Pictures → This Device. Right-click → Size and Position: set the width and keep "Lock aspect ratio" ticked. Don't stretch a small image larger. |
| **Google Docs** | PNG at 300 dpi | Insert → Image → Upload from computer. |
| **PowerPoint / Keynote** | SVG (365) or high-resolution PNG | Insert → Pictures. Make the figure at slide size with large fonts. |
| **LaTeX / Overleaf** | PDF | Upload the PDF, then use the code below. |
| **Journal submission** | what the guidelines say (often PDF/EPS for plots, TIFF for images) | Name files `Figure1.pdf`, `Figure2.pdf`, and keep the code that made them. |

```text
% LaTeX: include a vector PDF figure at full text width
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/figure2.pdf}
  \caption{(a) Signal over time for three groups. (b) Score distributions;
           treated participants score higher.}
  \label{fig:results}
\end{figure}
```

## Writing a good figure caption {B}

A caption should let a reader understand the figure without reading the main text:

1. **Title sentence** stating what is shown, or better, the finding ("Treated participants scored higher than controls").
2. **What each panel shows** (a, b, c).
3. **What the marks mean**: error bars (SD, SEM or 95% CI?), shaded bands, dot sizes, colours.
4. **Sample sizes** (n per group) and, if relevant, the statistical test.

## Colour and accessibility {I}

```py title="Check a figure in grayscale" caption="Left: the colour figure. Right: the same figure converted to grayscale. If the groups are still distinguishable, the figure will survive black-and-white printing."
fig, ax = plt.subplots(figsize=(3.5, 2.6))                     #> A figure using the Okabe-Ito palette.
for i, c in enumerate(okabe_ito[:4]):                          #> Four lines.
    ax.plot(x, x * (i + 1) / 4, color=c, lw=2, ls=["-", "--", ":", "-."][i], label=f"g{i + 1}")   #> Colour AND line style: redundant encoding.
ax.legend(frameon=False)                                       #> Legend.
fig.savefig("colour_check.png", dpi=150, bbox_inches="tight")  #> Save it.
plt.close(fig)                                                 #> Close.
img = Image.open("colour_check.png")                           #> Reload the saved image.
fig, axes = plt.subplots(1, 2, figsize=(8, 3))                 #> Side-by-side comparison.
axes[0].imshow(img); axes[0].set_title("colour")               #> Original.
axes[1].imshow(img.convert("L"), cmap="gray"); axes[1].set_title("grayscale")   #> convert("L") = grayscale.
for a in axes: a.axis("off")                                   #> Hide axes around the pictures.
plt.show()                                                     #> Display.
```

## Final-figure checklist {B}

- [ ] Made at the final printed size, with fonts of at least 7 pt at that size.
- [ ] Axes labelled, with units; tick labels readable.
- [ ] Colour-blind-safe palette, plus line styles or markers so it still works in grayscale.
- [ ] Error bars and bands defined in the caption (SD, SEM or 95% CI), with n stated.
- [ ] Saved as PDF (vector) and 300-dpi PNG; TIFF or EPS if the journal asks.
- [ ] Fonts embedded (`pdf.fonttype = 42`); dense scatter layers rasterised.
- [ ] Panel letters on multi-panel figures; consistent styling across all figures.
- [ ] The code that generated it saved in your project, so it can be regenerated.

:::ex Try it yourself
1. Rebuild one figure from [[ch:viz-mpl-gallery]] at 3.5 inches wide with 8-pt fonts and save it with `save_figure()`.
2. Make a 7 x 3 inch, three-panel figure (a, b, c) from the insurance data and export it as PDF and PNG.
3. Open your PNG in an image viewer and confirm its pixel width equals inches × dpi.
:::
