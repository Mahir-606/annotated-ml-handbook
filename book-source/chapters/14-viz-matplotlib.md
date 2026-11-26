@part 4

# Matplotlib and Seaborn: from first plot to publication figure

A good figure shows in seconds what a table hides. This chapter teaches Matplotlib's object model (the source of most plotting confusion), the essential chart types, Seaborn's statistical plots, and how to export figures that meet journal requirements.

:::note About the colours in this book
Figures here use a light house style (no top/right spines, a soft grid, the book's palette). Your default Matplotlib output will look slightly different but contain the same information. Section 14.6 shows how to set a style of your own.
:::

## The anatomy of a figure {B}

:::def
**Figure**: the whole drawing canvas in Matplotlib, which can contain one or more plots.

**Axes**: one individual plot inside a figure, with its own x-axis, y-axis, title and data. Most plotting methods are called on an Axes.

**Legend**: the key that tells readers which colour or marker belongs to which series.

**Colormap**: a mapping from numbers to colours. **Sequential** maps (viridis) go from low to high; **diverging** maps (RdBu) emphasise departures from a centre value.
:::


A **Figure** is the whole canvas. It contains one or more **Axes**: individual plots, each with its own x-axis, y-axis, title and data. Almost every method you call lives on an Axes.

```py title="The object-oriented interface (recommended)" caption="Every labelled element on this chart is set by one line of the code above it."
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
import matplotlib.pyplot as plt                   #> pyplot: the plotting interface, conventionally imported as plt.

x = np.linspace(0, 2 * np.pi, 200)  #> 200 points from 0 to 2π for smooth curves.
fig, ax = plt.subplots(figsize=(7, 3.8))          #> Create a Figure and one Axes. figsize is in inches (width, height).
ax.plot(x, np.sin(x), label="sin(x)", lw=2)       #> A line plot. `lw` = line width. `label` is used by the legend.
ax.plot(x, np.cos(x), label="cos(x)", ls="--")    #> `ls` = line style: "-", "--", ":", "-.".
ax.axhline(0, color="grey", lw=0.8)               #> A horizontal reference line across the whole axes.
ax.set_title("Sine and cosine")                   #> Axes title.
ax.set_xlabel("angle (radians)")                  #> Always label axes, with units.
ax.set_ylabel("value")  #> Label the y-axis.
ax.set_xlim(0, 2 * np.pi)                         #> Control the visible range.
ax.legend(loc="lower left")                       #> Show the labels; `loc` chooses the corner.
ax.annotate("peak", xy=(np.pi / 2, 1), xytext=(2.3, 0.8),  #> Text "peak" placed at (2.3, 0.8), pointing to the maximum at (π/2, 1)...
            arrowprops=dict(arrowstyle="->"))     #> Point an arrow at a data coordinate.
plt.show()                                        #> Render the figure (in scripts this opens a window).
```

:::tip `plt.plot()` vs `ax.plot()`
Tutorials often call `plt.plot(...)`, `plt.title(...)`: the *pyplot state machine*, which acts on "the current axes". It is fine for quick one-off plots. As soon as you have more than one panel, use `fig, ax = plt.subplots()` and call methods on `ax`; the setter names are `ax.set_title`, `ax.set_xlabel`, and so on.
:::

## The essential chart types {B}

```py title="Six workhorse plots in a grid" caption="One question per chart type: trend (line), relationship (scatter), comparison (bar), distribution (histogram), spread by group (box), matrix values (heatmap)."
rng = np.random.default_rng(0)  #> Seeded generator for the example data.
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))    #> A 2x3 grid; `axes` is a 2-D array of Axes objects.

t = np.arange(30)  #> 30 time steps.
axes[0, 0].plot(t, np.cumsum(rng.normal(0.5, 1, 30)), marker="o", ms=3)   #> LINE: change over an ordered variable. ms = marker size.
axes[0, 0].set_title("Line: trend over time")  #> Panel title (row 0, column 0).

a = rng.normal(size=150)  #> 150 random x values.
axes[0, 1].scatter(a, 2 * a + rng.normal(size=150), c=a, cmap="viridis", s=15, alpha=0.8)   #> SCATTER: two numeric variables; colour encodes a third.
axes[0, 1].set_title("Scatter: relationship")  #> Panel title.

cats, vals = ["A", "B", "C", "D"], [23, 45, 12, 36]  #> Category names and their values.
bars = axes[0, 2].bar(cats, vals, color="#0091b3")    #> BAR: compare a quantity across categories. Bars must start at zero.
axes[0, 2].bar_label(bars)                            #> Write each bar's value on top.
axes[0, 2].set_title("Bar: compare categories")  #> Panel title.

axes[1, 0].hist(rng.normal(0, 1, 1000), bins=30, edgecolor="white")   #> HISTOGRAM: the distribution of one variable. Try several bin counts.
axes[1, 0].set_title("Histogram: distribution")  #> Panel title.

groups = [rng.normal(m, 1, 80) for m in (0, 1, 0.5)]  #> Three groups of 80 values with different means.
axes[1, 1].boxplot(groups, tick_labels=["ctrl", "drug A", "drug B"])   #> BOX: median, quartiles, whiskers (1.5 IQR) and outliers per group.
axes[1, 1].set_title("Box: spread by group")  #> Panel title.

im = axes[1, 2].imshow(rng.random((6, 6)), cmap="magma")    #> HEATMAP: values of a matrix as colours.
fig.colorbar(im, ax=axes[1, 2])                            #> A colour bar is required to read a heatmap.
axes[1, 2].set_title("Heatmap: matrix")  #> Panel title.
axes[1, 2].grid(False)  #> Grid lines would cut across the heatmap cells.

fig.suptitle("Choosing a chart", fontsize=14, fontweight="bold")   #> A title for the whole figure.
fig.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

### Which chart for which question?

| Question | Chart | Matplotlib | Seaborn |
|---|---|---|---|
| How is one variable distributed? | histogram, KDE, box, violin | `ax.hist` | `sns.histplot`, `sns.kdeplot`, `sns.violinplot` |
| How do two numeric variables relate? | scatter (+ trend line) | `ax.scatter` | `sns.scatterplot`, `sns.regplot` |
| How does a value change over time? | line | `ax.plot` | `sns.lineplot` |
| How do groups compare? | bar (counts/means), box, strip | `ax.bar`, `ax.boxplot` | `sns.barplot`, `sns.boxplot`, `sns.stripplot` |
| How do many variables correlate? | correlation heatmap, pair plot | `ax.imshow` | `sns.heatmap`, `sns.pairplot` |
| How good is my model? | confusion matrix, ROC, residuals, learning curve | scikit-learn `*Display` classes | — |

:::warn Chart crimes to avoid
Truncated y-axes on bar charts; pie charts with many slices; 3-D effects; rainbow ("jet") colour maps for ordered values (use `viridis`, `cividis`, `magma`); red/green as the only distinction (about 8% of men are red-green colour-blind); and unlabelled axes.
:::

## Subplots, shared axes and layouts {I}

```py title="Shared axes and uneven layouts" caption="A 2x2 mosaic layout: one wide panel on top and two below that share a y-axis."
fig, axd = plt.subplot_mosaic([["top", "top"], ["left", "right"]],   #> Name each panel; repeating a name spans cells.
                              figsize=(9, 5.5), layout="constrained")   #> "constrained" layout avoids overlapping labels automatically.
days = pd.date_range("2025-01-01", periods=90)  #> 90 consecutive dates for the x-axis.
axd["top"].plot(days, np.cumsum(rng.normal(size=90)))  #> A random walk over time in the wide panel.
axd["top"].set_title("Wide panel: a time series")  #> Title.
axd["left"].hist(rng.normal(size=500), bins=25)  #> Histogram in the lower-left panel.
axd["left"].set_title("Left")  #> Title.
axd["right"].hist(rng.normal(1, 1.5, 500), bins=25, color="#e0245e")  #> A wider distribution in the lower-right panel.
axd["right"].sharey(axd["left"])                                      #> Share the y-axis with the left panel for fair comparison.
axd["right"].set_title("Right (same y-scale)")  #> Title.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Seaborn: statistical plots in one line {B}

Seaborn sits on top of Matplotlib, works directly with DataFrames, and handles grouping, colour and statistics for you.

```py title="Distribution and relationship by group" caption="Seaborn's hue argument splits any plot by a categorical variable. The smoker/non-smoker gap is the dominant pattern in this data."
import seaborn as sns  #> Import Seaborn (statistical plots built on Matplotlib) under the short name `sns`.

ins = pd.read_csv("insurance.csv")                     #> The practice dataset from Part III.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))  #> One row, two panels.
sns.histplot(data=ins, x="charges", hue="smoker", bins=40,  #> Histogram of charges, one colour per smoker group...
             element="step", stat="density", common_norm=False, ax=axes[0])   #> Density per group so groups of different size compare fairly.
axes[0].set_title("Charges by smoking status")  #> Title.
sns.scatterplot(data=ins, x="age", y="charges", hue="smoker",  #> Age vs charges, coloured by smoker...
                style="smoker", s=18, alpha=0.7, ax=axes[1])   #> `hue` -> colour, `style` -> marker shape (redundant encoding helps colour-blind readers).
axes[1].set_title("Charges vs age")  #> Title.
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))   #> Move the legend outside the plot so it hides no points.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="Comparing groups" caption="Box plots summarise; adding the raw points (strip plot) shows sample size and shape too."
fig, axes = plt.subplots(1, 2, figsize=(11, 4))  #> One row, two panels.
sns.boxplot(data=ins, x="region", y="charges", hue="smoker", ax=axes[0])     #> Grouped box plots.
axes[0].set_title("Charges by region and smoking")  #> Title.
axes[0].tick_params(axis="x", rotation=20)                                   #> Rotate long tick labels.
sns.move_legend(axes[0], "lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, title=None)   #> Legend above the panel, in one row.
sns.violinplot(data=ins, x="children", y="bmi", inner="quart", color="#cfe8f0", ax=axes[1])   #> Violin = mirrored density + quartile lines.
sns.stripplot(data=ins, x="children", y="bmi", size=2, alpha=0.4, color="k", ax=axes[1])     #> Overlay the raw points.
axes[1].set_title("BMI by number of children")  #> Title.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="Correlation heatmap" caption="Pearson correlations between numeric columns. A diverging colour map centred on zero separates positive from negative correlations."
num = ins.assign(smoker=ins["smoker"].eq("yes").astype(int)).select_dtypes("number")   #> Encode smoker as 0/1, keep numeric columns.
corr = num.corr()                                                    #> Pairwise Pearson correlation matrix.
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)                  #> Hide the redundant upper triangle.
fig, ax = plt.subplots(figsize=(6.2, 5))  #> A near-square figure suits a square matrix.
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",  #> Draw the matrix with numbers to 2 decimals, red = positive, blue = negative...
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, ax=ax)   #> annot writes the numbers in the cells.
ax.set_title("Correlation matrix")  #> Title.
ax.grid(False)                                                       #> No grid lines across the coloured cells (some styles add them).
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="Pair plot: every pair of variables at once" caption="A pair plot shows each variable's distribution on the diagonal and every pairwise scatter elsewhere. Invaluable at the start of a project."
g = sns.pairplot(ins.sample(300, random_state=0), vars=["age", "bmi", "charges"],  #> Pair plot of 300 random rows and three variables...
                 hue="smoker", corner=True, plot_kws={"s": 12, "alpha": 0.6}, height=2.3)   #> sample() keeps it fast; corner=True drops the duplicate upper half.
g.figure.suptitle("Pair plot", y=1.02)  #> A title above the whole grid (y > 1 places it just above).
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

```py title="Trend lines with uncertainty" caption="regplot fits a linear regression and shades its 95% confidence interval (bootstrapped)."
fig, ax = plt.subplots(figsize=(7, 4))  #> One panel.
for status, color in [("no", "#1f4fff"), ("yes", "#e0245e")]:  #> Loop over `[("no", "#1f4fff"), ("yes", "#e0245e")]`, naming each item `status, color`.
    sub = ins[ins["smoker"] == status]  #> Rows for this smoker group only.
    sns.regplot(data=sub, x="age", y="charges", scatter_kws={"s": 8, "alpha": 0.35},  #> Scatter plus fitted line; small, faint points...
                line_kws={"lw": 2}, color=color, label=f"smoker = {status}", ax=ax)   #> One fitted line per group.
ax.legend()  #> Show a legend built from each plotted series' `label`.
ax.set_title("Charges rise with age in both groups")  #> A title that states the finding.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Plotting directly from pandas {B}

```py title="DataFrame.plot for quick looks" caption="pandas' plot methods are thin wrappers around Matplotlib, so the result is a normal Axes you can customise."
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))  #> Two panels side by side.
ins.groupby("region")["charges"].median().sort_values().plot.barh(ax=axes[0], color="#0091b3")   #> Horizontal bars, sorted: easy to read.
axes[0].set(title="Median charges by region", xlabel="charges")  #> Title and x label.
ins["bmi"].plot.kde(ax=axes[1])                            #> Kernel density estimate: a smooth histogram (needs SciPy).
axes[1].set(title="BMI density", xlabel="BMI")  #> Title and x label.
plt.tight_layout()  #> Adjust spacing so titles and labels don't overlap.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

## Styles, colours and publication-quality export {I}

```py title="A reusable publication style" caption="A figure styled for a journal: small but legible fonts, colour-blind-safe palette, panel letters, no chart junk."
plt.rcParams.update({                     #> rcParams = global defaults for every figure in the session.
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,  #> Small fonts that stay legible at print size.
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,  #> Tick and legend text sizes.
    "axes.spines.top": False, "axes.spines.right": False,  #> Remove the top and right box lines.
    "savefig.dpi": 300, "savefig.bbox": "tight",  #> Saved files: 300 dpi, trimmed to the content.
})  #> Close the dictionary and the call or tuple opened above.
okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]   #> The Okabe-Ito colour-blind-safe palette.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))       #> ~7 inches = a typical two-column journal figure width.
for i, (name, grp) in enumerate(ins.groupby("region")):  #> Loop over `enumerate(ins.groupby("region"))`, naming each item `i, (name, grp)`.
    axes[0].scatter(grp["age"], grp["charges"] / 1000, s=5, alpha=0.6, color=okabe_ito[i], label=name)  #> One colour-blind-safe colour per region; charges shown in thousands.
axes[0].set(xlabel="Age (years)", ylabel="Charges (thousand USD)")  #> Axis labels with units.
axes[0].set_ylim(5, 64)                          #> Head-room at the top for the legend.
axes[0].legend(markerscale=2, frameon=False, ncol=2, loc="upper left")  #> Bigger legend markers, no frame, two columns, top-left.
axes[1].hist([ins.loc[ins.smoker == s, "charges"] / 1000 for s in ("no", "yes")],  #> Two histograms (non-smokers, smokers)...
             bins=30, stacked=True, color=okabe_ito[3:5], label=["non-smoker", "smoker"])  #> ...stacked on top of each other, in two palette colours.
axes[1].set(xlabel="Charges (thousand USD)", ylabel="Count")  #> Axis labels.
axes[1].legend(frameon=False)  #> Legend without a box.
for ax, letter in zip(axes, "ab"):  #> Loop over `zip(axes, "ab")`, naming each item `ax, letter`.
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")   #> Panel letters in axes coordinates (0-1).
fig.savefig("figure1.pdf")                       #> Vector PDF: infinitely scalable, preferred by journals.
fig.savefig("figure1.png", dpi=300)              #> High-resolution raster for slides or the web.
fig.savefig("figure1.svg")                       #> SVG: editable in Inkscape or Illustrator.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
plt.rcdefaults()                                 #> Restore Matplotlib's defaults afterwards.
```

:::research Figure checklist for papers
Vector format (PDF/EPS/SVG) for plots, at least 300 dpi for raster images; fonts readable at final print size (≥ 7 pt); axes labelled with units; a colour-blind-safe palette with redundant encoding (markers or line styles); error bars defined in the caption (SD, SEM or 95% CI?); the code that generated each figure kept in your repository.
:::

:::bug My saved figure is blank or cut off
Call `fig.savefig(...)` **before** `plt.show()`: in scripts, `show()` can close the figure. Clipped labels: use `bbox_inches="tight"` or `layout="constrained"`. Tiny text: set `figsize` to the final printed size instead of shrinking a big figure later.
:::

:::ex Try it yourself
1. Plot the histogram of `charges` and of `np.log(charges)` side by side. Which looks closer to normal?
2. Recreate the pair plot with `hue="region"`. Does region matter?
3. Make a bar chart of mean charges by number of children with error bars (`sns.barplot` shows a 95% CI by default).
:::
