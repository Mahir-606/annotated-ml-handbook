# The Seaborn chart gallery

Seaborn is a statistical plotting library built on Matplotlib. It works directly with tidy DataFrames (one row per observation, one column per variable), handles grouping and colour automatically, and computes statistics such as densities, confidence intervals and regression fits for you. This chapter catalogues its charts, with what each one shows and when to use it.

:::def
**Tidy (long) data**: a table where every row is one observation and every column is one variable. Seaborn, Plotly Express and most statistical tools expect this shape.

**Axes-level function**: a Seaborn function that draws onto one Matplotlib Axes and accepts `ax=` (for example `histplot`, `boxplot`). **Figure-level function**: one that creates its own figure and can split it into a grid of panels with `col=` and `row=` (for example `displot`, `relplot`, `catplot`).
:::

## Set-up {B}

```py title="Imports, theme and data"
import numpy as np                                             #> Arrays.
import pandas as pd                                            #> Tables.
import matplotlib.pyplot as plt                                #> Seaborn draws on Matplotlib figures.
import seaborn as sns                                          #> The statistical plotting library.
from sklearn.datasets import load_iris                         #> A small built-in dataset (no download needed).
sns.set_theme(style="whitegrid", palette="deep")               #> One call sets a clean style and colour palette for every plot.
ins = pd.read_csv("insurance.csv")                             #> The Part III practice dataset.
iris = load_iris(as_frame=True).frame                          #> Iris as one DataFrame.
iris["species"] = iris["target"].map(dict(enumerate(load_iris().target_names)))   #> Readable species names instead of 0/1/2.
print(ins.shape, iris.shape)                                   #> Check both loaded.
```

| Family | Axes-level functions | Figure-level wrapper |
|---|---|---|
| Distributions | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` | `displot` |
| Relationships | `scatterplot`, `lineplot` | `relplot` |
| Categories | `stripplot`, `swarmplot`, `boxplot`, `violinplot`, `boxenplot`, `pointplot`, `barplot`, `countplot` | `catplot` |
| Regression | `regplot`, `residplot` | `lmplot` |
| Matrices | `heatmap` | `clustermap` |
| Multi-variable | — | `pairplot`, `jointplot` |

## Distribution plots {B}

### histplot and kdeplot

:::def
**KDE (kernel density estimate)**: a smooth curve estimating the probability density of a variable, made by placing a small bump on every data point and adding them up. Think of it as a smoothed histogram.
:::

**Use a histogram** to see the actual counts; **use a KDE** to compare the shapes of several groups smoothly. A KDE can show density where no data exist (beyond the data's range), and its smoothness depends on the bandwidth (`bw_adjust`).

```py title="Histogram with KDE, and overlaid KDEs" caption="Left: histogram of BMI with a KDE curve. Right: KDEs of charges for smokers and non-smokers, each normalised separately."
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))                                    #> Two panels.
sns.histplot(data=ins, x="bmi", bins=30, kde=True, ax=axes[0])                       #> kde=True adds the smooth curve over the bars.
axes[0].set_title("histplot + KDE")                                                  #> Title.
sns.kdeplot(data=ins, x="charges", hue="smoker", fill=True, common_norm=False, ax=axes[1])   #> common_norm=False: each group's curve has area 1.
axes[1].set_title("kdeplot by group")                                                #> Title.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
```

### ecdfplot and rugplot

:::def
**Rug plot**: a small tick on the axis for every single observation, showing exactly where the data lie.
:::

```py title="ECDF with a rug" caption="ECDFs of age by smoker status; the rug ticks along the bottom mark each individual observation."
fig, ax = plt.subplots(figsize=(7, 3.6))                                             #> One panel.
sns.ecdfplot(data=ins, x="age", hue="smoker", ax=ax)                                 #> Cumulative proportion for each group.
sns.rugplot(data=ins.sample(150, random_state=0), x="age", hue="smoker", height=0.05, ax=ax, legend=False)   #> A rug of 150 sampled points.
ax.set_title("ecdfplot + rugplot")                                                   #> Title.
plt.show()                                                                           #> Draw.
```

### displot: small multiples of distributions

```py title="One histogram per region and smoker status" caption="displot splits the data into a grid of panels (columns = smoker, rows = sex): an easy way to compare many subgroups."
g = sns.displot(data=ins, x="charges", col="smoker", row="sex", bins=25,             #> Figure-level: col and row define the grid.
                height=2.6, aspect=1.6, color="#0f9b62")                             #> Each panel 2.6 inches tall, 1.6 times as wide.
g.set_titles("{row_name} | smoker = {col_name}")                                     #> Panel titles from the grid variables.
plt.show()                                                                           #> Draw.
```

## Relationship plots {B}

### scatterplot and lineplot

:::def
**Confidence interval (in a line plot)**: Seaborn's `lineplot` aggregates repeated observations at each x into a mean and shades a 95% confidence interval for that mean, computed by bootstrapping.
:::

```py title="Scatter with four encodings; line with a confidence band" caption="Left: age vs charges with colour (smoker), marker style (sex) and size (children). Right: mean charges by age with a 95% CI band."
fig, axes = plt.subplots(1, 2, figsize=(12, 4))                                      #> Two panels.
sns.scatterplot(data=ins, x="age", y="charges", hue="smoker", style="sex", size="children",   #> Four variables in one chart.
                sizes=(10, 80), alpha=0.6, ax=axes[0])                               #> Range of marker sizes; transparency.
axes[0].set_title("scatterplot")                                                     #> Title.
axes[0].legend(fontsize=7, loc="upper left", ncol=2)                                 #> Compact legend.
sns.lineplot(data=ins, x="age", y="charges", hue="smoker", errorbar=("ci", 95), ax=axes[1])   #> Mean per age with a 95% CI band.
axes[1].set_title("lineplot (mean ± 95% CI)")                                        #> Title.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
```

### relplot: faceted relationships

```py title="Relationship split by region" caption="relplot draws one scatter panel per region, wrapped into two columns, with consistent axes for fair comparison."
g = sns.relplot(data=ins, x="bmi", y="charges", hue="smoker", col="region", col_wrap=2,   #> One panel per region, 2 per row.
                height=2.6, aspect=1.4, s=12, alpha=0.6)                             #> Panel size, marker size, transparency.
plt.show()                                                                           #> Draw.
```

## Regression plots {I}

### regplot, residplot and lmplot

:::def
**Residual**: the difference between an observed value and the value a model predicts. **Residual plot**: residuals against the predictor; a shapeless cloud around zero means the model fits well, a curve or funnel means it is missing something.
:::

```py title="A fit and its residuals" caption="Left: linear fit of petal width on petal length with a 95% CI. Right: the residuals; a curved pattern would signal a non-linear relationship."
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))                                    #> Two panels.
sns.regplot(data=iris, x="petal length (cm)", y="petal width (cm)", ax=axes[0],       #> Scatter + least-squares line + CI band.
            scatter_kws={"s": 14, "alpha": 0.6})                                     #> Smaller, transparent points.
axes[0].set_title("regplot")                                                         #> Title.
sns.residplot(data=iris, x="petal length (cm)", y="petal width (cm)", ax=axes[1],     #> Residuals of the same linear fit.
              scatter_kws={"s": 14, "alpha": 0.6})                                   #> Same styling.
axes[1].set_title("residplot")                                                       #> Title.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
```

```py title="lmplot: one regression per group" caption="lmplot fits a separate regression line for each species: the relationship differs between groups."
g = sns.lmplot(data=iris, x="sepal length (cm)", y="sepal width (cm)", hue="species",  #> Figure-level: one fit per hue level.
               height=3.6, aspect=1.5, scatter_kws={"s": 14})                        #> Size and marker styling.
plt.show()                                                                           #> Draw.
```

## Categorical plots {B}

### strip, swarm, box, violin and boxen

:::def
**Strip plot**: every observation drawn as a dot along a category axis, with random jitter so dots don't overlap. **Swarm plot**: like a strip plot, but dots are arranged so none overlap, which shows the distribution's shape. **Boxen (letter-value) plot**: an extended box plot that shows more quantiles; useful for large datasets.
:::

```py title="Five ways to show a numeric variable across categories" caption="The same data (petal length by species) as a strip, swarm, box, violin and boxen plot."
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)                       #> Five panels sharing the y-axis.
kw = dict(data=iris, x="species", y="petal length (cm)")                             #> Shared arguments, unpacked with **kw below.
sns.stripplot(**kw, ax=axes[0], size=3, alpha=0.7); axes[0].set_title("stripplot")   #> Jittered dots.
sns.swarmplot(**kw, ax=axes[1], size=2); axes[1].set_title("swarmplot")              #> Non-overlapping dots (small markers so all 150 fit).
sns.boxplot(**kw, ax=axes[2]); axes[2].set_title("boxplot")                          #> Quartiles and whiskers.
sns.violinplot(**kw, ax=axes[3], inner="quart"); axes[3].set_title("violinplot")     #> Density with quartile lines.
sns.boxenplot(**kw, ax=axes[4]); axes[4].set_title("boxenplot")                      #> Letter-value plot.
for a in axes: a.tick_params(axis="x", rotation=25)                                  #> Tilt the species names.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
```

### barplot, pointplot and countplot

:::def
**Bar plot (Seaborn)**: the height of each bar is a summary statistic (by default the mean) of a numeric variable for that category, with an error bar (by default a 95% confidence interval). **Point plot**: the same estimate drawn as a point joined by lines. **Count plot**: bars showing how many rows fall in each category.
:::

**Use pointplot** to compare how a mean changes across levels of two factors: non-parallel lines reveal an interaction.

```py title="Means, interactions and counts" caption="Left: mean charges with 95% CIs. Middle: point plot; the two lines run parallel, so the smoking gap stays about the same whatever the number of children (no interaction). Right: counts per region."
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))                                    #> Three panels.
sns.barplot(data=ins, x="region", y="charges", hue="smoker", ax=axes[0])             #> Mean per group with 95% CI error bars.
axes[0].set_title("barplot (mean ± 95% CI)"); axes[0].tick_params(axis="x", rotation=20)   #> Title; tilt labels.
sns.pointplot(data=ins, x="children", y="charges", hue="smoker", dodge=0.3, ax=axes[1])   #> dodge separates the two groups slightly.
axes[1].set_title("pointplot")                                                       #> Title.
sns.countplot(data=ins, x="region", order=ins["region"].value_counts().index, ax=axes[2])   #> Bars ordered from most to least frequent.
axes[2].set_title("countplot"); axes[2].tick_params(axis="x", rotation=20)           #> Title; tilt labels.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
```

### catplot: faceted categorical plots

```py title="Any categorical plot as a grid" caption="catplot with kind='box' draws one box-plot panel per sex. Change kind to 'violin', 'strip', 'bar' and so on."
g = sns.catplot(data=ins, x="smoker", y="bmi", col="sex", kind="box",                 #> kind selects the plot type.
                height=3.2, aspect=1.1)                                              #> Panel size.
plt.show()                                                                           #> Draw.
```

## Matrix plots {I}

### heatmap and clustermap

:::def
**Clustermap**: a heatmap whose rows and columns are reordered by hierarchical clustering, with dendrograms (tree diagrams) showing which rows and columns are most similar.
:::

```py title="Correlation heatmap" caption="Pearson correlations among the iris measurements, annotated, on a diverging colour scale centred at 0."
corr = iris.drop(columns=["target", "species"]).corr()                               #> 4x4 correlation matrix.
fig, ax = plt.subplots(figsize=(6, 4.6))                                             #> One panel.
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,   #> Numbers in cells, fixed -1..1 scale...
            square=True, linewidths=0.5, ax=ax)                                      #> ...square cells with thin gaps.
ax.set_title("heatmap")                                                              #> Title.
plt.show()                                                                           #> Draw.
```

```py title="Clustermap groups similar rows and columns" caption="Rows (flowers) and columns (measurements) reordered so similar ones sit together; the row colours show species. The clusters largely line up with species."
sub = iris.sample(60, random_state=1)                                                #> 60 flowers keep the plot readable.
lut = dict(zip(sub["species"].unique(), ["#1f4fff", "#e0245e", "#0f9b62"]))          #> Species -> colour.
g = sns.clustermap(sub.drop(columns=["target", "species"]), standard_scale=1,        #> standard_scale=1 rescales each column to 0..1.
                   row_colors=sub["species"].map(lut), cmap="mako", figsize=(6, 6),   #> Colour strip showing each row's species.
                   yticklabels=False)                                                #> Hide 60 row labels.
plt.show()                                                                           #> Draw.
```

## Multi-variable plots {I}

### pairplot and jointplot

:::def
**Pair plot (scatter-plot matrix)**: a grid with every variable plotted against every other, and each variable's distribution on the diagonal. **Joint plot**: one bivariate plot with each variable's marginal distribution drawn along its edges.
:::

```py title="pairplot" caption="Every pair of iris measurements, coloured by species; distributions on the diagonal. Petal measurements separate the species best."
g = sns.pairplot(iris.drop(columns="target"), hue="species", corner=True, height=1.9,   #> corner=True drops the redundant upper triangle.
                 plot_kws={"s": 12, "alpha": 0.7})                                   #> Small, transparent points.
plt.show()                                                                           #> Draw.
```

```py title="jointplot with three kinds" caption="The same two variables as scatter + histograms, as a hexbin density, and as 2-D KDE contours."
for kind in ["scatter", "hex", "kde"]:                                               #> Three styles.
    g = sns.jointplot(data=ins, x="age", y="bmi", kind=kind, height=3.4,             #> A joint plot of this kind.
                      color="#7b3ff2")                                               #> One colour.
    g.figure.suptitle(f"jointplot(kind='{kind}')", y=1.02)                           #> Title above the figure.
plt.show()                                                                           #> Draw all three figures.
```

## Styling Seaborn {I}

```py title="Themes, contexts and palettes" caption="The same plot in the 'paper' and 'talk' contexts: context scales fonts and lines for the medium."
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))                                    #> Two panels.
for ax, ctx in zip(axes, ["paper", "talk"]):                                         #> Two contexts.
    with sns.plotting_context(ctx):                                                  #> Temporarily set fonts and line widths for this medium.
        sns.boxplot(data=ins, x="region", y="bmi", hue="smoker", palette="colorblind", ax=ax)   #> A colour-blind-safe palette.
        ax.set_title(f"context = '{ctx}'")                                           #> Title.
        ax.tick_params(axis="x", rotation=20)                                        #> Tilt labels.
plt.tight_layout()                                                                   #> Tidy.
plt.show()                                                                           #> Draw.
print(sns.color_palette("colorblind").as_hex()[:4])                                  #> The palette's first four colours as hex codes.
```

| Setting | Options | Effect |
|---|---|---|
| `sns.set_theme(style=)` | `"whitegrid"`, `"darkgrid"`, `"white"`, `"dark"`, `"ticks"` | background and grid |
| `context=` | `"paper"`, `"notebook"`, `"talk"`, `"poster"` | font and line sizes |
| `palette=` | `"deep"`, `"colorblind"`, `"viridis"`, `"rocket"`, `"mako"`, `"Set2"` | colours |

## Saving Seaborn figures {B}

Axes-level plots live on a normal Matplotlib figure, so use `fig.savefig()`. Figure-level plots return a grid object with its own `savefig`.

```py title="Saving both kinds"
fig, ax = plt.subplots(figsize=(4, 3))                                               #> An axes-level plot on our own figure.
sns.histplot(data=ins, x="bmi", ax=ax)                                               #> Draw.
fig.savefig("bmi_hist.png", dpi=300, bbox_inches="tight")                            #> Save via the figure.
plt.close(fig)                                                                       #> Close without displaying.
g = sns.relplot(data=ins, x="age", y="charges", hue="smoker", height=3)              #> A figure-level plot.
g.savefig("age_charges.pdf")                                                         #> FacetGrid objects have their own savefig().
plt.close(g.figure)                                                                  #> Close it.
print("saved bmi_hist.png and age_charges.pdf")                                      #> Confirmation.
```

:::ex Try it yourself
1. Use `catplot(kind="violin")` to show `charges` by `children`, split into columns by `smoker`.
2. Make a `jointplot` of `bmi` vs `charges` with `kind="reg"`. What does the fitted line suggest?
3. Draw a `clustermap` of the insurance numeric columns (scaled) for 80 random customers, with smokers marked in the row colours.
:::
