# The Matplotlib chart gallery

This chapter is a catalogue of the charts Matplotlib can draw. For each one you get a **definition**, **what it shows**, **when to use it** (and when not to), and a complete, line-by-line explained example with its real output. Use it as a menu: find the question you are asking, pick the chart, copy the code.

## Set-up for the gallery {B}

```py title="Shared imports and example data"
import numpy as np                                     #> Arrays and random numbers.
import pandas as pd                                    #> Tables.
import matplotlib.pyplot as plt                        #> The plotting interface.
rng = np.random.default_rng(42)                        #> One seeded generator: everyone gets identical charts.
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]    #> Category labels reused below.
sales_a = np.array([12, 15, 14, 18, 21, 24])           #> Monthly sales, product A.
sales_b = np.array([8, 9, 13, 12, 15, 19])             #> Monthly sales, product B.
print(len(months), sales_a.sum(), sales_b.sum())       #> Quick check of the data.
```

| Your question | Chart(s) in this chapter |
|---|---|
| How does a value change over time? | line, step, area, stacked area, twin axes |
| How are two numeric variables related? | scatter, bubble, 2-D histogram, hexbin, contour |
| How do categories compare? | bar, horizontal bar, grouped bar, stacked bar, lollipop, pie/donut |
| What does a distribution look like? | histogram, ECDF, box plot, violin plot |
| How uncertain are my estimates? | error bars, confidence bands |
| What does a matrix or grid look like? | heatmap, contour, 3-D surface |
| Direction, flow, cycles, schedules | quiver, streamplot, polar, radar, Gantt, event plot |

## Charts for change over time {B}

### Line plot

:::def
**Line plot**: points connected by straight lines in the order of the x-axis. It shows how a quantity changes along an ordered variable, usually time.
:::

**Use it when** the x-axis is ordered (time, dose, epoch) and you care about trends, peaks and turning points. **Avoid it** for unordered categories: joining "red" to "blue" with a line implies an order that doesn't exist.

```py title="Line plot with markers and a highlighted point" caption="Two products' monthly sales. Markers show the actual data points; the line shows the trend between them."
fig, ax = plt.subplots(figsize=(7, 3.6))                          #> One figure with one panel (Axes).
ax.plot(months, sales_a, marker="o", lw=2, label="Product A")     #> Line + circle markers; lw = line width.
ax.plot(months, sales_b, marker="s", ls="--", label="Product B")  #> Square markers, dashed line.
ax.annotate("record month", xy=(5, 24), xytext=(3.1, 23),         #> Label the highest point...
            arrowprops=dict(arrowstyle="->"))                      #> ...with an arrow pointing to it.
ax.set(title="Monthly sales", xlabel="month", ylabel="units sold")  #> Title and axis labels in one call.
ax.legend()                                                        #> Show which line is which.
plt.show()                                                         #> Draw it.
```

### Step plot

:::def
**Step plot**: a line that stays flat between data points and jumps at each one. It shows quantities that change in discrete jumps.
:::

**Use it when** a value holds constant until it changes, such as prices, inventory, the number of patients still at risk (survival curves), or a learning-rate schedule.

```py title="A learning-rate schedule as a step plot" caption="The learning rate is constant within each phase and drops at epochs 10, 20 and 30: steps, not slopes."
epochs = np.arange(0, 41)                                        #> Epoch numbers 0..40.
lr = 0.1 * 0.5 ** (epochs // 10)                                 #> Halve every 10 epochs (// is floor division).
fig, ax = plt.subplots(figsize=(7, 3.2))                         #> New figure.
ax.step(epochs, lr, where="post", lw=2)                          #> where="post": each value holds until the next x.
ax.set(title="Step decay schedule", xlabel="epoch", ylabel="learning rate")   #> Labels.
plt.show()                                                       #> Draw.
```

### Area chart and confidence band

:::def
**Area chart**: a line chart with the region below it filled. **Confidence band**: the region between a lower and an upper line, usually showing uncertainty around an estimate.
:::

**Use an area** to emphasise a volume or cumulative total. **Use a band** (`fill_between`) to show uncertainty such as a 95% confidence interval or ± one standard deviation around a mean.

```py title="Mean with a ± 1 SD band" caption="The solid line is the mean of 30 simulated runs; the shaded band covers ± 1 standard deviation."
x = np.linspace(0, 10, 100)                                      #> 100 x positions.
runs = np.sin(x) + rng.normal(0, 0.3, size=(30, 100))            #> 30 noisy repetitions of the same curve.
mean, sd = runs.mean(axis=0), runs.std(axis=0)                   #> Mean and SD at each x (down the 30 runs).
fig, ax = plt.subplots(figsize=(7, 3.4))                         #> New figure.
ax.fill_between(x, mean - sd, mean + sd, alpha=0.3, label="± 1 SD")   #> Shade between the lower and upper curves.
ax.plot(x, mean, lw=2, label="mean of 30 runs")                  #> The mean on top.
ax.set(title="Mean and variability", xlabel="x", ylabel="value") #> Labels.
ax.legend()                                                       #> Legend.
plt.show()                                                        #> Draw.
```

### Stacked area chart

:::def
**Stacked area chart**: several area series stacked on top of each other, so the top edge shows the total and each band shows one component's contribution.
:::

**Use it when** you want both the total over time and how it is made up. **Avoid it** with many series or wiggly data: middle bands are hard to read because their baseline moves.

```py title="Composition over time with stackplot" caption="Total monthly sales (top edge) split into products A and B."
fig, ax = plt.subplots(figsize=(7, 3.4))                                       #> New figure.
ax.stackplot(months, sales_a, sales_b, labels=["A", "B"], alpha=0.85)          #> Stack B on top of A.
ax.set(title="Total sales by product", xlabel="month", ylabel="units")         #> Labels.
ax.legend(loc="upper left")                                                     #> Legend in the empty corner.
plt.show()                                                                      #> Draw.
```

### Two y-axes (twin axes)

:::def
**Twin axes**: two y-axes sharing one x-axis, so two quantities with different units can be drawn together.
:::

**Use it sparingly**: when two series share the x-axis but have different units (temperature and rainfall, loss and accuracy). Colour each axis to match its line. Readers can be misled because the relative scales are arbitrary; two stacked panels are often clearer.

```py title="Rainfall (bars) and temperature (line)" caption="Bars use the left axis, the line uses the right axis; matching colours make the pairing clear."
rain = np.array([80, 65, 50, 30, 20, 10])                       #> mm per month.
temp = np.array([12, 14, 17, 21, 25, 29])                       #> °C per month.
fig, ax1 = plt.subplots(figsize=(7, 3.6))                       #> The first (left) axes.
ax1.bar(months, rain, color="#0091b3", alpha=0.7)               #> Bars on the left axis.
ax1.set_ylabel("rainfall (mm)", color="#0091b3")                #> Left label in the bar colour.
ax2 = ax1.twinx()                                               #> A second axes sharing the same x-axis, with its own y-axis on the right.
ax2.plot(months, temp, color="#e0245e", marker="o", lw=2)       #> The line on the right axis.
ax2.set_ylabel("temperature (°C)", color="#e0245e")             #> Right label in the line colour.
ax2.grid(False)                                                 #> Avoid two overlapping grids.
ax1.set_title("Climate summary")                                #> Title.
plt.show()                                                      #> Draw.
```

### Logarithmic axes

:::def
**Log scale**: an axis where equal distances represent equal *ratios* (1, 10, 100, 1000) rather than equal differences.
:::

**Use it when** values span several orders of magnitude, or when growth is exponential (a straight line on a log axis means a constant growth rate). Always label the axis as logarithmic.

```py title="The same data on linear and log axes" caption="Exponential growth looks explosive on a linear axis (left) and becomes a straight line on a log axis (right)."
days = np.arange(1, 31)                                         #> Days 1..30.
cases = 5 * 1.25 ** days                                        #> 25% growth per day.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))         #> Two panels side by side.
ax1.plot(days, cases); ax1.set(title="Linear y-axis", xlabel="day", ylabel="count")   #> Linear scale.
ax2.plot(days, cases); ax2.set(title="Log y-axis", xlabel="day", ylabel="count (log scale)", yscale="log")   #> yscale="log".
plt.tight_layout()                                              #> Prevent overlaps.
plt.show()                                                      #> Draw.
```

## Charts for relationships {B}

### Scatter and bubble plot

:::def
**Scatter plot**: one dot per observation at (x, y). It shows the relationship, clusters and outliers between two numeric variables. **Bubble plot**: a scatter plot where dot size encodes a third variable.
:::

**Use it when** you want to see whether two variables move together, and how strongly; colour and size can add a third and fourth variable. **Avoid it** with hundreds of thousands of points: they overplot into a blob, so use transparency, a 2-D histogram or hexbin instead.

```py title="Scatter with colour and size encoding" caption="Each dot is one simulated city: x = income, y = life expectancy, colour = region, size = population."
n = 60                                                          #> 60 observations.
income = rng.uniform(5, 60, n)                                  #> x variable.
life = 55 + 8 * np.log(income) + rng.normal(0, 2, n)            #> y depends on log(income), plus noise.
region = rng.integers(0, 3, n)                                  #> A category coded 0, 1, 2.
pop = rng.uniform(0.2, 10, n)                                   #> Population in millions: the bubble size.
fig, ax = plt.subplots(figsize=(7, 4.2))                        #> New figure.
sc = ax.scatter(income, life, c=region, s=pop * 30, cmap="viridis", alpha=0.7, edgecolor="k")   #> c = colour by value, s = marker area.
ax.set(title="Income vs life expectancy", xlabel="income (thousand $)", ylabel="life expectancy (years)")   #> Labels.
handles, labels = sc.legend_elements()                          #> Legend entries for the colours used.
ax.legend(handles, ["North", "Centre", "South"], title="region", loc="lower right")   #> A legend with readable names.
plt.show()                                                      #> Draw.
```

### 2-D histogram and hexbin

:::def
**2-D histogram**: the plane is divided into rectangular cells and each cell is coloured by how many points fall inside it. **Hexbin**: the same idea with hexagonal cells. Both show the density of a large cloud of points.
:::

**Use them when** a scatter plot would be an unreadable blob (thousands of points or more).

```py title="Density of 20,000 points" caption="Left: square cells (hist2d). Right: hexagonal cells (hexbin). Bright cells hold the most points."
xs = rng.normal(0, 1, 20000)                                    #> 20,000 x values.
ys = xs * 0.6 + rng.normal(0, 0.8, 20000)                       #> Correlated y values.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))         #> Two panels.
h = ax1.hist2d(xs, ys, bins=40, cmap="magma")                   #> 40 x 40 grid of counts.
fig.colorbar(h[3], ax=ax1, label="count")                       #> h[3] is the image object needed for a colour bar.
ax1.set(title="hist2d", xlabel="x", ylabel="y")                 #> Labels.
hb = ax2.hexbin(xs, ys, gridsize=30, cmap="magma")              #> Hexagons, about 30 across.
fig.colorbar(hb, ax=ax2, label="count")                         #> Colour bar for the hexbin.
ax2.set(title="hexbin", xlabel="x")                             #> Labels.
for a in (ax1, ax2): a.grid(False)                              #> Grid lines would cut through the cells.
plt.tight_layout()                                              #> Tidy spacing.
plt.show()                                                      #> Draw.
```

### Contour and filled contour

:::def
**Contour plot**: lines joining points of equal value on a 2-D surface, like height lines on a map. **Filled contour**: the bands between contour lines filled with colour.
:::

**Use it when** a quantity depends on two continuous variables: a loss surface over two parameters, a probability density, a decision function.

```py title="A loss surface with contours" caption="Contour lines of a two-parameter loss function; the minimum lies at the centre of the innermost ring."
w1, w2 = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))   #> Grids of all (w1, w2) pairs.
loss = (w1 - 1) ** 2 + 2 * (w2 + 0.5) ** 2 + 0.5 * w1 * w2              #> A bowl-shaped loss.
fig, ax = plt.subplots(figsize=(6, 4.6))                        #> New figure.
cf = ax.contourf(w1, w2, loss, levels=20, cmap="viridis")       #> Filled bands at 20 levels.
cs = ax.contour(w1, w2, loss, levels=8, colors="white", linewidths=0.8)   #> 8 thin white contour lines on top.
ax.clabel(cs, fontsize=8)                                       #> Write the value on each line.
fig.colorbar(cf, label="loss")                                  #> Colour scale.
ax.set(title="Loss surface", xlabel="w1", ylabel="w2")          #> Labels.
ax.grid(False)                                                  #> No grid over the colours.
plt.show()                                                      #> Draw.
```

## Charts for comparing categories {B}

### Bar and horizontal bar chart

:::def
**Bar chart**: one rectangle per category, with length proportional to its value. **Horizontal bar chart**: the same, with bars running left to right.
:::

**Use it to** compare a quantity across categories. Bars must start at zero, because readers compare lengths. **Prefer horizontal bars** when category names are long or there are many categories, and sort them by value.

```py title="Vertical and sorted horizontal bars" caption="Left: vertical bars with values on top. Right: horizontal bars sorted by value, which makes ranking instant."
langs = ["Python", "R", "Julia", "MATLAB", "SQL"]               #> Categories.
users = np.array([68, 22, 4, 9, 41])                            #> Values (illustrative survey percentages).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))         #> Two panels.
bars = ax1.bar(langs, users, color="#0091b3")                   #> Vertical bars.
ax1.bar_label(bars, fmt="%d%%")                                 #> Write each value on its bar (%% prints a literal %).
ax1.set(title="Vertical bars", ylabel="% of respondents")       #> Labels.
order = np.argsort(users)                                       #> Indices that sort the values ascending.
ax2.barh(np.array(langs)[order], users[order], color="#e0245e") #> Horizontal bars, smallest at the bottom.
ax2.set(title="Horizontal bars, sorted", xlabel="% of respondents")   #> Labels.
ax2.grid(axis="y", visible=False)                               #> Keep only vertical grid lines.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

### Grouped and stacked bar charts

:::def
**Grouped bar chart**: bars for several series placed side by side within each category. **Stacked bar chart**: the series stacked on top of each other, so the full bar is the total.
:::

**Use grouped bars** to compare series within each category. **Use stacked bars** to show totals and their composition. Comparing the middle segments of a stacked bar is hard, because they don't share a baseline.

```py title="Grouped vs stacked" caption="The same data two ways: grouped bars compare A and B month by month; stacked bars show the monthly total and its split."
x = np.arange(len(months))                                      #> Bar positions 0..5.
w = 0.4                                                         #> Width of each bar.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))         #> Two panels.
ax1.bar(x - w / 2, sales_a, w, label="A")                       #> Shift A left by half a bar...
ax1.bar(x + w / 2, sales_b, w, label="B")                       #> ...and B right by half a bar.
ax1.set(xticks=x, xticklabels=months, title="Grouped")          #> Put month names at the group centres.
ax1.legend()                                                    #> Legend.
ax2.bar(months, sales_a, label="A")                             #> Bottom segment.
ax2.bar(months, sales_b, bottom=sales_a, label="B")             #> `bottom=` starts B where A ends.
ax2.set(title="Stacked")                                        #> Title.
ax2.legend()                                                    #> Legend.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

### Lollipop chart

:::def
**Lollipop chart**: a bar chart with each bar replaced by a thin line and a dot at its end. It carries the same information with less ink.
:::

**Use it** instead of a bar chart when there are many categories or the values are similar, so heavy bars would dominate.

```py title="A lollipop chart with stem()" caption="Each stem rises from zero to the value; the dot marks the exact value."
fig, ax = plt.subplots(figsize=(7, 3.4))                        #> New figure.
ax.stem(months, sales_a, basefmt=" ")                           #> stem() draws lines with markers; basefmt=" " hides the baseline.
ax.set(title="Sales of product A", ylabel="units", ylim=(0, 27))   #> Labels; start the axis at 0.
plt.show()                                                      #> Draw.
```

### Pie and donut chart

:::def
**Pie chart**: a circle divided into slices whose angles are proportional to each category's share of the whole. **Donut chart**: a pie with the centre removed.
:::

**Use it only** for a few categories (ideally 2 to 5) that add up to a meaningful whole, when the message is "this part is about a quarter". People judge angles poorly, so a bar chart is usually more accurate.

```py title="Pie and donut" caption="Shares of four categories. The donut leaves room in the middle for a total."
shares = [45, 30, 15, 10]                                       #> Percentages summing to 100.
names = ["Train", "Validation", "Test", "Held back"]            #> Slice labels.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))          #> Two panels.
ax1.pie(shares, labels=names, autopct="%1.0f%%", startangle=90) #> autopct writes each share; start at 12 o'clock.
ax1.set_title("Pie")                                            #> Title.
ax2.pie(shares, labels=names, startangle=90, wedgeprops=dict(width=0.4))   #> width < 1 cuts out the centre.
ax2.text(0, 0, "100%", ha="center", va="center", fontsize=14, fontweight="bold")   #> Text in the hole.
ax2.set_title("Donut")                                          #> Title.
plt.show()                                                      #> Draw.
```

## Charts for distributions {B}

### Histogram

:::def
**Histogram**: the range of a numeric variable is split into intervals (bins), and a bar shows how many observations fall in each. It shows the shape of a distribution: centre, spread, skew, multiple peaks and outliers.
:::

**Use it** to look at any single numeric variable. **Try several bin counts**, because the shape can change with them. Use `density=True` to compare groups of different sizes.

```py title="Bins change the story" caption="The same 500 values with 5, 20 and 60 bins: too few hides the two peaks, too many adds noise."
data = np.r_[rng.normal(0, 1, 300), rng.normal(4, 0.8, 200)]   #> A two-peaked (bimodal) sample.
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False) #> Three panels.
for ax, bins in zip(axes, [5, 20, 60]):                         #> One bin count per panel.
    ax.hist(data, bins=bins, edgecolor="white")                 #> White edges separate the bars.
    ax.set_title(f"{bins} bins")                                #> Title.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

### ECDF (empirical cumulative distribution)

:::def
**ECDF**: for every value x, the share of observations less than or equal to x. It rises from 0 to 1 and needs no bins.
:::

**Use it to** compare distributions precisely and read off percentiles directly: where the curve crosses 0.5 is the median.

```py title="Comparing two groups with ECDFs" caption="Group B's curve is to the right: its values are generally larger. The 0.5 line crosses each curve at its median."
g1, g2 = rng.normal(50, 10, 300), rng.normal(58, 12, 300)       #> Two samples.
fig, ax = plt.subplots(figsize=(7, 3.6))                        #> New figure.
ax.ecdf(g1, label="group A")                                    #> ax.ecdf (Matplotlib >= 3.8) draws the step-shaped ECDF.
ax.ecdf(g2, label="group B")                                    #> Second group.
ax.axhline(0.5, ls=":", color="grey")                           #> The median line.
ax.set(title="Empirical CDFs", xlabel="score", ylabel="proportion ≤ score")   #> Labels.
ax.legend()                                                     #> Legend.
plt.show()                                                      #> Draw.
```

### Box plot

:::def
**Box plot (box-and-whisker plot)**: the box spans the first to the third quartile (the middle 50% of the data), the line inside is the median, the whiskers reach the most extreme points within 1.5 × IQR of the box, and dots beyond them are flagged as outliers.
:::

**Use it to** compare the centre and spread of several groups compactly. It hides the shape (two peaks look like one), so add the raw points or use a violin plot when the shape matters.

```py title="Annotated box plot" caption="Three treatment groups. Read each box: median line, middle 50% in the box, whiskers, and outliers as circles."
groups = [rng.normal(10, 2, 80), rng.normal(13, 3, 80), np.r_[rng.normal(11, 1.5, 76), [20, 21, 3, 2]]]   #> The third group has 4 outliers.
fig, ax = plt.subplots(figsize=(7, 3.8))                        #> New figure.
ax.boxplot(groups, tick_labels=["control", "drug A", "drug B"], patch_artist=True,   #> patch_artist=True allows filled boxes.
           boxprops=dict(facecolor="#cfe8f0"), medianprops=dict(color="#e0245e", lw=2))   #> Box fill and a bold median.
ax.set(title="Response by treatment", ylabel="response")       #> Labels.
plt.show()                                                      #> Draw.
```

### Violin plot

:::def
**Violin plot**: a mirrored density curve for each group, so the width at each height shows how common that value is. It combines the summary of a box plot with the shape of a histogram.
:::

**Use it when** the shape of each group's distribution matters, for example to reveal two peaks a box plot would hide.

```py title="Violins reveal hidden structure" caption="Group 2 has two peaks. Its box plot would look ordinary; the violin shows the split clearly."
vio = [rng.normal(0, 1, 300), np.r_[rng.normal(-2, 0.6, 150), rng.normal(2, 0.6, 150)], rng.normal(1, 0.5, 300)]   #> Group 2 is bimodal.
fig, ax = plt.subplots(figsize=(7, 3.8))                        #> New figure.
parts = ax.violinplot(vio, showmedians=True)                    #> One violin per group, with a median line.
ax.set(xticks=[1, 2, 3], xticklabels=["group 1", "group 2", "group 3"], title="Violin plot", ylabel="value")   #> Labels.
plt.show()                                                      #> Draw.
```

## Charts for uncertainty {I}

### Error bars

:::def
**Error bar**: a line through a data point showing the uncertainty or spread of that value, for example ± 1 standard deviation, ± 1 standard error, or a 95% confidence interval.
:::

**Always say** in the caption what the bars represent: SD, SEM and 95% CI look alike but mean very different things.

```py title="Means with 95% confidence intervals" caption="Mean accuracy of four models over 10 runs; the bars show 95% confidence intervals (mean ± 1.96 × SEM)."
models = ["LogReg", "SVM", "RF", "Boosting"]                    #> Categories.
acc = [rng.normal(m, 0.015, 10) for m in (0.88, 0.90, 0.91, 0.915)]   #> 10 simulated runs per model.
means = [a.mean() for a in acc]                                 #> Mean of each model.
ci = [1.96 * a.std(ddof=1) / np.sqrt(len(a)) for a in acc]      #> 95% CI half-width = 1.96 × standard error.
fig, ax = plt.subplots(figsize=(7, 3.6))                        #> New figure.
ax.errorbar(models, means, yerr=ci, fmt="o", capsize=6, ms=8, lw=2)   #> fmt="o": dots only; capsize: little caps on the bars.
ax.set(title="Model accuracy (mean ± 95% CI)", ylabel="accuracy")      #> Labels.
plt.show()                                                      #> Draw.
```

## Charts for matrices and grids {I}

### Heatmap

:::def
**Heatmap**: a grid of cells whose colours represent values. It shows patterns in a matrix, such as a correlation matrix, a confusion matrix, or a table of values over two categories.
:::

**Use a sequential colour map** (`viridis`, `magma`) for values from low to high, and a **diverging** one (`RdBu_r`) centred on zero for values that can be positive or negative. Write the numbers in the cells when readers need exact values.

```py title="Annotated heatmap with imshow" caption="Average activity by weekday and hour block. Numbers in the cells give exact values; the colour shows the pattern."
days_ = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]       #> Row labels.
blocks = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]      #> Column labels.
vals = rng.integers(0, 50, (7, 6)) + np.array([0, 5, 30, 35, 25, 10])   #> Busier around midday.
fig, ax = plt.subplots(figsize=(7, 4.4))                        #> New figure.
im = ax.imshow(vals, cmap="YlOrRd")                             #> Draw the matrix as coloured cells.
ax.set(xticks=range(6), xticklabels=blocks, yticks=range(7), yticklabels=days_,   #> Tick labels...
       xlabel="hour of day", title="Activity heatmap")         #> ...axis label and title.
for i in range(7):                                              #> For every row...
    for j in range(6):                                          #> ...and every column...
        ax.text(j, i, vals[i, j], ha="center", va="center", fontsize=8)   #> ...write the value in the cell.
fig.colorbar(im, label="events")                                #> Colour scale.
ax.grid(False)                                                  #> No grid over the cells.
plt.show()                                                      #> Draw.
```

### 3-D surface and 3-D scatter

:::def
**3-D surface plot**: a surface z = f(x, y) drawn in perspective. **3-D scatter plot**: points placed by three coordinates.
:::

**Use 3-D sparingly.** Perspective distorts values and hides points behind others; for print, a contour plot or a heatmap is usually clearer. 3-D shines for intuition (loss landscapes, geometry) and in interactive tools (see Plotly in [[ch:viz-plotly]]).

```py title="Surface and scatter in 3-D" caption="Left: a 3-D surface of a 'Mexican hat' function. Right: three clusters of points in 3-D space."
fig = plt.figure(figsize=(11, 4.4))                             #> An empty figure; 3-D axes are added explicitly.
ax1 = fig.add_subplot(1, 2, 1, projection="3d")                 #> projection="3d" creates a 3-D axes.
X3, Y3 = np.meshgrid(np.linspace(-4, 4, 60), np.linspace(-4, 4, 60))   #> A grid of (x, y).
R = np.sqrt(X3 ** 2 + Y3 ** 2)                                  #> Distance from the centre.
ax1.plot_surface(X3, Y3, np.sin(R) / (R + 0.1), cmap="viridis", linewidth=0)   #> Surface coloured by height.
ax1.set_title("plot_surface")                                   #> Title.
ax2 = fig.add_subplot(1, 2, 2, projection="3d")                 #> Second 3-D axes.
for k, c in enumerate(["#1f4fff", "#e0245e", "#0f9b62"]):       #> Three clusters, three colours.
    pts = rng.normal(k * 2.5, 0.7, (60, 3))                     #> 60 points around (2.5k, 2.5k, 2.5k).
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=12) #> x, y, z coordinates.
ax2.set_title("3-D scatter")                                    #> Title.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

## Charts for direction, cycles and schedules {A}

### Quiver and streamplot

:::def
**Quiver plot**: arrows on a grid showing the direction and strength of a vector field. **Streamplot**: continuous lines that follow the field, like the flow of water.
:::

**Use them for** physical flows (wind, currents), gradients (the direction gradient descent will move), and dynamical systems.

```py title="A vector field two ways" caption="Left: arrows (quiver) show direction and magnitude at grid points. Right: streamlines trace the flow, coloured by speed."
gx, gy = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))   #> Coarse grid for arrows.
u, v = -gy, gx                                                  #> A rotating field: (u, v) at each point.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))         #> Two panels.
ax1.quiver(gx, gy, u, v, np.hypot(u, v), cmap="viridis")        #> Arrows coloured by length (hypot = sqrt(u² + v²)).
ax1.set(title="quiver", aspect="equal")                         #> Equal aspect keeps circles round.
fx, fy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))   #> Fine grid for streamlines.
fu, fv = -fy - 0.3 * fx, fx - 0.3 * fy                          #> A spiral sink.
ax2.streamplot(fx, fy, fu, fv, color=np.hypot(fu, fv), cmap="magma", density=1.2)   #> Streamlines coloured by speed.
ax2.set(title="streamplot", aspect="equal")                     #> Title.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

### Polar plot and radar chart

:::def
**Polar plot**: points placed by angle and distance from the centre. It suits cyclic data such as wind direction, time of day or seasons. **Radar (spider) chart**: several variables on spokes around a circle, with each item drawn as a closed polygon.
:::

**Use a polar plot** for genuinely cyclic data. **Use a radar chart** to compare the profile of a few items across a handful of variables on the same scale; with many items it becomes a tangle.

```py title="Polar bars and a radar chart" caption="Left: activity by hour on a 24-hour clock. Right: two models' profiles across five criteria (scores 0-10)."
fig = plt.figure(figsize=(10, 4.4))                             #> Empty figure.
axp = fig.add_subplot(1, 2, 1, projection="polar")              #> A polar axes.
hours = np.linspace(0, 2 * np.pi, 24, endpoint=False)           #> 24 angles around the circle.
activity = 5 + 4 * np.sin(hours - np.pi / 2) + rng.uniform(0, 1.5, 24)   #> Peaks around midday (angle π = 12h on this clock).
axp.bar(hours, activity, width=2 * np.pi / 24, alpha=0.8)       #> One wedge per hour.
axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)   #> Midnight at the top, clockwise like a clock.
axp.set_xticks(hours[::3], [f"{h}h" for h in range(0, 24, 3)])  #> Label every third hour.
axp.set_title("Activity by hour")                               #> Title.
crit = ["accuracy", "speed", "interpretability", "robustness", "simplicity"]   #> Radar spokes.
ang = np.linspace(0, 2 * np.pi, len(crit), endpoint=False).tolist()   #> One angle per criterion.
axr = fig.add_subplot(1, 2, 2, projection="polar")              #> Second polar axes.
for name, sc_ in [("Random forest", [9, 6, 4, 8, 6]), ("Logistic reg.", [7, 9, 9, 6, 9])]:   #> Two items to compare.
    vals_ = sc_ + sc_[:1]                                       #> Repeat the first value to close the polygon.
    axr.plot(ang + ang[:1], vals_, lw=2, label=name)            #> Outline.
    axr.fill(ang + ang[:1], vals_, alpha=0.2)                   #> Light fill.
axr.set_xticks(ang, crit)                                       #> Spoke labels.
axr.set_ylim(0, 10)                                             #> Same scale on every spoke.
axr.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.15))     #> Legend outside the circle.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

### Gantt chart (timeline) and event plot

:::def
**Gantt chart**: horizontal bars placed on a time axis, showing when each task starts and ends. **Event plot (raster plot)**: a short vertical tick for each event in time, one row per series.
:::

**Use a Gantt chart** for project and experiment schedules. **Use an event plot** for timestamps of discrete events: spikes of neurons, customer visits, machine faults.

```py title="A research project timeline and an event raster" caption="Left: a thesis schedule drawn with broken_barh. Right: event times for five sensors."
tasks = [("Literature review", 0, 3), ("Data collection", 2, 5), ("Modelling", 6, 4), ("Writing", 9, 4)]   #> (name, start month, duration).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))         #> Two panels.
for i, (name, start, dur) in enumerate(tasks):                  #> One row per task.
    ax1.broken_barh([(start, dur)], (i - 0.35, 0.7), color=f"C{i}")   #> A bar from `start` lasting `dur`, 0.7 high.
ax1.set(yticks=range(len(tasks)), yticklabels=[t[0] for t in tasks], xlabel="month", title="Gantt chart")   #> Labels.
ax1.invert_yaxis()                                              #> First task at the top.
events = [np.sort(rng.uniform(0, 10, rng.integers(5, 25))) for _ in range(5)]   #> Random event times for 5 sensors.
ax2.eventplot(events, colors="#ff6a00", lineoffsets=range(5), linelengths=0.7)   #> One row of ticks per sensor.
ax2.set(xlabel="time (s)", ylabel="sensor", title="Event plot")  #> Labels.
plt.tight_layout()                                              #> Tidy.
plt.show()                                                      #> Draw.
```

## Combining charts: layouts, insets and annotations {I}

```py title="A dashboard-style figure" caption="A main chart with an inset zoom, a marginal histogram, and a shaded region: four Matplotlib techniques in one figure."
fig, axd = plt.subplot_mosaic([["main", "side"]], width_ratios=[3, 1], figsize=(10, 3.8), layout="constrained")   #> Main panel 3x wider than the side panel.
t = np.linspace(0, 20, 400)                                     #> Time axis.
sig = np.sin(t) * np.exp(-t / 12) + rng.normal(0, 0.05, t.size) #> A damped, noisy signal.
axd["main"].plot(t, sig, lw=1)                                  #> The signal.
axd["main"].axvspan(4, 6, color="#ffd56b", alpha=0.4)           #> Shade an interval of interest.
axd["main"].set(title="Damped oscillation", xlabel="time", ylabel="signal")   #> Labels.
ins = axd["main"].inset_axes([0.62, 0.55, 0.35, 0.4])           #> A small axes inside the main one: [x, y, width, height] as fractions.
ins.plot(t, sig, lw=1); ins.set_xlim(4, 6); ins.set_ylim(-1, 0.2)   #> Same data, zoomed into the shaded interval.
ins.set_title("zoom 4-6", fontsize=8)                           #> Inset title.
axd["main"].indicate_inset_zoom(ins, edgecolor="grey")          #> Draw lines linking the inset to its region.
axd["side"].hist(sig, bins=30, orientation="horizontal", color="#7b3ff2")   #> Distribution of the signal values, sideways.
axd["side"].set(title="value distribution", xlabel="count")     #> Labels.
plt.show()                                                      #> Draw.
```

## Getting the figure out: PNG, PDF and SVG {B}

Every Matplotlib figure is saved with `fig.savefig()`. The file extension chooses the format. [[ch:viz-export]] covers sizes, fonts, journals, Word and LaTeX in detail; here is the essential pattern.

```py title="Save one figure in three formats"
fig, ax = plt.subplots(figsize=(3.5, 2.6))                      #> 3.5 inches wide: a typical single-column journal width.
ax.plot(months, sales_a, marker="o")                            #> Any chart.
ax.set(title="Sales", ylabel="units")                           #> Labels.
fig.savefig("sales.png", dpi=300, bbox_inches="tight")          #> Raster image at 300 dots per inch, trimmed to the content.
fig.savefig("sales.pdf", bbox_inches="tight")                   #> Vector PDF: sharp at any zoom, ideal for papers and LaTeX.
fig.savefig("sales.svg", bbox_inches="tight")                   #> Vector SVG: editable in Inkscape or Illustrator, insertable in Word.
plt.close(fig)                                                  #> Free the memory; we don't need to display it.
from pathlib import Path                                        #> For checking the files.
for f in ["sales.png", "sales.pdf", "sales.svg"]:               #> Report each file.
    print(f, Path(f).stat().st_size // 1024, "KB")              #> File size in kilobytes.
```

:::ex Try it yourself
1. Draw a grouped bar chart of mean `charges` by `region` for smokers and non-smokers, using `insurance.csv`.
2. Make an ECDF of `bmi` for each sex on one axes. Which group has the higher median?
3. Save any chart from this chapter as a 600-dpi PNG and as a PDF, and compare the file sizes.
:::
