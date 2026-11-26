# Interactive charts with Plotly

Plotly draws **interactive** charts: readers can hover to see exact values, zoom, pan, and click legend entries to hide or show series. That makes it ideal for exploring data, for dashboards, and for sharing results as web pages. With a static-image exporter, the same charts become PNG, PDF or SVG files for papers. In the web edition of this book, every chart in this chapter is live; in the PDF, you see the exported image.

:::def
**Interactive chart**: a chart that responds to the reader, with hover tooltips, zoom, pan and toggling series, usually displayed in a web browser or notebook.

**Plotly Express (`px`)**: Plotly's high-level interface. One function call builds a complete chart from a tidy DataFrame. **Graph objects (`go`)**: the low-level interface for full control and for chart types Express doesn't cover.
:::

:::note Installation
`pip install plotly` installs Plotly. To export static images (PNG, PDF, SVG) also run `pip install kaleido`. Recent Kaleido versions use a Chrome or Chromium browser installed on your computer to render images.
:::

## Set-up {B}

```py title="Imports and built-in datasets"
import numpy as np                                              #> Arrays.
import pandas as pd                                             #> Tables.
import plotly.express as px                                     #> High-level Plotly.
import plotly.graph_objects as go                               #> Low-level Plotly.
from plotly.subplots import make_subplots                       #> Grids of Plotly charts.
gap = px.data.gapminder()                                       #> Built-in dataset: countries' life expectancy, GDP and population, 1952-2007.
iris = px.data.iris()                                           #> Built-in iris dataset.
tips = px.data.tips()                                           #> Built-in restaurant tips dataset.
print(gap.shape, iris.shape, tips.shape)                        #> These ship with Plotly: no download needed.
print(gap.head(3))                                              #> A first look at gapminder.
```

Every chart below ends with `fig.show()`. In Jupyter it displays inline; in a script it opens your browser.

## Scatter and bubble charts {B}

```py title="Hover, colour, size and log axis" caption="Gapminder 2007: GDP per capita (log scale) vs life expectancy, bubble size = population, colour = continent. Hover over a bubble to see the country."
df07 = gap[gap["year"] == 2007]                                 #> One year only.
fig = px.scatter(df07, x="gdpPercap", y="lifeExp", size="pop", color="continent",   #> One call: x, y, bubble size, colour.
                 hover_name="country", log_x=True, size_max=45,                     #> Tooltip title; log x-axis; largest bubble size.
                 labels={"gdpPercap": "GDP per capita (US$)", "lifeExp": "Life expectancy (years)"},   #> Readable axis titles.
                 title="Wealth and health of nations, 2007", width=820, height=460) #> Title and size in pixels.
fig.show()                                                      #> Display.
```

### Animated scatter

```py title="Animation over time" caption="The same chart animated from 1952 to 2007. In the web edition press Play or drag the slider; the PDF shows the first frame."
fig = px.scatter(gap, x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="country",   #> All years this time...
                 animation_frame="year", animation_group="country", log_x=True, size_max=45,   #> ...one animation frame per year; each country keeps its identity.
                 range_x=[200, 60000], range_y=[25, 90], width=820, height=500,  #> Fixed ranges so axes don't jump between frames.
                 title="Gapminder 1952-2007")                                     #> Title.
fig.show()                                                      #> Display.
```

## Line and area charts {B}

```py title="Time series with a range slider" caption="Life expectancy over time for five countries. The slider under the chart zooms into a period."
sel = gap[gap["country"].isin(["Bangladesh", "India", "China", "Brazil", "Japan"])]   #> Five countries.
fig = px.line(sel, x="year", y="lifeExp", color="country", markers=True,          #> One line per country, with markers.
              title="Life expectancy, 1952-2007", width=820, height=440)          #> Title and size.
fig.update_xaxes(rangeslider_visible=True)                                        #> Add an interactive range slider below the x-axis.
fig.show()                                                      #> Display.
```

```py title="Stacked area" caption="Population by continent over time, stacked: the top edge is world population in the dataset."
pop = gap.groupby(["year", "continent"], as_index=False)["pop"].sum()             #> Total population per continent and year.
fig = px.area(pop, x="year", y="pop", color="continent",                          #> Areas stacked by continent.
              title="Population by continent", width=820, height=420)             #> Title and size.
fig.show()                                                      #> Display.
```

## Bar charts {B}

```py title="Grouped bars with values" caption="Average tip by day and sex. barmode='group' places bars side by side; text_auto writes the values."
avg = tips.groupby(["day", "sex"], as_index=False)["tip"].mean()                  #> Mean tip per day and sex.
fig = px.bar(avg, x="day", y="tip", color="sex", barmode="group", text_auto=".2f",   #> Grouped bars, values to 2 decimals.
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"]},              #> Put the days in calendar order.
             title="Average tip by day", width=820, height=420)                   #> Title and size.
fig.show()                                                      #> Display.
```

## Distributions {B}

```py title="Histogram with marginal box plot" caption="Distribution of total bills by time of day. The marginal box plots above summarise each group."
fig = px.histogram(tips, x="total_bill", color="time", nbins=30, marginal="box",  #> Histogram plus box plots in the margin.
                   opacity=0.75, barmode="overlay",                               #> Overlay the two groups semi-transparently.
                   title="Total bill distribution", width=820, height=440)        #> Title and size.
fig.show()                                                      #> Display.
```

```py title="Box and violin side by side" caption="Left: box plots with all points shown. Right: violin plots of the same data. Hover to see quartiles and individual values."
fig = make_subplots(rows=1, cols=2, subplot_titles=("Box", "Violin"))             #> A 1x2 grid of Plotly charts.
for day in ["Thur", "Fri", "Sat", "Sun"]:                                          #> One trace per day.
    d = tips[tips["day"] == day]["total_bill"]                                     #> That day's bills.
    fig.add_trace(go.Box(y=d, name=day, boxpoints="all", jitter=0.4, showlegend=False), row=1, col=1)   #> Box with jittered points.
    fig.add_trace(go.Violin(y=d, name=day, box_visible=True, showlegend=False), row=1, col=2)          #> Violin with an inner box.
fig.update_layout(title="Total bill by day", width=820, height=440)               #> Layout settings for the whole figure.
fig.show()                                                      #> Display.
```

## Relationships among many variables {I}

```py title="Scatter-plot matrix" caption="Every pair of iris measurements; hover shows the values, and selecting points in one panel highlights them in all panels."
fig = px.scatter_matrix(iris, dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],   #> Variables to pair.
                        color="species", title="Iris scatter matrix", width=820, height=700)               #> Colour by species.
fig.update_traces(diagonal_visible=False, marker=dict(size=4))                    #> Hide the empty diagonal; smaller markers.
fig.show()                                                      #> Display.
```

:::def
**Parallel coordinates plot**: each variable gets its own vertical axis, and each observation is a line crossing all the axes. Patterns across many variables show up as bundles of similar lines.
:::

```py title="Parallel coordinates" caption="Each line is one flower crossing four measurement axes, coloured by species. In the web edition, drag on any axis to filter."
fig = px.parallel_coordinates(iris, dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],   #> One axis per measurement.
                              color="species_id", color_continuous_scale=px.colors.qualitative.Bold[:3],        #> Colour by species code.
                              title="Iris in parallel coordinates", width=820, height=440)                     #> Title and size.
fig.show()                                                      #> Display.
```

## Density and heatmaps {I}

```py title="2-D density heatmap and contours" caption="Left: density heatmap of bill vs tip. Right: the same as density contours. Hover gives counts per cell."
fig = make_subplots(rows=1, cols=2, subplot_titles=("density_heatmap", "density_contour"))   #> Two panels.
h = px.density_heatmap(tips, x="total_bill", y="tip", nbinsx=25, nbinsy=20)       #> A full Express figure...
c = px.density_contour(tips, x="total_bill", y="tip")                             #> ...and another one.
fig.add_trace(h.data[0], row=1, col=1)                                            #> Copy each figure's trace into the grid.
fig.add_trace(c.data[0], row=1, col=2)                                            #> Second trace.
fig.update_layout(title="Bill vs tip density", width=820, height=420, coloraxis_showscale=False)   #> Hide the colour bar for a cleaner grid.
fig.show()                                                      #> Display.
```

```py title="Annotated heatmap with imshow" caption="Correlation matrix of the iris measurements with values written in the cells."
corr = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].corr().round(2)   #> 4x4 correlations.
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,      #> Diverging colours, fixed scale, values in cells.
                title="Correlation heatmap", width=620, height=520)                           #> Title and size.
fig.show()                                                      #> Display.
```

## Parts of a whole {B}

:::def
**Sunburst chart**: nested rings showing a hierarchy, with the centre as the root and each outer ring splitting its parent into parts. **Treemap**: the same hierarchy shown as nested rectangles whose areas are proportional to the values.
:::

```py title="Pie, sunburst and treemap" caption="Left: a donut of tips by day. Middle: sunburst of continent then country (2007 population). Right: the same hierarchy as a treemap. Click a segment to zoom in (web edition)."
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "domain"}] * 3],             #> "domain" subplots hold pie-like charts.
                    subplot_titles=("Pie / donut", "Sunburst", "Treemap"))       #> Titles.
by_day = tips.groupby("day", as_index=False)["tip"].sum()                         #> Total tips per day.
fig.add_trace(go.Pie(labels=by_day["day"], values=by_day["tip"], hole=0.45), row=1, col=1)   #> hole makes it a donut.
big = df07[df07["pop"] > 4e7]                                                     #> Only countries with more than 40 million people.
sb = px.sunburst(big, path=["continent", "country"], values="pop")                #> path = hierarchy levels from the centre outwards.
tm = px.treemap(big, path=["continent", "country"], values="pop")                 #> Same hierarchy as rectangles.
fig.add_trace(sb.data[0], row=1, col=2)                                           #> Copy the sunburst into the grid.
fig.add_trace(tm.data[0], row=1, col=3)                                           #> Copy the treemap.
fig.update_layout(width=900, height=420, showlegend=False)                        #> Size; no legend.
fig.show()                                                      #> Display.
```

## Flows: the Sankey diagram {I}

:::def
**Sankey diagram**: nodes connected by bands whose widths are proportional to the flow between them. It shows how quantities move or split between stages.
:::

```py title="Where did the data go?" caption="A dataset of 1,000 records flowing through cleaning, splitting and evaluation. Band widths are proportional to record counts."
labels = ["Raw data", "Duplicates removed", "Missing target", "Clean", "Train", "Test", "Correct", "Errors"]   #> Node names (index 0..7).
source = [0, 0, 0, 3, 3, 5, 5]                                                    #> Each band starts at this node index...
target = [1, 2, 3, 4, 5, 6, 7]                                                    #> ...and ends at this one.
value = [40, 60, 900, 720, 180, 160, 20]                                          #> Records flowing along each band.
fig = go.Figure(go.Sankey(node=dict(label=labels, pad=18, thickness=16,           #> Node settings: labels, spacing, bar width.
                                    color=["#7f8aa3", "#d42a2a", "#d42a2a", "#1f4fff", "#0f9b62", "#ff6a00", "#0f9b62", "#d42a2a"]),
                          link=dict(source=source, target=target, value=value)))  #> The flows.
fig.update_layout(title="Records through an ML pipeline", width=820, height=420)  #> Title and size.
fig.show()                                                      #> Display.
```

## 3-D charts {I}

```py title="Rotatable 3-D scatter and surface" caption="Left: iris in 3-D. Right: a surface. In the web edition, drag to rotate and scroll to zoom."
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],   #> "scene" subplots are 3-D.
                    subplot_titles=("scatter_3d", "surface"))                     #> Titles.
s3 = px.scatter_3d(iris, x="sepal_length", y="petal_length", z="petal_width", color="species")   #> 3-D scatter coloured by species.
for tr in s3.data:                                                                 #> One trace per species...
    tr.marker.size = 3                                                             #> ...with small markers...
    fig.add_trace(tr, row=1, col=1)                                                #> ...copied into the left panel.
xg = np.linspace(-3, 3, 50)                                                        #> Grid values.
Z = np.sin(np.sqrt(xg[:, None] ** 2 + xg[None, :] ** 2))                           #> z = sin(distance from centre), via broadcasting.
fig.add_trace(go.Surface(x=xg, y=xg, z=Z, colorscale="Viridis", showscale=False), row=1, col=2)   #> The surface.
fig.update_layout(width=900, height=460, showlegend=False)                        #> Size.
fig.show()                                                      #> Display.
```

## Maps {I}

Plotly draws maps with `px.choropleth` (regions coloured by value), `px.scatter_geo` (points on a map) and tile maps. The country outlines are downloaded by the browser when the chart is displayed, so maps need an internet connection.

```py norun title="A choropleth world map (needs internet when displayed)"
fig = px.choropleth(df07, locations="iso_alpha", color="lifeExp",                 #> iso_alpha holds 3-letter country codes such as BGD.
                    hover_name="country", color_continuous_scale="Viridis",       #> Tooltip and colour scale.
                    title="Life expectancy, 2007")                                #> Title.
fig.show()                                                                        #> Display in the browser.
```

## Customising and exporting Plotly charts {B}

```py title="Templates, fonts, annotations and export"
fig = px.scatter(df07, x="gdpPercap", y="lifeExp", color="continent", log_x=True, #> A base chart...
                 template="simple_white", width=700, height=420)                  #> ...using a clean built-in template.
fig.update_layout(font=dict(family="Arial", size=13),                             #> Font for the whole figure.
                  title=dict(text="Richer countries live longer", x=0.02),        #> Left-aligned title stating the finding.
                  legend=dict(orientation="h", y=-0.2))                           #> Horizontal legend below the chart.
fig.add_annotation(x=np.log10(42951.65), y=78.24, text="USA", showarrow=True)       #> On a log axis, x positions are given as log10 values.
fig.write_image("lifeexp.png", width=700, height=420, scale=3)                    #> PNG at 3x resolution (2100 px wide): sharp for print.
fig.write_image("lifeexp.pdf")                                                    #> Vector PDF for papers and LaTeX.
fig.write_image("lifeexp.svg")                                                    #> Vector SVG.
fig.write_html("lifeexp.html", include_plotlyjs="cdn")                            #> A standalone interactive web page (loads Plotly from the internet).
from pathlib import Path                                                          #> Check the files.
print({f: f"{Path(f).stat().st_size // 1024} KB" for f in ["lifeexp.png", "lifeexp.pdf", "lifeexp.svg", "lifeexp.html"]})   #> Sizes.
```

| Plotly method | Produces | Use for |
|---|---|---|
| `fig.write_image("f.png", scale=3)` | raster image | Word, PowerPoint, Google Docs |
| `fig.write_image("f.pdf")` / `"f.svg"` | vector image | journals, LaTeX, posters |
| `fig.write_html("f.html")` | interactive web page | sharing, supplementary material, websites |
| `fig.to_json()` / `pio.from_json()` | the figure's data | saving and reloading figures exactly |

| Template | Look |
|---|---|
| `"plotly"` | the default, light blue-grey background |
| `"plotly_white"`, `"simple_white"` | white backgrounds, good for papers |
| `"plotly_dark"` | dark background for slides |
| `"ggplot2"`, `"seaborn"` | mimic those libraries' styles |

:::bug `ValueError: Image export using the "kaleido" engine requires the kaleido package`
Install it with `pip install kaleido` into the same environment. Recent versions also need Chrome or Chromium installed; if it can't find a browser, run `plotly_get_chrome` (installed with Kaleido) or install Chrome. A blank chart in Jupyter usually means the notebook renderer isn't set up: try `import plotly.io as pio; pio.renderers.default = "notebook"` (or `"browser"`).
:::

:::ex Try it yourself
1. Make an animated bar chart of the five most populous countries' life expectancy over time (`animation_frame="year"`).
2. Draw a sunburst of `tips` by `day`, then `time`, then `sex`, with `values="tip"`.
3. Export one chart as a 300-dpi-equivalent PNG for a 7-inch-wide figure (hint: `width=700, scale=3`).
:::
