"""The Annotated ML Handbook, Chapter 17: Interactive charts with Plotly

By Mahir Absar Khan. The code of the chapter only; the full text with line-by-line
explanations is in notebooks/part-4-visualization/17-viz-plotly.ipynb.
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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
gap = px.data.gapminder()
iris = px.data.iris()
tips = px.data.tips()
print(gap.shape, iris.shape, tips.shape)
print(gap.head(3))


# %%
df07 = gap[gap["year"] == 2007]
fig = px.scatter(df07, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=45,
                 labels={"gdpPercap": "GDP per capita (US$)", "lifeExp": "Life expectancy (years)"},
                 title="Wealth and health of nations, 2007", width=820, height=460)
fig.show()


# %%
fig = px.scatter(gap, x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="country",
                 animation_frame="year", animation_group="country", log_x=True, size_max=45,
                 range_x=[200, 60000], range_y=[25, 90], width=820, height=500,
                 title="Gapminder 1952-2007")
fig.show()


# %%
sel = gap[gap["country"].isin(["Bangladesh", "India", "China", "Brazil", "Japan"])]
fig = px.line(sel, x="year", y="lifeExp", color="country", markers=True,
              title="Life expectancy, 1952-2007", width=820, height=440)
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# %%
pop = gap.groupby(["year", "continent"], as_index=False)["pop"].sum()
fig = px.area(pop, x="year", y="pop", color="continent",
              title="Population by continent", width=820, height=420)
fig.show()


# %%
avg = tips.groupby(["day", "sex"], as_index=False)["tip"].mean()
fig = px.bar(avg, x="day", y="tip", color="sex", barmode="group", text_auto=".2f",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"]},
             title="Average tip by day", width=820, height=420)
fig.show()


# %%
fig = px.histogram(tips, x="total_bill", color="time", nbins=30, marginal="box",
                   opacity=0.75, barmode="overlay",
                   title="Total bill distribution", width=820, height=440)
fig.show()


# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("Box", "Violin"))
for day in ["Thur", "Fri", "Sat", "Sun"]:
    d = tips[tips["day"] == day]["total_bill"]
    fig.add_trace(go.Box(y=d, name=day, boxpoints="all", jitter=0.4, showlegend=False), row=1, col=1)
    fig.add_trace(go.Violin(y=d, name=day, box_visible=True, showlegend=False), row=1, col=2)
fig.update_layout(title="Total bill by day", width=820, height=440)
fig.show()


# %%
fig = px.scatter_matrix(iris, dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                        color="species", title="Iris scatter matrix", width=820, height=700)
fig.update_traces(diagonal_visible=False, marker=dict(size=4))
fig.show()


# %%
fig = px.parallel_coordinates(iris, dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                              color="species_id", color_continuous_scale=px.colors.qualitative.Bold[:3],
                              title="Iris in parallel coordinates", width=820, height=440)
fig.show()


# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("density_heatmap", "density_contour"))
h = px.density_heatmap(tips, x="total_bill", y="tip", nbinsx=25, nbinsy=20)
c = px.density_contour(tips, x="total_bill", y="tip")
fig.add_trace(h.data[0], row=1, col=1)
fig.add_trace(c.data[0], row=1, col=2)
fig.update_layout(title="Bill vs tip density", width=820, height=420, coloraxis_showscale=False)
fig.show()


# %%
corr = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].corr().round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Correlation heatmap", width=620, height=520)
fig.show()


# %%
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "domain"}] * 3],
                    subplot_titles=("Pie / donut", "Sunburst", "Treemap"))
by_day = tips.groupby("day", as_index=False)["tip"].sum()
fig.add_trace(go.Pie(labels=by_day["day"], values=by_day["tip"], hole=0.45), row=1, col=1)
big = df07[df07["pop"] > 4e7]
sb = px.sunburst(big, path=["continent", "country"], values="pop")
tm = px.treemap(big, path=["continent", "country"], values="pop")
fig.add_trace(sb.data[0], row=1, col=2)
fig.add_trace(tm.data[0], row=1, col=3)
fig.update_layout(width=900, height=420, showlegend=False)
fig.show()


# %%
labels = ["Raw data", "Duplicates removed", "Missing target", "Clean", "Train", "Test", "Correct", "Errors"]
source = [0, 0, 0, 3, 3, 5, 5]
target = [1, 2, 3, 4, 5, 6, 7]
value = [40, 60, 900, 720, 180, 160, 20]
fig = go.Figure(go.Sankey(node=dict(label=labels, pad=18, thickness=16,
                                    color=["#7f8aa3", "#d42a2a", "#d42a2a", "#1f4fff", "#0f9b62", "#ff6a00", "#0f9b62", "#d42a2a"]),
                          link=dict(source=source, target=target, value=value)))
fig.update_layout(title="Records through an ML pipeline", width=820, height=420)
fig.show()


# %%
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                    subplot_titles=("scatter_3d", "surface"))
s3 = px.scatter_3d(iris, x="sepal_length", y="petal_length", z="petal_width", color="species")
for tr in s3.data:
    tr.marker.size = 3
    fig.add_trace(tr, row=1, col=1)
xg = np.linspace(-3, 3, 50)
Z = np.sin(np.sqrt(xg[:, None] ** 2 + xg[None, :] ** 2))
fig.add_trace(go.Surface(x=xg, y=xg, z=Z, colorscale="Viridis", showscale=False), row=1, col=2)
fig.update_layout(width=900, height=460, showlegend=False)
fig.show()


# %%
fig = px.scatter(df07, x="gdpPercap", y="lifeExp", color="continent", log_x=True,
                 template="simple_white", width=700, height=420)
fig.update_layout(font=dict(family="Arial", size=13),
                  title=dict(text="Richer countries live longer", x=0.02),
                  legend=dict(orientation="h", y=-0.2))
fig.add_annotation(x=np.log10(42951.65), y=78.24, text="USA", showarrow=True)
fig.write_image("lifeexp.png", width=700, height=420, scale=3)
fig.write_image("lifeexp.pdf")
fig.write_image("lifeexp.svg")
fig.write_html("lifeexp.html", include_plotlyjs="cdn")
from pathlib import Path
print({f: f"{Path(f).stat().st_size // 1024} KB" for f in ["lifeexp.png", "lifeexp.pdf", "lifeexp.svg", "lifeexp.html"]})
