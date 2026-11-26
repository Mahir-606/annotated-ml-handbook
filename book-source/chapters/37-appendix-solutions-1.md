@kind appendix

# Solutions: Parts I–IV

Worked solutions to every "Try it yourself" exercise in Parts I to IV. Try each exercise yourself first: struggling for a few minutes is where the learning happens. Every solution was run when this book was built, so the outputs are real. There is usually more than one correct answer; if yours produces the same result, it is right.

## [[ch:python-basics]]: Values, variables and types

```py title="Exercise 1: area of a circle"
import math                                    #> For the constant pi.
r = 2.5                                        #> The radius.
area = math.pi * r ** 2                        #> Area = π r².
print(f"area = {area:.3f}")                    #> `.3f` = 3 decimal places.
```

```py title="Exercise 2: normalise an e-mail address"
email = "  Ada.Lovelace@Example.COM "          #> Messy input with spaces and mixed case.
print(email.strip().lower())                   #> Methods chain left to right: strip spaces, then lowercase.
```

```py title="Exercise 3: predict, then check"
print(7 // -2)                                 #> Floor division rounds toward minus infinity: -3.5 becomes -4.
print(bool("False"))                           #> Any non-empty string is truthy, even "False": True.
print("3" * 3)                                 #> Multiplying a string repeats it: "333".
```

## [[ch:python-collections]]: Data structures

```py title="Exercise 1: unique lowercase words"
words = ["Data", "science", "IS", "fun", "data"]   #> The input list.
print({w.lower() for w in words})                  #> A set comprehension lowercases and removes duplicates.
```

```py title="Exercise 2: fizz dictionary"
fizz = {n: "fizz" if n % 3 == 0 else n for n in range(1, 11)}   #> Key n; value "fizz" or n itself.
print(fizz)                                                    #> {1: 1, 2: 2, 3: 'fizz', ...}
```

```py title="Exercise 3: a mutable list inside an immutable tuple"
t = ([1, 2], 3)                        #> A tuple whose first item is a list.
t[0].append(9)                         #> We are not changing the tuple: it still holds the same list object...
print(t)                               #> ...we are changing the list, which is mutable.
```

The tuple stores *references* to its items. Which objects it refers to can never change, but a mutable object it refers to can still change internally.

## [[ch:python-control]]: Control flow

```py title="Exercise 1: FizzBuzz"
out = []                                       #> Collect results to print on one line.
for n in range(1, 31):                         #> 1 to 30.
    if n % 15 == 0:                            #> Divisible by both 3 and 5: check this first!
        out.append("FizzBuzz")                 #> Add the word.
    elif n % 3 == 0:                           #> Divisible by 3 only.
        out.append("Fizz")                     #> Add the word.
    elif n % 5 == 0:                           #> Divisible by 5 only.
        out.append("Buzz")                     #> Add the word.
    else:                                      #> Neither.
        out.append(str(n))                     #> The number itself, as text.
print(" ".join(out))                           #> One line, separated by spaces.
```

```py title="Exercise 2: dot product with zip"
a, b = [1, 2, 3], [4, 5, 6]                    #> Two vectors.
total = 0                                      #> Running sum.
for x, y in zip(a, b):                         #> Pairs (1, 4), (2, 5), (3, 6).
    total += x * y                             #> Add each product.
print(total)                                   #> 4 + 10 + 18 = 32.
```

```py title="Exercise 3: halving until below 0.01"
lr, halvings = 1.0, 0                          #> Start value and counter.
while lr >= 0.01:                              #> Keep going while still too big.
    lr /= 2                                    #> Halve it.
    halvings += 1                              #> Count.
print(halvings, lr)                            #> 7 halvings: 1/128 = 0.0078.
```

## [[ch:python-functions]]: Functions

```py title="Exercise 1: describe()"
def describe(values):
    """Return basic summary statistics of a list of numbers."""   #> Docstring.
    return {"n": len(values), "mean": sum(values) / len(values),  #> Count and mean...
            "min": min(values), "max": max(values)}               #> ...smallest and largest.

print(describe([4, 8, 15, 16, 23, 42]))                           #> Try it.
```

```py title="Exercise 2: a call-counting decorator"
from functools import wraps                    #> Keeps the wrapped function's name.

def count_calls(func):
    count = 0                                  #> Lives in the closure: remembered between calls.
    @wraps(func)
    def wrapper(*args, **kwargs):              #> Accept any arguments.
        nonlocal count                         #> Let the inner function modify the enclosing variable.
        count += 1                             #> One more call.
        print(f"{func.__name__} called {count} time(s)")   #> Report.
        return func(*args, **kwargs)           #> Run the original.
    return wrapper                             #> The decorated function.

@count_calls
def greet(name):
    return f"hi {name}"                        #> The original behaviour.

greet("Ada"); greet("Alan")                    #> Two calls.
```

```py title="Exercise 3: running mean generator"
def running_mean(xs):
    total = 0                                  #> Sum so far.
    for i, x in enumerate(xs, start=1):        #> i counts the items seen.
        total += x                             #> Update the sum.
        yield total / i                        #> Hand out the current mean, then pause.

print(list(running_mean([2, 4, 6, 8])))        #> [2.0, 3.0, 4.0, 5.0]
```

## [[ch:python-oop]]: Classes and objects

```py title="Exercise 1: a Counter class"
class Counter:
    def __init__(self):
        self.counts = {}                       #> item -> count.
    def add(self, item):
        self.counts[item] = self.counts.get(item, 0) + 1   #> Start at 0 if unseen, then add 1.
        return self                            #> Allow chaining.
    def most_common(self):
        return max(self.counts, key=self.counts.get)       #> The key with the largest count.

c = Counter().add("a").add("b").add("a")       #> Chained adds.
print(c.most_common(), c.counts)               #> 'a' appears twice.
```

```py title="Exercise 2: __len__ for Dataset"
class Dataset:
    def __init__(self, name, values):
        self.name, self.values = name, list(values)   #> Store attributes.
    def __len__(self):
        return len(self.values)                       #> len(ds) now calls this.

ds = Dataset("heights", [1.6, 1.7, 1.8])              #> An instance.
print(len(ds))                                        #> 3
```

```py title="Exercise 3: MedianRegressor"
import statistics                                     #> For median().

class Model:
    def fit(self, X, y):
        self.fitted = True                            #> Mark as fitted.
        return self                                   #> Chainable.

class MedianRegressor(Model):
    def fit(self, X, y):
        self.median_ = statistics.median(y)           #> Learn the training median.
        return super().fit(X, y)                      #> Run the parent's fit too.
    def predict(self, X):
        return [self.median_] * len(X)                #> Predict the median for every input.

m = MedianRegressor().fit([[0]] * 5, [1, 2, 3, 100, 4])   #> The median ignores the outlier 100.
print(m.predict([[0], [0]]), m.fitted)                    #> [3, 3] True
```

## [[ch:python-errors-files]]: Errors, files and modules

```py title="Exercise 1: read_numbers() skipping bad lines"
from pathlib import Path                               #> Paths.
Path("nums.txt").write_text("1.5\n2\nn/a\n\n3.25\n")    #> A test file with a bad line and a blank line.

def read_numbers(path):
    numbers = []                                       #> Results.
    with open(path, encoding="utf-8") as f:            #> Open safely.
        for line in f:                                 #> One line at a time.
            try:
                numbers.append(float(line))            #> float() also tolerates the trailing newline.
            except ValueError:                         #> "n/a" and "" can't be converted...
                continue                               #> ...so skip them.
    return numbers                                     #> The clean list.

print(read_numbers("nums.txt"))                        #> [1.5, 2.0, 3.25]
```

```py title="Exercise 2: save and load a config"
import json                                            #> JSON support.
cfg = {"lr": 0.001, "epochs": 50, "layers": [64, 32]}  #> Hyperparameters.
Path("config.json").write_text(json.dumps(cfg, indent=2))   #> Save as readable JSON.
loaded = json.loads(Path("config.json").read_text())   #> Load it back.
print(loaded == cfg, loaded["layers"])                 #> Identical: True.
```

```py title="Exercise 3: count CSV files recursively"
for sub in ["tree/a", "tree/b/c"]:                     #> Build a small folder tree.
    Path(sub).mkdir(parents=True, exist_ok=True)       #> Create folders.
for f in ["tree/x.csv", "tree/a/y.csv", "tree/b/c/z.csv", "tree/b/notes.txt"]:   #> Some files.
    Path(f).write_text("")                             #> Empty files.
print(len(list(Path("tree").glob("**/*.csv"))))        #> ** searches every sub-folder: 3.
```

## [[ch:python-toolkit]]: The Pythonic toolkit

```py title="Exercise 1: the most common words"
from collections import Counter                       #> Counting tool.
text = "the model learns the data and the data teaches the model"   #> A paragraph.
print(Counter(text.lower().split()).most_common(3))   #> Lowercase, split on whitespace, count, top 3.
```

```py title="Exercise 2: all 12 combinations"
from itertools import product                         #> Cartesian product.
combos = list(product([0.1, 1, 10], ["linear", "rbf", "poly", "sigmoid"]))   #> Every (C, kernel) pair.
print(len(combos), combos[:4])                        #> 3 x 4 = 12.
```

```py title="Exercise 3: extract e-mail addresses"
import re                                             #> Regular expressions.
s = "Contact ada@lab.org or alan.t@uni.ac.uk; not me@ or @bad.com"   #> Some valid and invalid addresses.
print(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", s))   #> name@domain.tld, with one or more dot-separated parts.
```

## [[ch:numpy-arrays]]: NumPy arrays

```py title="Exercise 1: the centre 3x3 block"
import numpy as np                                    #> NumPy.
A = np.arange(1, 26).reshape(5, 5)                    #> 1..25 as a 5x5 grid.
print(A[1:4, 1:4])                                    #> Rows 1-3 and columns 1-3.
```

```py title="Exercise 2: fraction beyond 2 standard deviations"
x = np.random.default_rng(0).normal(size=1000)       #> 1000 standard-normal values.
z = (x - x.mean()) / x.std()                          #> Standardise.
print((np.abs(z) > 2).mean())                         #> The mean of a boolean array is the fraction True: about 0.05.
```

```py title="Exercise 3: sums along each axis of a 3-D array"
B = np.arange(24).reshape(2, 3, 4)                    #> Shape (2, 3, 4).
for ax in range(3):                                   #> Each axis in turn.
    print(ax, B.sum(axis=ax).shape)                   #> The summed axis disappears: (3,4), (2,4), (2,3).
```

## [[ch:numpy-compute]]: Computing with NumPy

```py title="Exercise 1: min-max scaling with broadcasting"
rng = np.random.default_rng(1)                         #> Seeded.
M = rng.normal(5, 3, (100, 4))                         #> A random (100, 4) matrix.
scaled = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0))   #> (100,4) op (4,) broadcasts per column.
print(scaled.min(axis=0), scaled.max(axis=0))          #> Every column now spans exactly 0 to 1.
```

```py title="Exercise 2: class proportions"
labels = rng.choice([0, 1], 500, p=[.8, .2])           #> Imbalanced labels.
vals, counts = np.unique(labels, return_counts=True)   #> Distinct values and how often each occurs.
print(dict(zip(vals.tolist(), (counts / counts.sum()).round(3).tolist())))   #> Proportions near 0.8 and 0.2.
```

```py title="Exercise 3: a linear system"
A = np.array([[3, 2], [1, -1]])                        #> Coefficients of 3x + 2y and x - y.
b = np.array([7, -1])                                  #> Right-hand sides.
print(np.linalg.solve(A, b))                           #> x = 1, y = 2.
```

## [[ch:numpy-ml-scratch]]: Machine learning from scratch

```py title="Exercise 1: L2-regularised gradient descent"
rng = np.random.default_rng(7)                         #> Same data recipe as the chapter.
X = rng.uniform(0, 10, (200, 1))                       #> One feature.
y = 2.5 * X[:, 0] - 4 + rng.normal(0, 2, 200)          #> True slope 2.5.
Xb = np.column_stack([np.ones(200), X])                #> Add the intercept column.

def gd_ridge(alpha, lr=0.02, epochs=3000):
    w = np.zeros(2)                                    #> [intercept, slope].
    for _ in range(epochs):
        grad = 2 / len(y) * Xb.T @ (Xb @ w - y)        #> MSE gradient.
        grad[1:] += 2 * alpha * w[1:]                  #> L2 penalty gradient, excluding the intercept.
        w -= lr * grad                                 #> Step.
    return w                                           #> Learned weights.

for alpha in [0, 0.1, 1, 10]:                          #> Increasing penalty.
    print(alpha, gd_ridge(alpha).round(3))             #> The slope shrinks toward 0 as alpha grows.
```

```py title="Exercise 2: overlapping classes"
def sigmoid(z):
    return 1 / (1 + np.exp(-z))                        #> Probability squashing.

for centre in [1.5, 0.5]:                              #> Far-apart vs close classes.
    Xc = np.vstack([rng.normal(-centre, 1, (150, 2)), rng.normal(centre, 1, (150, 2))])   #> Two clouds.
    yc = np.r_[np.zeros(150), np.ones(150)]            #> Labels.
    Xcb = np.column_stack([np.ones(300), Xc])          #> Intercept column.
    w = np.zeros(3)                                    #> Start at zero.
    for _ in range(1000):
        w -= 0.5 * Xcb.T @ (sigmoid(Xcb @ w) - yc) / 300   #> Log-loss gradient step.
    print(f"centres ±{centre}: accuracy = {np.mean((sigmoid(Xcb @ w) >= 0.5) == yc):.3f}")   #> Lower when the clouds overlap.
```

Accuracy falls because the two clouds overlap heavily: many points sit on the "wrong" side of any straight line. No model can separate them perfectly; the problem itself is harder (the Bayes error is higher).

```py title="Exercise 3: multiclass kNN with bincount"
def knn_predict(X_train, y_train, X_new, k=5):
    d = np.sqrt(((X_new[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))   #> Distance matrix.
    nearest = np.argsort(d, axis=1)[:, :k]                                 #> Indices of the k nearest.
    votes = y_train[nearest]                                               #> Their labels, shape (n_new, k).
    return np.array([np.bincount(row).argmax() for row in votes])          #> Most frequent label in each row.

from sklearn.datasets import load_iris                 #> Three classes.
Xi, yi = load_iris(return_X_y=True)                    #> Features and labels.
perm = rng.permutation(150)                            #> Shuffle.
tr, te = perm[:120], perm[120:]                        #> 120 train, 30 test.
print("accuracy:", np.mean(knn_predict(Xi[tr], yi[tr], Xi[te], k=5) == yi[te]))   #> Works for 3 classes.
```

## [[ch:pandas-basics]]: pandas foundations

```py title="Exercises 1-3"
import pandas as pd                                    #> pandas.
df = pd.read_csv("insurance.csv")                      #> The practice data.
q1 = df[(df["sex"] == "female") & (df["smoker"] == "yes") & (df["region"] == "southeast") & (df["age"] > 40)]   #> Four conditions combined with &.
print("Exercise 1:", len(q1))                          #> The count.
df["bmi_class"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, np.inf],   #> Bin edges; np.inf catches everything above 30.
                         labels=["under", "normal", "over", "obese"], right=False)   #> right=False makes bins [a, b), so 30 counts as obese.
print("Exercise 2:", df["bmi_class"].value_counts().to_dict())         #> How many in each class.
q3 = df[df["children"] == 4].nsmallest(5, "age").sort_values("charges")   #> 5 youngest with 4 children, then sort by charges.
print("Exercise 3:\n", q3[["age", "children", "charges"]])             #> The result.
```

## [[ch:pandas-cleaning]]: Cleaning data

```py title="Exercise 1: group-wise median imputation"
df = pd.read_csv("insurance.csv")                       #> Fresh copy.
print("missing before:", df["bmi"].isna().sum())         #> 40 missing.
df["bmi"] = df["bmi"].fillna(df.groupby(["sex", "smoker"])["bmi"].transform("median"))   #> Each gap gets its group's median.
print("missing after:", df["bmi"].isna().sum())          #> 0.
```

```py title="Exercise 2: IQR outliers in charges"
q1, q3 = df["charges"].quantile([0.25, 0.75])           #> Quartiles.
iqr = q3 - q1                                           #> Interquartile range.
out = df[(df["charges"] < q1 - 1.5 * iqr) | (df["charges"] > q3 + 1.5 * iqr)]   #> Beyond Tukey's fences.
print(len(out), "outliers;", out["smoker"].value_counts().to_dict())   #> Who are they?
```

Most of the high-charge outliers are smokers. They are not errors: they reflect a real, large effect of smoking, which is a reason not to delete outliers blindly.

```py title="Exercise 3: a validation function"
def check_insurance(d):
    bad_age = d[~d["age"].between(18, 64)]                   #> Rows with age outside 18-64.
    if len(bad_age):
        raise ValueError(f"{len(bad_age)} ages outside 18-64, e.g. {bad_age['age'].head(3).tolist()}")   #> Clear message.
    if (d["charges"] < 0).any():                             #> Any negative charge?
        raise ValueError("negative charges found")           #> Fail loudly.
    return True                                              #> All good.

print(check_insurance(df))                                    #> Passes on the real data.
try:
    check_insurance(pd.DataFrame({"age": [30, 70], "charges": [100, 200]}))   #> A bad table.
except ValueError as e:
    print("caught:", e)                                       #> The check works.
```

## [[ch:pandas-transform]]: Grouping, reshaping, joining

```py title="Exercise 1: smoking share and median charges by region"
smoke_share = df.groupby("region")["smoker"].apply(lambda s: (s == "yes").mean()).rename("smoker_share")   #> Share of "yes" per region.
med = df.pivot_table(values="charges", index="region", columns="smoker", aggfunc="median")   #> Median charges, smokers vs non-smokers.
med.columns = ["median_nonsmoker", "median_smoker"]      #> Columns were "no", "yes" (alphabetical).
print(pd.concat([smoke_share, med], axis=1).round(2))    #> One table, aligned on region.
```

```py title="Exercise 2: melt, then group"
long = df.melt(value_vars=["age", "bmi", "charges"], var_name="variable", value_name="value")   #> Three columns -> one long column.
print(long.groupby("variable")["value"].mean().round(2))   #> One mean per original column.
```

```py title="Exercise 3: weekly resampling" caption="The chapter's simulated daily sales summed by week. Weekly totals smooth out the day-to-day noise."
import matplotlib.pyplot as plt                          #> Plotting.
rng = np.random.default_rng(1)                           #> Same recipe as the chapter.
dates = pd.date_range("2025-01-01", periods=120, freq="D")   #> 120 days.
sales = pd.Series(100 + np.arange(120) * 0.8 + rng.normal(0, 8, 120) + 15 * np.sin(np.arange(120) / 7 * 2 * np.pi), index=dates)   #> Daily sales.
weekly = sales.resample("W").sum()                       #> "W" = weekly bins ending on Sunday.
ax = weekly.plot(marker="o", figsize=(7, 3.2), title="Weekly sales")   #> Plot.
ax.set_ylabel("units per week")                          #> Label.
plt.show()                                               #> Draw.
```

## [[ch:viz-matplotlib]]: Matplotlib and Seaborn basics

```py title="Exercise 1: charges vs log(charges)" caption="Raw charges are strongly right-skewed with two humps (smokers vs non-smokers); the log makes the shape more symmetric, though still not normal."
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))        #> Two panels.
axes[0].hist(df["charges"], bins=40); axes[0].set_title("charges")             #> Raw.
axes[1].hist(np.log(df["charges"]), bins=40, color="#e0245e"); axes[1].set_title("log(charges)")   #> Log-transformed.
plt.tight_layout(); plt.show()                           #> Draw.
```

```py title="Exercises 2 and 3: region pair plot and bars with CIs" caption="Left: no region-related clustering is visible. Right: mean charges by number of children, with 95% confidence intervals; the intervals overlap heavily."
import seaborn as sns                                    #> Seaborn.
g = sns.pairplot(df.sample(300, random_state=0), vars=["age", "bmi", "charges"], hue="region", corner=True, height=2, plot_kws={"s": 10})   #> Colour by region.
plt.show()                                               #> Draw the pair plot.
fig, ax = plt.subplots(figsize=(6, 3.4))                 #> New figure.
sns.barplot(data=df, x="children", y="charges", ax=ax)   #> Mean per group with a 95% CI (the default).
plt.show()                                               #> Draw.
```

Region doesn't matter much: the regional colours are thoroughly mixed in every panel. Smoking is what separates the data.

## [[ch:viz-mpl-gallery]]: The Matplotlib gallery

```py title="Exercise 1: grouped bars of mean charges" caption="Smokers' mean charges dwarf non-smokers' in every region."
m = df.pivot_table(values="charges", index="region", columns="smoker", aggfunc="mean")   #> Rows = regions, columns = no/yes.
x = np.arange(len(m)); w = 0.4                           #> Positions and bar width.
fig, ax = plt.subplots(figsize=(7, 3.4))                 #> New figure.
ax.bar(x - w / 2, m["no"], w, label="non-smoker")        #> Left bars.
ax.bar(x + w / 2, m["yes"], w, label="smoker")           #> Right bars.
ax.set(xticks=x, xticklabels=m.index, ylabel="mean charges")   #> Region names at group centres.
ax.legend(); plt.show()                                  #> Legend and draw.
```

```py title="Exercise 2: ECDF of BMI by sex" caption="The two curves nearly coincide: the medians (where they cross 0.5) are very close."
fig, ax = plt.subplots(figsize=(7, 3.2))                 #> New figure.
for sex in ["female", "male"]:                           #> Two groups.
    ax.ecdf(df.loc[df["sex"] == sex, "bmi"].dropna(), label=sex)   #> ECDF ignores nothing, so drop NaNs first.
ax.axhline(0.5, ls=":", color="grey"); ax.legend(); plt.show()     #> Median line, legend, draw.
print(df.groupby("sex")["bmi"].median())                 #> The exact medians.
```

```py title="Exercise 3: 600-dpi PNG vs PDF"
from pathlib import Path                                 #> File sizes.
fig, ax = plt.subplots(figsize=(3.5, 2.6))               #> A small chart.
ax.plot(range(10), np.arange(10) ** 2, marker="o")       #> Content.
fig.savefig("ex600.png", dpi=600); fig.savefig("ex600.pdf"); plt.close(fig)   #> Save both; close.
print({f: Path(f).stat().st_size // 1024 for f in ["ex600.png", "ex600.pdf"]})   #> Sizes in KB: the vector PDF is much smaller.
```

## [[ch:viz-seaborn-gallery]]: The Seaborn gallery

```py title="Exercise 1: violins by children and smoker" caption="catplot(kind='violin') with one column per smoker group."
g = sns.catplot(data=df, x="children", y="charges", col="smoker", kind="violin", height=3, aspect=1.3)   #> Faceted violins.
plt.show()                                               #> Draw.
```

```py title="Exercise 2: jointplot with a regression line" caption="BMI vs charges: a weak positive slope with enormous scatter, because smoking drives most of the variation."
g = sns.jointplot(data=df, x="bmi", y="charges", kind="reg", height=4, scatter_kws={"s": 8, "alpha": 0.4})   #> Scatter + fit + marginals.
plt.show()                                               #> Draw.
```

The fitted line rises slightly: higher BMI goes with higher charges on average, but the spread around the line is huge, so BMI alone explains little.

```py title="Exercise 3: clustermap with smokers marked" caption="Rows reorder into a smoker block (orange strip) with high charges, showing that charges dominate the clustering."
sub = df.dropna().sample(80, random_state=0)             #> 80 complete rows.
num = sub[["age", "bmi", "children", "charges"]]         #> Numeric columns.
g = sns.clustermap(num, standard_scale=1, row_colors=sub["smoker"].map({"yes": "#ff6a00", "no": "#1f4fff"}),   #> Orange = smoker.
                   cmap="mako", figsize=(5.5, 5.5), yticklabels=False)   #> Hide row labels.
plt.show()                                               #> Draw.
```

## [[ch:viz-plotly]]: Interactive charts with Plotly

```py title="Exercises 1 and 2: animated bars and a three-level sunburst" caption="Left (web edition: press Play): life expectancy of the five most populous countries over time. Right: tips split by day, time and sex."
import plotly.express as px                              #> Plotly Express.
gap = px.data.gapminder()                                #> Built-in data.
top5 = gap[gap["year"] == 2007].nlargest(5, "pop")["country"]   #> The five most populous in 2007.
fig = px.bar(gap[gap["country"].isin(top5)], x="country", y="lifeExp", color="country",   #> One bar per country...
             animation_frame="year", range_y=[0, 85], width=700, height=420,   #> ...animated by year, fixed y-range.
             title="Life expectancy of the five most populous countries")   #> Title.
fig.show()                                               #> Display.
fig2 = px.sunburst(px.data.tips(), path=["day", "time", "sex"], values="tip", width=520, height=480)   #> Three levels from the centre outwards.
fig2.show()                                              #> Display.
```

```py title="Exercise 3: a 7-inch, 300-dpi-equivalent PNG"
fig2.write_image("sunburst.png", width=700, height=600, scale=3)   #> 700 px x 3 = 2100 px = 7 in at 300 dpi.
from PIL import Image                                    #> Check the pixel size.
with Image.open("sunburst.png") as im:                   #> Open (and close automatically).
    print(im.size, "->", im.size[0] / 300, "inches wide at 300 dpi")   #> (2100, 1800) -> 7.0 inches.
```

## [[ch:viz-export]]: Figures for papers

```py title="Exercises 1-3: final-size export and checking pixels"
plt.rcParams.update({"font.size": 8})                    #> 8-pt fonts.

def save_figure(fig, name, folder="figures", formats=("pdf", "png"), dpi=300):
    out = Path(folder); out.mkdir(parents=True, exist_ok=True)   #> Make the folder.
    paths = [out / f"{name}.{f}" for f in formats]               #> One path per format.
    for p in paths:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")             #> Save each.
    return paths                                                 #> Return them.

fig, axes = plt.subplots(1, 3, figsize=(7, 3), layout="constrained")   #> 7 x 3 in, three panels.
axes[0].hist(df["age"], bins=20); axes[0].set_xlabel("Age (years)")    #> a: age distribution.
axes[1].scatter(df["age"], df["charges"] / 1000, s=3, c=(df["smoker"] == "yes"), cmap="coolwarm")   #> b: age vs charges coloured by smoker.
axes[1].set_xlabel("Age (years)"); axes[1].set_ylabel("Charges (k$)")  #> Labels.
df.groupby("smoker")["charges"].median().div(1000).plot.bar(ax=axes[2], color=["#1f4fff", "#ff6a00"])   #> c: median charges.
axes[2].set_ylabel("Median charges (k$)")                #> Label.
for ax, letter in zip(axes, "abc"):                      #> Panel letters.
    ax.text(-0.25, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)   #> Top-left of each panel.
paths = save_figure(fig, "three_panel")                  #> PDF + PNG.
plt.close(fig)                                           #> Close.
with Image.open(paths[1]) as im:                         #> The PNG.
    print(im.size, "pixels ≈", round(im.size[0] / 300, 2), "x", round(im.size[1] / 300, 2), "inches at 300 dpi")   #> Width ≈ 7 in (bbox_inches='tight' trims a little).
plt.rcdefaults()                                         #> Restore defaults.
```
