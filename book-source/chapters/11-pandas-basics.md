@part 3

# pandas foundations: Series, DataFrames and selection

pandas gives you the **DataFrame**: a table with labelled rows and columns where each column can have its own type. It is how nearly every ML project loads, inspects and prepares data. This book uses **pandas 3**, which changed a few long-standing behaviours. Those changes are flagged where they matter.

## Series and DataFrame {B}

:::def
**Series**: pandas' one-dimensional labelled array: one column of data plus an index of row labels.

**DataFrame**: pandas' two-dimensional labelled table: columns (each a Series, possibly of different types) sharing one row index.

**Index (pandas)**: the row labels of a Series or DataFrame (by default 0, 1, 2, …), used for lookup and for aligning data.
:::


A **Series** is one labelled column. A **DataFrame** is a collection of Series that share the same row labels (the **index**).

```py title="Building both by hand"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd                                   #> The universal alias.

s = pd.Series([4.2, 3.9, 5.1], index=["a", "b", "c"], name="yield")   #> Values + row labels + a name.
print(s)  #> Display the value of `s`.
print(s["b"], s.mean())                               #> Label-based lookup; NumPy-style maths.

df = pd.DataFrame({                                   #> A dict of column name -> list of values.
    "plot": ["P1", "P2", "P3", "P4"],  #> Column "plot": a text label for each field plot.
    "fertilizer": ["A", "B", "A", "B"],  #> Column "fertilizer": the treatment.
    "rain_mm": [120, 95, 140, 80],  #> Column "rain_mm": integers.
    "yield_t": [4.2, 3.9, 5.1, 3.4],  #> Column "yield_t": floats.
})  #> Close the dictionary and the call or tuple opened above.
print(df)  #> Display the value of `df`.
print(df.shape, list(df.columns), df.index)           #> (rows, cols), column names, the default RangeIndex 0..3.
```

## The practice dataset {B}

The rest of Part III uses a realistic synthetic table of 1,000 insurance customers. Generating it yourself means you can reproduce every output in this part exactly.

```py title="Create and save the practice dataset"
rng = np.random.default_rng(2026)                         #> Fixed seed: everyone gets identical data.
n = 1000  #> Number of customers.
age = rng.integers(18, 65, n)                             #> Ages 18..64.
sex = rng.choice(["female", "male"], n)  #> Random sex, about 50/50.
bmi = np.round(rng.normal(30, 6, n).clip(16, 53), 1)      #> Body-mass index, clipped to a plausible range.
children = rng.choice([0, 1, 2, 3, 4], n, p=[.43, .24, .18, .12, .03])  #> Number of children with realistic probabilities.
smoker = rng.choice(["yes", "no"], n, p=[.2, .8])  #> About 20% smokers.
region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)  #> One of four regions.
charges = (2500 + 260 * age + 320 * (bmi - 30).clip(0) + 450 * children  #> Charges grow with age, with BMI above 30 and with children...
           + np.where(smoker == "yes", 23000, 0)          #> Smokers cost much more: a strong signal to discover later.
           + rng.gamma(2.0, 1500, n)).round(2)            #> Right-skewed noise, like real costs.
ins = pd.DataFrame({"age": age, "sex": sex, "bmi": bmi, "children": children,  #> Assemble the columns into a DataFrame...
                    "smoker": smoker, "region": region, "charges": charges})  #> ...continued.
ins.loc[rng.choice(n, 40, replace=False), "bmi"] = np.nan           #> Inject 40 missing BMI values to clean later.
ins.to_csv("insurance.csv", index=False)                  #> Save WITHOUT the index column (a very common mistake otherwise).
print("saved", ins.shape)  #> (1000, 7)
```

## Reading and writing data {B}

```py title="read_csv and friends"
df = pd.read_csv("insurance.csv")              #> Parse a CSV into a DataFrame, inferring column types.
print(df.head())                               #> First 5 rows (`df.head(10)` for 10). `df.tail()` shows the last rows.
```

| Task | Read | Write |
|---|---|---|
| CSV | `pd.read_csv("f.csv", sep=",", na_values=["?", "NA"])` | `df.to_csv("f.csv", index=False)` |
| Excel | `pd.read_excel("f.xlsx", sheet_name="Sheet1")` (needs `openpyxl`) | `df.to_excel("f.xlsx", index=False)` |
| Parquet (fast, typed) | `pd.read_parquet("f.parquet")` (needs `pyarrow`) | `df.to_parquet("f.parquet")` |
| JSON | `pd.read_json("f.json")` | `df.to_json("f.json", orient="records")` |
| SQL | `pd.read_sql("SELECT * FROM t", conn)` | `df.to_sql("t", conn)` |

:::tip Useful `read_csv` arguments
`usecols=[...]` loads only some columns; `dtype={"id": "str"}` forces types; `parse_dates=["date"]` converts date columns; `nrows=1000` previews a huge file; `encoding="latin-1"` fixes `UnicodeDecodeError`; `sep=";"` handles European CSVs; `index_col=0` uses the first column as the index.
:::

## Inspecting a DataFrame {B}

These five calls are the first thing to run on any new dataset.

```py title="First look"
print(df.shape)                       #> (1000, 7): rows, columns.
df.info()                             #> Column names, non-null counts, dtypes and memory. Reveals missing values at a glance.
```

```py title="Summary statistics"
print(df.describe().round(2))                         #> count, mean, std, min, quartiles, max for numeric columns.
print(df.describe(include="str"))                     #> For text columns: count, number of unique values, most frequent (top), its frequency.
```

```py title="Counts and missing values"
print(df["region"].value_counts())                    #> Frequency of each category, sorted.
print(df["smoker"].value_counts(normalize=True))      #> As proportions: check class balance before modelling!
print(df.isna().sum())                                #> Missing values per column.
print(df.nunique())                                   #> Number of distinct values per column.
```

:::note What changed in pandas 3
Text columns now get a dedicated string dtype, shown as `str`, instead of the old catch-all `object`. That is why the call above is `describe(include="str")`. And **Copy-on-Write** is always on: any DataFrame you derive from another behaves like an independent copy. See section 11.7.
:::

## Selecting data {B}

:::def
**`.loc` and `.iloc`**: `.loc[rows, cols]` selects by labels (slices include the end); `.iloc[rows, cols]` selects by integer positions (slices exclude the end).
:::


### Columns

```py title="One column, several columns"
ages = df["age"]                         #> Single brackets + one name -> a Series.
print(type(ages).__name__, ages.shape)  #> Series, shape (1000,).
sub = df[["age", "bmi", "charges"]]      #> Double brackets (a LIST of names) -> a DataFrame.
print(type(sub).__name__, sub.shape)  #> DataFrame, shape (1000, 3).
print(df.age.mean())                     #> Attribute access works for simple names, but df["age"] always works. Prefer it.
```

### Rows: `.loc` (labels) and `.iloc` (positions)

`.loc[rows, cols]` selects by **label**; `.iloc[rows, cols]` selects by **integer position**, like NumPy.

```py title="loc vs iloc"
print(df.loc[0, "charges"])                       #> Row label 0, column "charges".
print(df.loc[0:2, ["age", "smoker"]])             #> .loc slices INCLUDE the end label: rows 0, 1 AND 2.
print(df.iloc[0:2, 0:3])                          #> .iloc slices EXCLUDE the end, like Python: rows 0-1, cols 0-2.
print(df.iloc[-1])                                #> Last row, as a Series.
```

:::warn `.loc` includes the endpoint, `.iloc` does not
`df.loc[0:2]` returns 3 rows; `df.iloc[0:2]` returns 2. After filtering or sorting, row labels no longer match positions. Use `.iloc` when you mean "the first n rows".
:::

### Filtering with conditions

```py title="Boolean filtering"
smokers = df[df["smoker"] == "yes"]                              #> Keep rows where the condition is True.
print(len(smokers))  #> Number of smokers.
older_smokers = df[(df["age"] >= 50) & (df["smoker"] == "yes")]  #> Combine with & | ~ and PARENTHESES, as in NumPy.
print(len(older_smokers))  #> Smokers aged 50 or more.
south = df[df["region"].isin(["southeast", "southwest"])]        #> Membership in a list of values.
print(south["region"].unique())  #> Only the two southern regions remain.
print(df.loc[df["bmi"] > 45, ["age", "bmi"]].head(3))            #> Filter rows AND choose columns in one .loc.
print(df.query("age < 25 and children >= 2").shape)              #> `query` takes a readable string expression.
limit = 40  #> A Python variable to use inside query().
print(df.query("bmi > @limit").shape)                            #> `@name` refers to a Python variable inside query.
```

## Adding, changing and removing columns {B}

```py title="Creating new columns"
df = pd.read_csv("insurance.csv")  #> Reload the original data.
df["charges_k"] = df["charges"] / 1000                          #> Vectorized arithmetic creates a new column.
df["is_smoker"] = (df["smoker"] == "yes").astype(int)           #> Boolean -> 0/1: a numeric feature.
df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 64],      #> Bin a numeric column into labelled intervals.
                         labels=["18-30", "31-45", "46-64"])  #> Names for the three bins.
df = df.assign(log_charges=np.log(df["charges"]))               #> `assign` returns a new DataFrame: good for chaining.
df.loc[df["children"] >= 3, "big_family"] = True                #> Conditional assignment; other rows get NaN.
df = df.rename(columns={"bmi": "BMI"})                          #> Rename columns.
df = df.drop(columns=["big_family"])                            #> Remove a column (axis=1). Rows: df.drop(index=[...]).
print(df.head(3))  #> The first three rows with the new columns.
```

### Sorting

```py title="sort_values and nlargest"
print(df.sort_values("charges", ascending=False).head(3)[["age", "smoker", "charges"]])   #> Most expensive first.
print(df.sort_values(["smoker", "age"], ascending=[True, False]).head(3)[["smoker", "age"]])  #> Sort by several keys.
print(df.nlargest(3, "BMI")[["BMI", "age"]])                  #> Shortcut for the top-n rows by a column.
```

## Copy-on-Write and the "chained assignment" trap {I}

:::def
**Copy-on-Write**: pandas 3 behaviour in which any DataFrame derived from another acts as an independent copy; the data is only copied when one of them is modified.

**Chained assignment**: two indexing operations followed by `=`, as in `df[mask]["col"] = v`. In pandas 3 it never modifies the original; use one `.loc` call instead.
:::


In pandas 3, every new DataFrame derived from another (by filtering, selecting or slicing) behaves like an independent copy. One consequence: **chained assignment**, meaning two indexing operations in a row followed by `=`, never modifies the original.

```py warn title="This does NOT change df"
df = pd.read_csv("insurance.csv")  #> Reload the original data.
df[df["age"] > 60]["bmi"] = 0          #> Chained: df[...] makes a new object, and ["bmi"] = 0 modifies only that temporary.
print((df["bmi"] == 0).sum())          #> Still 0 rows changed: the original df is untouched.
```

```py title="The correct way: one .loc call"
df.loc[df["age"] > 60, "bmi"] = 0      #> Row condition and column in ONE indexing call modifies df itself.
print((df["bmi"] == 0).sum())  #> Count rows whose BMI is now 0.
df = pd.read_csv("insurance.csv")      #> Reload the clean data.
young = df[df["age"] < 30].copy()      #> Want a separate subset to edit? Take it explicitly (the .copy() makes intent clear).
young["bmi"] = young["bmi"].round()    #> Editing the subset never affects df.
```

:::bug `SettingWithCopyWarning` / `ChainedAssignmentError`
Older pandas printed `SettingWithCopyWarning`; pandas 3 warns about chained assignment instead, and the assignment has no effect on the original. The fix is always the same: use a single `df.loc[row_condition, "column"] = value`.
:::

## The index {I}

```py title="Setting and resetting the index"
idx = df.set_index("region")                 #> Move a column into the row labels.
print(idx.loc["southwest"].shape)            #> Fast selection of all rows with that label.
print(idx.reset_index().columns.tolist()[:3])   #> `reset_index` moves the index back into a column.
r = df[df["age"] > 60]  #> Customers over 60.
print(r.index[:5].tolist())                  #> Filtering keeps the ORIGINAL labels...
print(r.reset_index(drop=True).index[:5].tolist())   #> ...reset_index(drop=True) renumbers 0..n-1 and discards the old labels.
```

:::bug Rows misalign after combining a filtered DataFrame with something else
pandas aligns operations by **index label**, not by position. If you filtered or sorted a DataFrame and then assign a NumPy array or another Series, labels may not line up and you get `NaN`s or shuffled values. Use `.reset_index(drop=True)` on both, or assign `.to_numpy()` when you mean "by position".
:::

## From pandas to NumPy and back {B}

```py title="Getting arrays out for modelling"
X = df[["age", "bmi", "children"]].to_numpy()     #> A 2-D NumPy array (n_samples, n_features).
y = df["charges"].to_numpy()                       #> A 1-D target array.
print(X.shape, X.dtype, y[:3])  #> Feature matrix shape and type, and the first three targets.
back = pd.DataFrame(X[:2], columns=["age", "bmi", "children"])   #> Wrap an array back into a labelled table.
print(back)  #> Display the value of `back`.
```

scikit-learn accepts DataFrames directly and remembers the column names, so in practice you often skip `to_numpy()` entirely.

:::ex Try it yourself
1. How many female smokers in the southeast are over 40?
2. Add a column `bmi_class` with "under" (< 18.5), "normal" (< 25), "over" (< 30), "obese" (≥ 30) using `pd.cut`.
3. Show the 5 youngest customers with 4 children, sorted by charges.
:::
