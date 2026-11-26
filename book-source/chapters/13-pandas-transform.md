# Grouping, reshaping, joining and time series

Clean data still needs to be summarised, reshaped and combined before it answers questions. This chapter covers the split-apply-combine pattern (`groupby`), pivot tables, wide/long reshaping, merging tables, and time-series tools.

## groupby: split, apply, combine {B}

:::def
**GroupBy (split-apply-combine)**: splitting rows into groups by key columns, applying a function such as a mean to each group, and combining the results into a new table.

**Aggregation**: reducing many values to one summary, such as a sum, mean, count or maximum.
:::


`groupby` **splits** rows into groups by one or more keys, **applies** a function to each group, and **combines** the results.

```py title="Group summaries"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.

ins = pd.read_csv("insurance.csv")  #> Load the practice dataset.
print(ins.groupby("smoker")["charges"].mean().round(0))           #> Mean charges for each smoker value.
print(ins.groupby(["region", "smoker"])["charges"].median().round(0).head(6))   #> Two keys -> a MultiIndex result.
print(ins.groupby("region").size())                               #> Number of rows per group.
```

### Several statistics at once with agg {B}

```py title="Named aggregation"
summary = ins.groupby("smoker").agg(                         #> Each keyword = output column name = (input column, function).
    n=("charges", "size"),  #> Number of rows in each group.
    mean_charges=("charges", "mean"),  #> Mean of charges.
    sd_charges=("charges", "std"),  #> Standard deviation of charges.
    median_bmi=("bmi", "median"),  #> Median BMI.
    max_age=("age", "max"),  #> Oldest age.
).round(1)  #> Close agg() and round every result to 1 decimal.
print(summary)  #> Display the value of `summary`.
print(ins.groupby("sex")[["age", "bmi"]].agg(["mean", "min", "max"]).round(1))   #> Several functions on several columns.
```

### transform and filter {I}

`agg` returns one row per group. `transform` returns a result **aligned with the original rows**, which is ideal for group-wise features.

```py title="Group-wise features"
ins["region_mean"] = ins.groupby("region")["charges"].transform("mean")    #> Each row gets its region's mean.
ins["vs_region"] = ins["charges"] / ins["region_mean"]                      #> How expensive relative to the region.
ins["rank_in_region"] = ins.groupby("region")["charges"].rank(ascending=False)  #> 1 = most expensive customer within the region.
ins["bmi_z_by_sex"] = ins.groupby("sex")["bmi"].transform(lambda s: (s - s.mean()) / s.std())   #> z-score within each group.
print(ins[["region", "charges", "region_mean", "vs_region", "rank_in_region"]].head().round(2))  #> The new group-wise columns.
big = ins.groupby("children").filter(lambda g: len(g) > 100)              #> Keep only groups meeting a condition.
print(sorted(big["children"].unique()))  #> Only groups with more than 100 rows survive.
```

## Pivot tables and crosstabs {B}

:::def
**Pivot table**: a table summarising one variable by two others: rows from one key, columns from another, cells holding an aggregate.

**Crosstab (contingency table)**: a table of counts for every combination of two categorical variables.
:::


```py title="Two-way summaries"
pt = ins.pivot_table(values="charges", index="region", columns="smoker",  #> Rows = regions, columns = smoker values, cells = mean charges...
                     aggfunc="mean", margins=True).round(0)        #> margins=True adds an "All" row and column.
print(pt)  #> Display the value of `pt`.
print(pd.crosstab(ins["region"], ins["smoker"]))                   #> Counts of each combination.
print(pd.crosstab(ins["sex"], ins["smoker"], normalize="index").round(3))   #> Row proportions: smoking rate by sex.
```

## Reshaping: wide ↔ long {I}

:::def
**Wide vs long format**: wide data has one row per subject with repeated measurements in separate columns; long (tidy) data has one row per measurement.
:::


Many plotting and modelling tools want **long** ("tidy") data: one row per observation, one column per variable.

```py title="melt and pivot"
wide = pd.DataFrame({"subject": ["s1", "s2"],  #> Wide format: one row per subject...
                     "week1": [5.1, 6.0], "week2": [5.5, 6.2], "week3": [5.9, 6.1]})  #> ...and one column per week.
long = wide.melt(id_vars="subject", var_name="week", value_name="score")   #> Wide -> long: one row per subject-week.
print(long)  #> Display the value of `long`.
back = long.pivot(index="subject", columns="week", values="score")        #> Long -> wide again.
print(back)  #> Display the value of `back`.
print(back.stack().head(3))                                                #> `stack` also turns columns into rows.
```

:::bug `ValueError: Index contains duplicate entries, cannot reshape`
`pivot` requires each (index, column) pair to be unique. If there are duplicates, decide how to combine them and use `pivot_table(..., aggfunc="mean")` instead.
:::

## Combining tables {B}

### concat: stack tables {B}

```py title="Stacking rows or columns"
a = pd.DataFrame({"id": [1, 2], "x": [10, 20]})  #> First table.
b = pd.DataFrame({"id": [3], "x": [30]})  #> Second table with the same columns.
print(pd.concat([a, b], ignore_index=True))                    #> Stack rows; ignore_index renumbers 0..n-1.
print(pd.concat([a, pd.DataFrame({"y": [7, 8]})], axis=1))     #> Side by side (aligned on index!).
```

### merge: database-style joins {B}

:::def
**Merge / join**: combining two tables by matching values in key columns. Inner, left, right and outer joins differ in which unmatched rows they keep.
:::


```py title="Joining on a key"
patients = pd.DataFrame({"pid": [1, 2, 3, 4], "age": [34, 51, 29, 62]})  #> One row per patient.
visits = pd.DataFrame({"pid": [1, 1, 2, 5], "bp": [120, 118, 135, 128]})  #> Visits: patient 1 twice; patient 5 isn't in `patients`.
print(pd.merge(patients, visits, on="pid", how="inner"))   #> Only keys present in BOTH tables.
print(pd.merge(patients, visits, on="pid", how="left"))    #> All patients; visits where available, else NaN.
print(pd.merge(patients, visits, on="pid", how="outer", indicator=True))   #> Everything; _merge column shows each row's source.
```

| `how=` | Keeps | Use when |
|---|---|---|
| `"inner"` | keys in both tables | you only want complete matches |
| `"left"` | every row of the left table | enriching a main table with extra info |
| `"right"` | every row of the right table | rarely; swap the tables and use left |
| `"outer"` | keys in either table | auditing what matched and what didn't |

```py title="Guarding against silent row explosions"
check = pd.merge(patients, visits, on="pid", how="left", validate="one_to_many")   #> Raises an error if pid is not unique in `patients`.
print(len(patients), "->", len(check), "rows after merge")                         #> Row count grew: one patient had two visits.
```

:::bug Row count jumps after a merge
If the key is duplicated in both tables, each match multiplies (a many-to-many join). Always compare `len()` before and after, and pass `validate="one_to_one"`, `"one_to_many"` or `"many_to_one"` to make pandas check your assumption.
:::

## Time series {I}

:::def
**Time series**: observations recorded in time order, usually at regular intervals.

**Resampling**: changing the frequency of a time series, such as daily → monthly (down-sampling with aggregation).

**Rolling window**: a calculation (often a mean) over the most recent k observations, recomputed at every time step.

**Lag**: a previous value of a series used as a feature, such as yesterday's sales for predicting today's.
:::


```py title="Dates as the index"
rng = np.random.default_rng(1)  #> Seeded generator.
dates = pd.date_range("2025-01-01", periods=120, freq="D")          #> 120 consecutive days.
sales = pd.Series(100 + np.arange(120) * 0.8 + rng.normal(0, 8, 120)  #> Synthetic sales: upward trend plus noise...
                  + 15 * np.sin(np.arange(120) / 7 * 2 * np.pi), index=dates, name="sales")   #> Trend + weekly cycle + noise.
print(sales.head(3))  #> The index is now dates.
print(sales.loc["2025-02"].shape)                                   #> Partial-string indexing: all of February.
monthly = sales.resample("ME").agg(["mean", "sum"]).round(1)        #> Down-sample to month-end ("ME") summaries.
print(monthly)  #> Display the value of `monthly`.
```

```py title="Rolling windows, lags and differences" caption="A 7-day rolling mean smooths out the weekly cycle to reveal the trend."
import matplotlib.pyplot as plt  #> Import Matplotlib's plotting interface under the short name `plt`.
feat = sales.to_frame()  #> Turn the Series into a one-column DataFrame.
feat["roll7"] = sales.rolling(window=7).mean()        #> Mean of the current and previous 6 days (the first 6 rows are NaN).
feat["lag1"] = sales.shift(1)                         #> Yesterday's value: a classic forecasting feature.
feat["diff1"] = sales.diff()                          #> Day-over-day change.
feat["pct"] = sales.pct_change()                      #> Relative change.
print(feat.iloc[6:10].round(2))  #> Rows 6-9, where the 7-day window is complete.
ax = feat[["sales", "roll7"]].plot(figsize=(8, 3.4), title="Daily sales and 7-day rolling mean")   #> pandas' .plot wraps Matplotlib.
ax.set_ylabel("units")  #> Label the y-axis.
plt.show()  #> Draw the figure (in a script this opens a window; in Jupyter it appears below the cell).
```

:::warn Time-series features must only look backwards
`shift(1)` and `rolling()` use past values, which is correct. Never use `shift(-1)` or centred windows to build features for predicting the future. That is leakage, and for time series the train/test split must also respect time order ([[ch:sklearn-model-selection]]).
:::

## Method chaining: a full mini-analysis {I}

```py title="From raw file to report table in one readable chain"
report = (pd.read_csv("insurance.csv")  #> Start the chain by loading the file.
          .dropna(subset=["bmi"])                                         #> Drop the rows missing BMI.
          .assign(obese=lambda d: d["bmi"] >= 30,                         #> Add flags.
                  age_band=lambda d: pd.cut(d["age"], [17, 35, 50, 64]))  #> Three age bands.
          .groupby(["age_band", "smoker"], observed=True)                 #> observed=True: only combinations that occur.
          .agg(n=("charges", "size"), median_charges=("charges", "median"),  #> Count and median charges per group...
               obese_rate=("obese", "mean"))  #> ...and the share of obese customers (mean of True/False).
          .round(2)  #> Round all numbers.
          .reset_index())  #> Turn the group keys back into columns; close the chain.
print(report)  #> Display the value of `report`.
```

:::ex Try it yourself
1. For each region, find the share of customers who are smokers and the median charge for smokers vs non-smokers in one table.
2. Make a long-format table of `age`, `bmi` and `charges` with `melt`, then compute each variable's mean with `groupby`.
3. Resample `sales` to weekly sums and plot them.
:::
