# Cleaning data with pandas

Data scientists often say most of a project is spent cleaning data, and that is no exaggeration. Models are only as good as their input. This chapter is a toolbox for the problems real datasets have: missing values, duplicates, wrong types, messy text, inconsistent categories, dates stored as strings, and outliers.

## A deliberately messy dataset {B}

:::def
**Data cleaning**: finding and fixing problems in raw data, such as wrong types, inconsistent labels, duplicates, missing and impossible values, before analysis.
:::


```py title="Read messy CSV text directly"
import io  #> Import the standard-library `io` module (in-memory files).
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.

raw = """id,Name ,age,income,city,signup_date,score
1, Alice ,34,"52,000",new york,2025-01-15,88
2,bob,,61000,New York,2025/02/03,92
3,Carol,29,?,Boston,15-03-2025,
4,dave,41,48000,boston ,2025-04-22,75
4,dave,41,48000,boston ,2025-04-22,75
5,Eve,230,70000,Chicago,2025-05-30,81
6,frank,38,,CHICAGO,2025-06-11,-1
"""                                                         #> A small CSV with nearly every common problem.
df = pd.read_csv(io.StringIO(raw), na_values=["?"])        #> io.StringIO makes a string behave like a file; "?" becomes NaN.
print(df)  #> Display the value of `df`.
print(df.dtypes)  #> Display the value of `df.dtypes`.
```

Spot the problems: a column name with a trailing space, names with odd capitalisation and spaces, a missing age, an income stored as text with a thousands separator, inconsistent city spellings, three date formats, a duplicated row, an impossible age (230), and a score of −1 that probably means "missing".

## Step 1: tidy column names {B}

```py title="Normalise headers"
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")   #> Strip spaces, lowercase, snake_case.
print(df.columns.tolist())  #> ['id', 'name', 'age', 'income', 'city', 'signup_date', 'score']
```

## Step 2: duplicates {B}

```py title="Find and drop duplicate rows"
print(df.duplicated().sum())                   #> Count rows that exactly repeat an earlier row.
print(df[df.duplicated(keep=False)])           #> keep=False marks ALL copies, so you can inspect them.
df = df.drop_duplicates().reset_index(drop=True)   #> Keep the first occurrence; renumber the index.
print(df["id"].is_unique, len(df))             #> Check that the ID column is now unique.
```

:::research Check duplicates on IDs too
Exact-duplicate rows are easy. Harder: the same subject appearing twice with slightly different values, or the same patient in both your training and test sets. `df["id"].duplicated().sum()` catches the first; [[ch:sklearn-model-selection]] handles the second with grouped splits.
:::

## Step 3: fix types {B}

```py title="Text to numbers"
df["income"] = pd.to_numeric(df["income"].str.replace(",", ""), errors="coerce")   #> Remove separators, then convert; errors="coerce" turns unparseable values into NaN.
print(df["income"].tolist(), df["income"].dtype)  #> Now real numbers (float64), with NaN where values were missing.
df["score"] = df["score"].replace(-1, np.nan)        #> A sentinel value (-1 for "missing") must become a real NaN.
print(df["score"].tolist())  #> The -1 is now NaN.
```

```py title="Mixed date formats: the tempting shortcut is wrong"
guess = pd.to_datetime(df["signup_date"], format="mixed", dayfirst=True)   #> Parse each value separately, treating ambiguous dates as day-first...
print(guess.dt.strftime("%Y-%m-%d").tolist())                          #> ...but look: "2025/02/03" became 2 March and "2025-06-11" became 6 November!
```

`dayfirst=True` was applied even to the year-first ISO dates. Silent errors like this are worse than crashes. The safe approach is to parse each **known** format explicitly and combine the results:

```py title="Parse each known format explicitly"
d = df["signup_date"]  #> A short name for the raw date strings.
parsed = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")                     #> ISO dates; anything else becomes NaT (missing time).
parsed = parsed.fillna(pd.to_datetime(d, format="%Y/%m/%d", errors="coerce"))     #> Fill the gaps with a second format...
parsed = parsed.fillna(pd.to_datetime(d, format="%d-%m-%Y", errors="coerce"))     #> ...and a third.
print(parsed.isna().sum(), "unparsed")                                            #> Always check that nothing was left behind.
df["signup_date"] = parsed  #> Replace the text column with real datetimes.
print(df["signup_date"].dt.strftime("%Y-%m-%d").tolist())  #> Every date is now correct and in one format.
print(df["signup_date"].dt.month.tolist(), df["signup_date"].dt.day_name().tolist()[:3])   #> The .dt accessor extracts parts of dates.
```

:::warn Ambiguous dates
`03/04/2025` is 3 April in most of the world and 4 March in the US. Pass an explicit `format=` whenever you know it, count the values that failed to parse, and check a few parsed dates by eye.
:::

## Step 4: clean text and categories {B}

```py title="The .str accessor"
df["name"] = df["name"].str.strip().str.title()       #> Remove surrounding spaces; capitalise each word.
df["city"] = df["city"].str.strip().str.title()       #> "boston ", "CHICAGO" -> "Boston", "Chicago".
print(df[["name", "city"]])  #> The tidied text columns.
print(df["city"].value_counts())  #> Three cities instead of six spellings.
print(df["name"].str.contains("a", case=False).tolist())   #> Pattern tests (regex by default) give boolean masks.
print(df["name"].str.len().tolist(), df["name"].str[0].tolist())   #> String length; first character.
```

Map inconsistent labels to a canonical set with a dictionary:

```py title="Recoding values"
s = pd.Series(["M", "male", "F", "female", "f", "unknown"])  #> Inconsistent codes for the same two categories.
mapping = {"m": "male", "male": "male", "f": "female", "female": "female"}  #> Lowercase spelling -> canonical label.
print(s.str.lower().map(mapping).tolist())           #> `map` with a dict: anything not in the dict becomes NaN.
print(s.replace({"M": "male", "F": "female"}).tolist())   #> `replace` only changes listed values and leaves the rest alone.
```

## Step 5: missing values {B}

:::def
**Missing value**: an absent observation, shown as `NaN`, `None` or `NaT` (for dates) in pandas.

**Imputation**: replacing missing values with estimates, such as the mean, the median, the most frequent value, or a model's prediction.
:::


First measure, then decide. There are three broad options: **drop**, **fill (impute)**, or **flag**.

```py title="Measure missingness"
print(df.isna().sum())                      #> Missing count per column.
print(df.isna().mean().round(2))            #> Missing fraction per column.
print(df[df.isna().any(axis=1)])            #> Rows with at least one missing value.
```

```py title="Drop or fill"
print(df.dropna().shape)                               #> Drop any row with a missing value: often far too aggressive.
print(df.dropna(subset=["age"]).shape)                 #> Drop only rows missing a critical column.
filled = df.copy()  #> Work on a copy so `df` keeps its missing values for comparison.
filled["age"] = filled["age"].fillna(filled["age"].median())    #> Median imputation: robust to outliers.
filled["income"] = filled["income"].fillna(filled.groupby("city")["income"].transform("median"))   #> Fill with the median of the same city.
filled["score_missing"] = filled["score"].isna().astype(int)    #> Flag missingness: it can be informative in itself.
filled["score"] = filled["score"].fillna(filled["score"].mean())  #> Mean imputation for score (after flagging which were missing).
print(filled[["age", "income", "score", "score_missing"]])  #> The filled columns and the missingness flag.
```

```py title="Ordered data: forward fill and interpolation"
ts = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])  #> An ordered series with gaps.
print(ts.ffill().tolist())          #> Carry the last known value forward (sensor readings, prices).
print(ts.interpolate().tolist())    #> Linear interpolation between known points.
```

:::research Impute inside the training data only
Computing a fill value (mean, median) on the full dataset before splitting leaks test-set information into training. In ML work, impute with scikit-learn's `SimpleImputer` inside a `Pipeline` ([[ch:sklearn-preprocessing]]) so it is fitted on training folds only. Pandas filling is fine for exploration and for decisions that use no statistics, such as "missing means zero".
:::

## Step 6: outliers and impossible values {I}

:::def
**Outlier**: an observation far from most others. It may be an error or a genuine but rare case.

**IQR (interquartile range)**: Q3 − Q1, the spread of the middle 50% of the data. Values beyond 1.5 × IQR outside the quartiles are commonly flagged as outliers.
:::


```py title="Domain rules and the IQR rule"
print(df[(df["age"] < 0) | (df["age"] > 110)])                  #> Domain knowledge first: an age of 230 is a data-entry error.
df.loc[df["age"] > 110, "age"] = np.nan                         #> Treat impossible values as missing rather than guessing.
x = pd.Series([10, 12, 11, 13, 12, 95, 11, 10])  #> One obvious outlier: 95.
q1, q3 = x.quantile([0.25, 0.75])                               #> First and third quartiles.
iqr = q3 - q1  #> Interquartile range: the spread of the middle 50%.
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr                         #> Tukey's fences: the rule behind boxplot whiskers.
print(f"bounds=({lo}, {hi}) outliers={x[(x < lo) | (x > hi)].tolist()}")  #> The fences and the values outside them.
print(x.clip(lower=lo, upper=hi).tolist())                      #> *Winsorising*: cap extreme values instead of deleting them.
```

:::warn Don't delete outliers by reflex
An outlier may be an error, or it may be the most interesting observation you have. Remove values that are *impossible* (an age of 230); investigate values that are merely *unusual*; and report what you did and why.
:::

## Transforming columns with map, apply and vectorized ops {I}

```py title="Fastest to slowest"
ins = pd.read_csv("insurance.csv")  #> Load the practice dataset.
ins["bmi_sq"] = ins["bmi"] ** 2                                         #> 1. Vectorized: always try this first.
ins["obese"] = np.where(ins["bmi"] >= 30, "yes", "no")                  #> 2. np.where for two-way choices.
ins["risk"] = np.select(                                                 #> 3. np.select for several conditions, checked in order.
    [ins["smoker"].eq("yes") & (ins["bmi"] >= 30), ins["smoker"].eq("yes")],  #> The conditions, checked in order: smoker AND obese; then smoker.
    ["high", "medium"], default="low")  #> The value for each condition; rows matching none get "low".
ins["region_code"] = ins["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})   #> 4. map with a dict.
ins["desc"] = ins.apply(lambda r: f"{r['age']}y {r['sex']}", axis=1)   #> 5. Row-wise apply: flexible but slow. Last resort.
print(ins[["bmi", "obese", "smoker", "risk", "region_code", "desc"]].head())  #> The new columns side by side.
print(ins["risk"].value_counts())  #> How many customers fall into each risk group.
```

## Categorical dtype {I}

:::def
**Categorical variable**: a variable taking one of a limited set of labels, such as region or blood type. **Ordinal** categories have a natural order (S < M < L); **nominal** ones don't.
:::


```py title="Memory-efficient, ordered categories"
ins["region"] = ins["region"].astype("category")                 #> Stores each label once plus small integer codes.
print(ins["region"].cat.categories.tolist(), ins["region"].cat.codes[:5].tolist())  #> The category labels and the integer code stored for each row.
size = pd.Categorical(["M", "S", "L", "M"], categories=["S", "M", "L"], ordered=True)   #> An ORDERED category: S < M < L.
print(size.sort_values().tolist(), (size > "S").tolist())  #> Sorting and comparisons follow the declared order S < M < L.
dummies = pd.get_dummies(ins["region"], prefix="reg", dtype=int)  #> One-hot encode a category: one 0/1 column per level.
print(dummies.head(3))  #> Four 0/1 columns, one per region.
```

## A reusable cleaning function {I}

Collect every step into a function, so cleaning is documented, repeatable and identical for new data.

```py title="clean_customers()"
def parse_dates(s, formats=("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y")):  #> Define a function `parse_dates` that takes `s`, `formats`, `"%Y/%m/%d"` and `"%d-%m-%Y")`.
    out = pd.Series(pd.NaT, index=s.index)                                   #> Start with all-missing dates.
    for fmt in formats:  #> Loop over `formats`, naming each item `fmt`.
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))    #> Try each known format on the remaining gaps.
    return out  #> Send `out` back to the caller.

def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:  #> Takes a raw DataFrame and returns a cleaned one.
    """Return a cleaned copy of the raw customer table."""  #> Docstring.
    return (raw  #> Send `(raw` back to the caller.
            .rename(columns=lambda c: c.strip().lower().replace(" ", "_"))   #> Tidy headers.
            .drop_duplicates()                                               #> Remove exact duplicates.
            .assign(                                                         #> Each keyword creates or replaces a column.
                name=lambda d: d["name"].str.strip().str.title(),  #> Tidy names.
                city=lambda d: d["city"].str.strip().str.title(),  #> Tidy cities.
                income=lambda d: pd.to_numeric(d["income"].astype("str").str.replace(",", ""), errors="coerce"),  #> Text incomes -> numbers (astype("str") first, in case the column was read as numbers).
                age=lambda d: d["age"].where(d["age"].between(0, 110)),     #> `where` keeps values meeting the condition, else NaN.
                score=lambda d: d["score"].replace(-1, np.nan),  #> Sentinel -1 -> NaN.
                signup_date=lambda d: parse_dates(d["signup_date"]),  #> Parse the dates with the helper above.
            )  #> Close the call or tuple opened above.
            .reset_index(drop=True))  #> Renumber rows 0..n-1 and close the chain.

clean = clean_customers(pd.read_csv(io.StringIO(raw), na_values=["?"]))  #> Run the whole cleaning recipe in one call.
print(clean)  #> Display the value of `clean`.
print(clean.dtypes)  #> Display the value of `clean.dtypes`.
```

:::tip Method chaining
Wrapping a chain in parentheses lets you put one operation per line. Using `lambda d:` inside `assign` refers to the DataFrame *at that point in the chain*, so later steps see the results of earlier ones.
:::

:::ex Try it yourself
1. In `insurance.csv`, fill missing BMI with the median BMI of the same sex and smoker group (`groupby(...).transform("median")`).
2. Count how many charges are outliers by the IQR rule. Are they mostly smokers?
3. Write a check function that raises an error if any age is outside 18–64 or any charge is negative.
:::
