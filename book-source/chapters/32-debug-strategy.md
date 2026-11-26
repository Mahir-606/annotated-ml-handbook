@part 8

# How to debug: reading tracebacks and a systematic method

Everyone writes bugs; experienced programmers just find them faster. This chapter teaches a repeatable method: read the error properly, reproduce it small, inspect the state, form a hypothesis, test it. It also covers the tools that make each step quick.

## The debugging loop {B}

:::def
**Bug**: a mistake in code that makes it fail or give wrong results.

**Debugging**: the systematic process of finding and fixing bugs.

**Debugger**: a tool that pauses a running program so you can inspect variables and step through code line by line, such as `pdb` or the debuggers in VS Code and PyCharm.

**Minimal reproducible example**: the shortest complete piece of code that still shows a problem.
:::


1. **Read** the whole error message, bottom line first.
2. **Locate** the line in *your* code where it happened (the last traceback frame that points to your file).
3. **Inspect** the values involved there: their `type`, `shape`, `dtype`, `len`, a few elements, and whether any are `NaN`.
4. **Hypothesise** one specific cause ("`X` is 1-D because I selected one column with single brackets").
5. **Test** the hypothesis with the smallest possible change, then re-run.
6. **Shrink** the problem if you're still stuck: reproduce it in five lines with fake data. Half the time the bug becomes obvious while you do this; the other half you now have a perfect question for a colleague or a forum.

## Anatomy of a traceback {B}

```py error title="A typical failure deep inside a library"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
from sklearn.linear_model import LinearRegression  #> Import `LinearRegression` from scikit-learn's `linear_model` module.

def fit_model(features, target):  #> Define a function `fit_model` that takes `features` and `target`.
    model = LinearRegression()  #> Create the model.
    model.fit(features, target)            #> The library call that fails.
    return model  #> Send `model` back to the caller.

heights = np.array([150, 160, 170, 180])   #> A 1-D array: shape (4,).
weights = np.array([50, 58, 69, 80])  #> Targets.
fit_model(heights, weights)                #> Your top-level call.
```

How to read it:

- **Bottom line** = exception type (`ValueError`) and message. scikit-learn messages often contain the fix ("Reshape your data either using array.reshape(-1, 1)…").
- **Frames** read top-down from your outermost call to where it died. Frames from `<cell>` (or your file names) are *your* code; the "frames inside the library" are scikit-learn's internals. The bug is almost always in the **last frame that is your code**, here `model.fit(features, target)`, because that is where you handed the library something it didn't expect.
- In a full traceback you may see `During handling of the above exception, another exception occurred:` or `The above exception was the direct cause of the following exception:`. Read the **first** exception in the chain for the root cause.

:::tip Search the message, not your variable names
Copy the generic part of the error ("Expected 2D array, got 1D array instead") into a search engine or the search box of this book. Remove your own file paths and variable names first.
:::

## Inspection tools {B}

```py title="The five-second state check"
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
def inspect(name, obj):  #> Define a function `inspect` that takes `name` and `obj`.
    """Print the facts that explain 90% of data bugs."""  #> Docstring.
    info = f"{name}: type={type(obj).__name__}"  #> Always show the type.
    if hasattr(obj, "shape"): info += f" shape={obj.shape}"  #> Add the shape if it has one.
    if hasattr(obj, "dtype"): info += f" dtype={obj.dtype}"  #> Arrays and Series have one dtype...
    elif hasattr(obj, "dtypes"): info += f" dtypes={dict(obj.dtypes.astype(str))}"  #> ...DataFrames have one per column.
    if hasattr(obj, "isna"): info += f" n_missing={int(obj.isna().sum().sum())}"  #> pandas objects: count missing values.
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == "f": info += f" n_nan={int(np.isnan(obj).sum())}"  #> Float arrays: count NaNs.
    print(info)  #> Display the value of `info`.

df = pd.DataFrame({"x": [1.0, None, 3.0], "g": ["a", "b", "a"]})  #> A small frame with one missing value.
inspect("heights", heights)  #> A 1-D array.
inspect("df", df)  #> The whole DataFrame.
inspect("df[['x']]", df[["x"]])  #> Double brackets: a 2-D DataFrame.
inspect("df['x']", df["x"])  #> Single brackets: a 1-D Series.
```

### print debugging, done well {B}

```py title="f-string debugging"
lr, epoch, loss = 0.01, 3, float("nan")  #> Example values, including a NaN loss.
print(f"{lr=} {epoch=} {loss=}")                       #> The `=` specifier prints names and values.
assert not np.isnan(loss) or epoch > 0, "loss is NaN at epoch 0"   #> Turn an assumption into a check that fails loudly.
```

### The interactive debugger {I}

`breakpoint()` pauses execution and opens `pdb`, Python's debugger, at that line. You can inspect and change variables, then continue.

```py norun title="Using breakpoint() and pdb"
def preprocess(df):  #> Define a function `preprocess` that takes `df`.
    cleaned = df.dropna()  #> Drop rows with missing values.
    breakpoint()                     #> Execution stops here and a (Pdb) prompt appears.
    return cleaned / cleaned.max()  #> Send `cleaned / cleaned.max()` back to the caller.

# At the (Pdb) prompt:
#   p cleaned.shape      -> print an expression
#   pp cleaned.dtypes    -> pretty-print
#   l                    -> list the surrounding source code
#   n                    -> execute the next line
#   s                    -> step INTO a function call
#   c                    -> continue running until the next breakpoint
#   u / d                -> move up / down the call stack
#   q                    -> quit
```

:::tip Post-mortem debugging
After an exception in Jupyter, run `%debug` in a new cell to open the debugger **at the point of failure**, with every variable intact. In scripts: `python -m pdb my_script.py`. VS Code and PyCharm offer the same with clickable breakpoints and a variable explorer.
:::

### Warnings are bugs in waiting {I}

```py title="Turn warnings into errors while debugging"
import warnings  #> Import the `warnings` module.
with warnings.catch_warnings():  #> Change warning settings only inside this block.
    warnings.simplefilter("error")                    #> Every warning now raises an exception with a full traceback...
    try:  #> Start a block whose errors we want to catch.
        np.array([1.0, 2.0]) / np.array([0.0, 1.0])   #> ...so a silent divide-by-zero becomes visible.
    except RuntimeWarning as w:  #> Catch the warning, now raised as an exception.
        print("caught:", w)  #> Show its message.
```

## Shrinking the problem {B}

Build the smallest example that still fails. This one technique solves most hard bugs.

```py error title="A minimal reproducible example"
import numpy as np                                   #> Imports only what's needed.
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
X = np.array([[1.0], [2.0], [np.nan]])               #> Three rows of fake data that trigger the same error.
y = np.array([0, 1, 0])  #> Matching labels.
LogisticRegression().fit(X, y)                       #> The one failing call.
```

A good minimal example has fake or tiny data, includes every import, shows the full error, and lists library versions (`pip show scikit-learn`). It's what people on forums and issue trackers need to help you.

## Defensive coding for data science {I}

```py title="Validate data at the boundaries"
def validate(df: pd.DataFrame) -> pd.DataFrame:  #> Takes and returns a DataFrame (returns it unchanged if valid).
    expected = {"age": "int64", "bmi": "float64", "smoker": "str"}  #> The schema we expect: column -> dtype.
    missing_cols = set(expected) - set(df.columns)  #> Expected columns that are absent.
    if missing_cols:  #> Run the indented block only if `missing_cols` is true.
        raise KeyError(f"missing columns: {sorted(missing_cols)}")  #> Fail with the list of missing columns.
    for col, dt in expected.items():  #> Loop over `expected.items()`, naming each item `col, dt`.
        if str(df[col].dtype) != dt:  #> Run the indented block only if `str(df[col].dtype) != dt` is true.
            raise TypeError(f"{col}: expected {dt}, got {df[col].dtype}")  #> Fail with the offending column and both dtypes.
    if not df["age"].between(0, 120).all():  #> Run the indented block only if `not df["age"].between(0, 120).all()` is true.
        raise ValueError(f"age out of range: {df.loc[~df['age'].between(0, 120), 'age'].tolist()[:5]}")  #> Fail and show up to 5 bad values.
    return df  #> Send `df` back to the caller.

ok = validate(pd.read_csv("insurance.csv"))  #> Validate right after loading.
print("validated", ok.shape)  #> Reached only if every check passed.
```

:::research Tests for analysis code
Short `pytest` tests (`def test_scaler_has_zero_mean(): ...`) catch regressions when you refactor. Test the parts that are easy to get subtly wrong: data cleaning rules, custom metrics, splitting logic, and any custom transformer.
:::

:::ex Try it yourself
1. Deliberately cause a `KeyError` in a pandas `groupby`, then use `%debug` (or `python -m pdb`) to inspect the DataFrame's columns at the failure point.
2. Write a minimal reproducible example for the last error you were stuck on.
3. Add a `validate()` function to your own project's loading code.
:::
