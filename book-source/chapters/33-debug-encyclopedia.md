# The error encyclopedia

Every error below was produced by actually running the code shown, so the message is exactly what you will see (with your own names and numbers). Use the search box with the error name or a phrase from the message to jump straight here. Each entry gives the **cause** and the **fix**.

## Python errors {B}

### `SyntaxError` and `IndentationError`

```py error title="Missing colon"
code = "if x > 1\n    print(x)"          #> We compile a string so the book can show the error; in a file you'd see it on load.
compile(code, "<example>", "exec")  #> Ask Python to parse the string as code.
```

**Cause:** Python can't parse the code: a missing `:` after `if/for/def/class`, unbalanced brackets or quotes, or `=` where `==` was meant. The caret `^` marks where parsing failed; the real mistake is often on the line **before**. **Fix:** check the end of the previous line for missing `:` or `)`. For `IndentationError`, make every block line up with 4 spaces and never mix tabs and spaces.

### `NameError: name '…' is not defined`

```py error title="Typo or never defined"
learning_rate = 0.01  #> Defined with the correct spelling...
print(learnig_rate)  #> Display the value of `learnig_rate`.
```

**Cause:** a typo, using a variable before assigning it, forgetting an import (`np` without `import numpy as np`), or, in notebooks, running cells out of order or restarting the kernel. **Fix:** check spelling (Python suggests near-matches), import what you use, and in Jupyter use *Restart & Run All* to confirm cells work top to bottom.

### `TypeError: unsupported operand type(s)` / `can only concatenate str`

```py error title="Mixing types"
total = "5" + 3  #> A string plus an integer.
```

**Cause:** an operation between incompatible types, typically numbers read from a file or `input()` as strings. **Fix:** convert explicitly (`int("5") + 3`, `f"{x} items"`); for DataFrame columns, `pd.to_numeric(df[col], errors="coerce")`.

### `TypeError: 'X' object is not callable`

```py error title="A built-in shadowed by a variable"
list = [1, 2, 3]            #> Overwrites the built-in `list` function.
list((4, 5))  #> `list` is now a list object, not the built-in function.
```

**Cause:** you put `()` after something that is not a function, very often because a variable named `list`, `sum`, `max`, `str` or `type` replaced the built-in, or because you wrote `df.shape()` instead of `df.shape`. **Fix:** rename the variable (`del list` restores the built-in), or drop the parentheses on attributes.

### `TypeError: … takes N positional arguments but M were given`

```py error title="Forgot self, or passed too many arguments"
class Scaler:  #> Define a class `Scaler`.
    def fit(X):                  #> Missing `self`.
        return X  #> Send `X` back to the caller.
Scaler().fit([1, 2])  #> Python passes the object as the first argument, plus [1, 2]: two arguments for one parameter.
```

**Cause:** wrong number of arguments; in methods, usually a missing `self`. **Fix:** add `self` as the first parameter of every method; check the function's signature with `help(func)`.

### `IndexError: list index out of range`

```py error title="Index past the end"
scores = [0.8, 0.9, 0.85]  #> Three items: valid indices are 0, 1, 2.
scores[3]  #> Index 3 doesn't exist.
```

**Cause:** indexing at or beyond `len(x)` (valid indices are `0` to `len(x)-1`), or into an empty list. **Fix:** check `len()`; use `x[-1]` for the last element; guard with `if x:` for possibly empty lists.

### `KeyError`

```py error title="Missing dictionary key"
config = {"lr": 0.01, "epochs": 10}  #> No "batch_size" key.
config["batch_size"]  #> Missing key.
```

**Cause:** the key doesn't exist; in pandas, the column name is misspelt, has stray spaces or different capitalisation. **Fix:** `config.get("batch_size", 32)`; print `list(d.keys())` or `df.columns.tolist()`; clean names with `df.columns = df.columns.str.strip()`.

### `AttributeError: '…' object has no attribute '…'`

```py error title="Method on the wrong type, or on None"
data = [3, 1, 2]  #> A list.
result = data.sort()          #> .sort() returns None...
result.append(4)              #> ...so `result` is None here.
```

**Cause:** calling a method the object doesn't have. Classic causes: a function returned `None` (in-place methods like `list.sort()`), using a NumPy method name on a list, or a misspelt attribute. **Fix:** print `type(obj)`; remember that in-place methods return `None`; check spelling with `dir(obj)`.

### `ValueError: invalid literal for int() with base 10`

```py error title="Converting a non-numeric string"
int("3.7")  #> A decimal string is not a valid integer.
```

**Cause:** the string isn't a valid integer (decimals, spaces, commas, "N/A"). **Fix:** `int(float("3.7"))`; strip characters first; use `pd.to_numeric(..., errors="coerce")` for columns.

### `ZeroDivisionError`

```py error title="Division by zero"
correct, total = 0, 0  #> Nothing was evaluated.
accuracy = correct / total  #> 0 / 0.
```

**Cause:** dividing by zero, often an empty group or empty dataset. **Fix:** guard (`acc = correct / total if total else float("nan")`) and ask *why* the denominator is zero. NumPy returns `inf`/`nan` with a warning instead of raising.

### `ModuleNotFoundError: No module named '…'`

```py error title="Library not installed in this environment"
import sklearn_extra_magic  #> Import the `sklearn_extra_magic` module.
```

**Cause:** the package isn't installed in the interpreter that is running your code, the import name differs from the pip name (`pip install scikit-learn`, `import sklearn`; `pip install opencv-python`, `import cv2`), or a local file shadows it. **Fix:** `python -m pip install <package>` with the same Python; `%pip install` in Jupyter; check `import sys; print(sys.executable)`.

### `FileNotFoundError: [Errno 2] No such file or directory`

```py error title="Wrong path or working directory"
open("data/raw/survey.csv")  #> Relative to the current working directory, which may not contain this path.
```

**Cause:** the path is relative to the **current working directory**, which may not be where your script or notebook lives. **Fix:** print `Path.cwd()`; build paths from the script location: `Path(__file__).parent / "data" / "survey.csv"`; check spelling and extension.

### `RecursionError: maximum recursion depth exceeded`

```py error title="Recursion without a base case"
def countdown(n):  #> Define a function `countdown` that takes `n`.
    return countdown(n - 1)          #> Never stops.
countdown(5)  #> Each call makes another call, forever.
```

**Cause:** a function calls itself forever (missing or unreachable base case), or a property getter that refers to itself. **Fix:** add a base case; in properties, store data in `self._x` rather than `self.x`.

## NumPy errors {B}

### `ValueError: operands could not be broadcast together with shapes`

```py error title="Incompatible shapes"
import numpy as np  #> Import NumPy (the array library) under the short name `np`.
X = np.ones((100, 3))  #> Shape (100, 3).
weights = np.ones(100)  #> Shape (100,), compared with X's last dimension (3).
X * weights  #> 100 vs 3: incompatible.
```

**Cause:** shapes aren't compatible under broadcasting (compared from the right, each pair must be equal or contain a 1). **Fix:** print both shapes; reshape with `weights[:, None]` (now `(100, 1)`), or transpose. See [[ch:numpy-compute]].

### `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`

```py error title="Matrix product with incompatible inner dimensions"
A = np.ones((4, 3))  #> Shape (4, 3).
B = np.ones((4, 2))  #> Shape (4, 2).
A @ B  #> Needs A.shape[1] == B.shape[0]: 3 vs 4.
```

**Cause:** for `A @ B`, `A.shape[1]` must equal `B.shape[0]`. **Fix:** check shapes; usually one side needs `.T`.

### `ValueError: cannot reshape array of size N into shape …`

```py error title="Total size mismatch"
np.arange(10).reshape(4, 3)  #> 10 elements into 12 slots.
```

**Cause:** the new shape's product differs from the number of elements. **Fix:** use `-1` for one dimension (`reshape(-1, 1)`); check that the size really is what you think.

### `ValueError: The truth value of an array with more than one element is ambiguous`

```py error title="Using `if` or `and` with an array"
a = np.array([1, 5, 3])  #> An array.
if a > 2:  #> Run the indented block only if `a > 2` is true.
    print("big")  #> Display the value of `"big"`.
```

**Cause:** Python needs one True/False, but the comparison gives an array of them. **Fix:** say what you mean: `(a > 2).any()`, `(a > 2).all()`, or element-wise `&`/`|` with parentheses for combining conditions. The same error appears with pandas Series (`if df["x"] > 0`).

### `IndexError: too many indices for array`

```py error title="Indexing a 1-D array as if it were 2-D"
v = np.array([1, 2, 3])  #> 1-D, shape (3,).
v[:, 0]  #> Asks for a second dimension that doesn't exist.
```

**Cause:** more indices than dimensions. **Fix:** check `.ndim` and `.shape`; reshape to 2-D if needed.

### `numpy.exceptions.AxisError: axis 1 is out of bounds`

```py error title="Axis beyond ndim"
np.array([1, 2, 3]).sum(axis=1)  #> A 1-D array has no axis 1.
```

**Cause:** asking for an axis the array doesn't have. **Fix:** a 1-D array has only `axis=0`.

## pandas errors {B}

### `KeyError` on a column

```py error title="Column name mismatch"
import pandas as pd  #> Import pandas (the table library) under the short name `pd`.
df = pd.DataFrame({"Age ": [30, 40], "Income": [50, 60]})   #> Note the trailing space in "Age ".
df["Age"]  #> "Age" (no space) isn't a column.
```

**Cause:** the column doesn't exist under that exact name (spaces, case, typos), or you meant a row label. **Fix:** print `df.columns.tolist()`; normalise with `df.columns = df.columns.str.strip().str.lower()`.

### `ValueError: Length of values does not match length of index`

```py error title="Assigning a list of the wrong length"
df = pd.DataFrame({"a": [1, 2, 3]})  #> Three rows.
df["b"] = [10, 20]  #> Only two values.
```

**Cause:** a new column's values must have one entry per row. **Fix:** check lengths; to add per-group results, use `groupby(...).transform(...)`, which returns a result aligned with the original rows.

### `ChainedAssignmentError` warning: the assignment has no effect

```py warn title="Chained indexing with assignment"
df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 0]})  #> A small frame.
df[df["a"] > 1]["b"] = 99             #> Two indexing steps, then `=`.
print(df)                             #> Unchanged!
```

**Cause:** under pandas 3's Copy-on-Write, `df[mask]` creates a new object; setting a value on it never reaches `df`. **Fix:** one `.loc` call: `df.loc[df["a"] > 1, "b"] = 99`. (Older pandas showed `SettingWithCopyWarning` for the same pattern.)

### `TypeError: Cannot perform reduction 'mean' with string dtype`

```py error title="Numbers stored as text"
df = pd.DataFrame({"price": ["1,200", "950", "1,050"]})  #> Prices stored as text.
df["price"].mean()  #> You can't average strings.
```

**Cause:** the column holds strings (check `df.dtypes`), often because of thousands separators, units, or a stray "N/A". **Fix:** `pd.to_numeric(df["price"].str.replace(",", ""), errors="coerce")`, or `pd.read_csv(..., thousands=",", na_values=["N/A"])`.

### `MergeError` / unexpected row counts after `merge`

```py error title="validate catches a many-to-many join"
left = pd.DataFrame({"id": [1, 1, 2], "x": [10, 11, 20]})  #> id 1 appears twice.
right = pd.DataFrame({"id": [1, 1, 2], "y": ["a", "b", "c"]})  #> id 1 appears twice here too.
pd.merge(left, right, on="id", validate="one_to_one")  #> We claim keys are unique on both sides; pandas checks and refuses.
```

**Cause:** duplicate keys on one or both sides multiply rows. **Fix:** deduplicate keys (`right.drop_duplicates("id")`), aggregate first, or choose the correct `validate=` relationship and keep it as a guard.

### `ValueError: Index contains duplicate entries, cannot reshape`

```py error title="pivot with duplicates"
long = pd.DataFrame({"subject": ["s1", "s1"], "week": ["w1", "w1"], "score": [5, 6]})  #> Two rows for the same subject and week.
long.pivot(index="subject", columns="week", values="score")  #> Which of 5 or 6 should go in the cell?
```

**Cause:** two rows share the same (index, column) pair, so `pivot` can't decide which value to use. **Fix:** `pivot_table(..., aggfunc="mean")`, or remove the duplicates.

### `UnicodeDecodeError` when reading a CSV

```py error title="Wrong text encoding"
from pathlib import Path  #> Import `Path` from `pathlib`.
Path("latin.csv").write_bytes("name\nJosé\n".encode("latin-1"))   #> A file saved in a non-UTF-8 encoding.
pd.read_csv("latin.csv")  #> pandas assumes UTF-8 and meets the Latin-1 byte for é.
```

**Cause:** the file isn't UTF-8 (common with files exported from Excel on Windows). **Fix:** `pd.read_csv("file.csv", encoding="latin-1")` (or `"cp1252"`); if unsure, try `encoding_errors="replace"` to load and inspect.

## scikit-learn errors {I}

### `ValueError: Expected a 2-dimensional container but got <class 'pandas.Series'> instead`

```py error title="One feature passed as a Series"
from sklearn.linear_model import LinearRegression  #> Import `LinearRegression` from scikit-learn's `linear_model` module.
df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.1, 5.9]})  #> One feature and a target.
LinearRegression().fit(df["x"], df["y"])  #> `df["x"]` is a 1-D Series.
```

**Cause:** `X` must be 2-D, `(n_samples, n_features)`. A single column selected with `df["x"]` is 1-D. (With a NumPy array the message reads *Expected 2D array, got 1D array instead*.) **Fix:** `df[["x"]]` (double brackets) or `x.reshape(-1, 1)`.

### `ValueError: Input X contains NaN`

```py error title="Missing values reach an estimator that can't handle them"
from sklearn.linear_model import LogisticRegression  #> Import `LogisticRegression` from scikit-learn's `linear_model` module.
X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 0.5]])  #> One NaN.
LogisticRegression().fit(X, [0, 1, 0])  #> LogisticRegression doesn't accept NaN.
```

**Cause:** most estimators reject NaN (also `inf`). **Fix:** add `SimpleImputer` to the pipeline, or use an estimator that supports missing values (`HistGradientBoosting*`). Find the culprits with `df.isna().sum()` and `np.isinf(X).sum()`.

### `ValueError: could not convert string to float`

```py error title="Text categories passed to a numeric model"
X = pd.DataFrame({"age": [30, 40, 50], "city": ["Oslo", "Lima", "Oslo"]})  #> A text column.
LogisticRegression().fit(X, [0, 1, 0])  #> LogisticRegression doesn't accept NaN.
```

**Cause:** categorical text columns weren't encoded. **Fix:** a `ColumnTransformer` with `OneHotEncoder` for categorical columns ([[ch:sklearn-preprocessing]]).

### `ValueError: Found input variables with inconsistent numbers of samples`

```py error title="X and y lengths differ"
X = np.random.default_rng(0).normal(size=(100, 3))  #> 100 rows.
y = np.zeros(98)  #> 98 labels.
LinearRegression().fit(X, y)  #> Lengths differ.
```

**Cause:** `X` and `y` have different numbers of rows: rows dropped from one but not the other (e.g. `dropna()` on `X` only), or mismatched splits. **Fix:** drop missing rows from the combined DataFrame **before** separating `X` and `y`; assert `len(X) == len(y)`.

### `NotFittedError: This … instance is not fitted yet`

```py error title="predict before fit"
from sklearn.preprocessing import StandardScaler  #> Import `StandardScaler` from scikit-learn's `preprocessing` module.
StandardScaler().transform([[1.0], [2.0]])  #> transform() on a scaler that was never fitted.
```

**Cause:** calling `predict`/`transform` on an unfitted estimator, often a fresh copy (`clone`) or a different variable than the one you fitted. **Fix:** call `fit` first, and use the fitted object; in pipelines, fit the pipeline, not the individual steps.

### `ValueError: Found unknown categories … during transform`

```py error title="A category in the test data that training never saw"
from sklearn.preprocessing import OneHotEncoder  #> Import `OneHotEncoder` from scikit-learn's `preprocessing` module.
enc = OneHotEncoder().fit(pd.DataFrame({"city": ["Oslo", "Lima"]}))  #> Learns two categories.
enc.transform(pd.DataFrame({"city": ["Pune"]}))  #> "Pune" was never seen.
```

**Cause:** the encoder raises on unseen categories by default. **Fix:** `OneHotEncoder(handle_unknown="ignore")` (unseen categories become all zeros), or group rare categories with `min_frequency=`.

### `ValueError: The feature names should match those that were passed during fit`

```py error title="Columns renamed or reordered after fitting"
model = LinearRegression().fit(pd.DataFrame({"a": [1.0, 2, 3], "b": [1.0, 0, 1]}), [1.0, 2, 3])  #> Fitted with columns a, b.
model.predict(pd.DataFrame({"b": [1.0], "a": [2.0]}))  #> Columns arrive in the order b, a.
```

**Cause:** scikit-learn remembers column names from `fit` (`model.feature_names_in_`). New data has different names or order. **Fix:** select the columns in the training order: `new[model.feature_names_in_]`.

### `ValueError: Invalid parameter … for estimator`

```py error title="Wrong parameter name in a grid search"
from sklearn.pipeline import make_pipeline  #> Import `make_pipeline` from scikit-learn's `pipeline` module.
from sklearn.model_selection import GridSearchCV  #> Import `GridSearchCV` from scikit-learn's `model_selection` module.
pipe = make_pipeline(StandardScaler(), LogisticRegression())  #> Steps are named standardscaler and logisticregression.
GridSearchCV(pipe, {"C": [0.1, 1]}).fit(np.random.default_rng(0).normal(size=(20, 2)), [0, 1] * 10)  #> "C" lacks the step prefix.
```

**Cause:** inside a pipeline, parameters need the step-name prefix. **Fix:** `{"logisticregression__C": [...]}`. List valid names with `pipe.get_params().keys()`.

### `ValueError: Target is multiclass but average='binary'`

```py error title="Binary metric on a multiclass problem"
from sklearn.metrics import f1_score  #> Import `f1_score` from scikit-learn's `metrics` module.
f1_score([0, 1, 2, 2], [0, 2, 1, 2])  #> Three classes with the default average="binary".
```

**Cause:** precision/recall/F1 default to `average="binary"`. **Fix:** `average="macro"` (classes equal), `"weighted"` (by support) or `"micro"`.

### `ValueError: The least populated classes in y have only 1 member`

```py error title="Stratifying with a class that has a single example"
from sklearn.model_selection import train_test_split  #> Import `train_test_split` from scikit-learn's `model_selection` module.
train_test_split(np.arange(6), [0, 0, 0, 1, 1, 2], stratify=[0, 0, 0, 1, 1, 2], test_size=0.5)  #> Class 2 has a single member.
```

**Cause:** stratification needs at least 2 members per class (and at least `n_splits` for stratified CV). **Fix:** merge or drop ultra-rare classes, collect more data, or don't stratify.

### `ConvergenceWarning: lbfgs failed to converge`

```py warn title="The optimiser hit max_iter"
from sklearn.datasets import load_breast_cancer  #> Import `load_breast_cancer` from scikit-learn's `datasets` module.
Xb, yb = load_breast_cancer(return_X_y=True)  #> Unscaled features with very different ranges.
LogisticRegression(max_iter=20).fit(Xb, yb)        #> Unscaled features + very few iterations.
```

**Cause:** the solver stopped before converging, usually because features are unscaled or `max_iter` is too low. **Fix:** put a `StandardScaler` before the model (this fixes it most of the time), then raise `max_iter` (e.g. 1000–5000).

## TensorFlow / Keras errors {I}

### `Input 0 … is incompatible with the layer: expected axis -1 of input shape to have value …`

```py error title="Data has a different number of features than the model expects"
import os  #> Import the standard-library `os` module (operating-system tools).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #> Silence TensorFlow's start-up logs (set before importing).
import tensorflow as tf  #> Import TensorFlow under the short name `tf`.
from tensorflow import keras  #> Import `keras` from `tensorflow`.
from tensorflow.keras import layers  #> Import `layers` from `tensorflow.keras`.
model = keras.Sequential([keras.Input(shape=(4,)), layers.Dense(1)])  #> Expects 4 features.
model.compile(optimizer="adam", loss="mse")  #> Compile.
model.fit(np.zeros((10, 5), dtype="float32"), np.zeros(10), verbose=0)  #> Data has 5 features.
```

**Cause:** the data's last dimension differs from `keras.Input(shape=...)` (wrong columns, a forgotten one-hot column, or a missing channel axis for images). **Fix:** print `X.shape` and `model.input_shape`; set `Input(shape=(X.shape[1],))`; for images add a channel axis with `X[..., None]`.

### `ValueError: Arguments target and output must have the same rank`: labels don't match the loss

```py error title="One-hot loss with integer labels"
clf = keras.Sequential([keras.Input(shape=(3,)), layers.Dense(4, activation="softmax")])  #> 4-class softmax output.
clf.compile(optimizer="adam", loss="categorical_crossentropy")  #> Expects one-hot labels.
clf.fit(np.zeros((8, 3), dtype="float32"), np.array([0, 1, 2, 3, 0, 1, 2, 3]), verbose=0)  #> Integer labels instead.
```

**Cause:** `categorical_crossentropy` expects one-hot labels of shape `(n, k)`, but integer labels of shape `(n,)` were given. **Fix:** use `sparse_categorical_crossentropy` for integer labels, or one-hot encode with `keras.utils.to_categorical(y)`. See the output/loss table in [[ch:keras-mlp]].

### `InvalidArgumentError: … expected to be a float tensor but is a double tensor`

```py error title="Mixing NumPy float64 with float32 tensors"
w = tf.Variable(tf.ones((2,)))                       #> float32
x64 = np.array([1.0, 2.0])                           #> NumPy default float64
tf.add(w, tf.constant(x64))  #> float32 + float64.
```

**Cause:** TensorFlow doesn't silently upcast. **Fix:** `X = X.astype("float32")`, or `tf.cast(t, tf.float32)`.

### Loss becomes `nan`

```py title="Detecting NaN during training"
bad_X = np.array([[1.0], [np.nan], [3.0]], dtype="float32")            #> One missing value hidden in the inputs.
m = keras.Sequential([keras.Input(shape=(1,)), layers.Dense(1)])  #> A tiny regression model.
m.compile(optimizer="adam", loss="mse")  #> Compile.
h = m.fit(bad_X, np.array([1.0, 2.0, 3.0]), epochs=2, verbose=0,  #> Train on data containing a NaN...
          callbacks=[keras.callbacks.TerminateOnNaN()])                 #> Stops training as soon as the loss is NaN.
print("losses:", h.history["loss"])  #> The loss is nan and training stopped.
```

**Cause:** NaN/inf in inputs or labels, a learning rate that is too high, `log(0)` in a custom loss, or unscaled targets. **Fix:** check `np.isnan(X).sum()` and `np.isinf(X).sum()`; scale inputs and targets; lower the learning rate 10×; add `clipnorm=1.0` to the optimiser; use the built-in losses with `from_logits=True` instead of hand-written log formulas.

### Out of memory (`ResourceExhaustedError: OOM when allocating tensor`)

**Cause:** the batch or model is too large for GPU memory. **Fix:** lower `batch_size`, use smaller images, enable memory growth (`tf.config.experimental.set_memory_growth(gpu, True)`), use mixed precision (`keras.mixed_precision.set_global_policy("mixed_float16")`), or stream data with `tf.data` instead of loading everything.

## Notebook and environment gotchas {B}

| Symptom | Likely cause | Fix |
|---|---|---|
| Code works in one run, fails after restart | cells run out of order; a variable defined in a deleted cell | *Restart & Run All* regularly |
| Edited `utils.py` but nothing changed | Python caches imported modules | `%load_ext autoreload` + `%autoreload 2`, or restart the kernel |
| `pip install` succeeded but import fails | installed into a different Python | `%pip install …` inside the notebook; check `sys.executable` |
| Plots don't appear (script) | no `plt.show()` | call `plt.show()` or save with `fig.savefig()` |
| Results change every run | unseeded randomness | set `random_state` / `set_random_seed` |
| Extremely slow pandas code | row-wise `apply` / Python loops | vectorise; use `groupby`/`merge`; NumPy |

:::ex Try it yourself
Pick any three errors from this chapter, reproduce them in your own environment, then fix each one. Being able to *cause* an error on purpose is the fastest way to recognise it in the wild.
:::
