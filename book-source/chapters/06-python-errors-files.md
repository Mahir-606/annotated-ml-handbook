# Errors, files and modules

Real programs read files, call other people's code and run into bad input. This chapter covers handling errors gracefully, reading and writing the file formats you will meet in research, and organising code into modules.

## Exceptions and try/except {B}

:::def
**Exception**: an object representing an error, which Python *raises* when something goes wrong, such as `ValueError` or `KeyError`. Unhandled, it stops the program.

**Exception handling**: catching an expected error with `try` / `except` so the program can respond instead of crashing.
:::


When something goes wrong Python *raises an exception*. If nothing catches it, the program stops and prints a **traceback**. You can catch exceptions you expect and handle them.

```py title="Catching specific exceptions"
def safe_divide(a, b):  #> Define a function `safe_divide` that takes `a` and `b`.
    try:                                      #> Run the risky code inside `try`.
        result = a / b  #> The line that may fail.
    except ZeroDivisionError:                 #> Runs only if THIS kind of error happened.
        print("Cannot divide by zero; returning None")  #> Handle the problem gracefully.
        return None  #> Send `None` back to the caller.
    except TypeError as err:                  #> `as err` binds the exception object so you can inspect it.
        print("Bad types:", err)  #> The exception object prints as its message.
        return None  #> Send `None` back to the caller.
    else:                                     #> Runs only if no exception occurred.
        return result  #> Send `result` back to the caller.
    finally:                                  #> ALWAYS runs, error or not: use it for clean-up.
        print("done:", a, b)  #> Printed on every call, even after an error.

print(safe_divide(10, 4))  #> Works: 2.5 (after the finally message).
print(safe_divide(1, 0))  #> Triggers the ZeroDivisionError branch.
print(safe_divide("1", 2))  #> Triggers the TypeError branch.
```

:::warn Never write a bare `except:`
`except:` or `except Exception: pass` swallows *every* error, including typos and bugs you need to see. Catch the specific exception you expect, and let everything else crash loudly.
:::

### Raising your own exceptions {I}

Validate inputs early and fail with a clear message. A helpful error message saves hours of debugging later.

```py title="Validating inputs"
class DataValidationError(ValueError):        #> A custom exception: subclass a built-in one.
    pass  #> Do nothing: the body is intentionally empty.

def check_probabilities(ps):  #> Define a function `check_probabilities` that takes `ps`.
    if not ps:  #> Run the indented block only if `not ps` is true.
        raise DataValidationError("empty list of probabilities")  #> Stop with a specific, clear error.
    bad = [p for p in ps if not 0 <= p <= 1]  #> Collect every value outside [0, 1].
    if bad:  #> Run the indented block only if `bad` is true.
        raise DataValidationError(f"{len(bad)} value(s) outside [0, 1]: {bad[:3]}")  #> The message says how many are bad and shows a few.
    return True  #> Send `True` back to the caller.

try:  #> Start a block whose errors we want to catch.
    check_probabilities([0.2, 1.4, -0.1])  #> Two invalid values.
except DataValidationError as e:  #> Catch only our custom error.
    print(type(e).__name__, "->", e)  #> The exception's class name and its message.
```

`assert` is a quick sanity check for *your own* assumptions while developing:

```py error title="assert stops the program when an assumption fails"
n_train, n_labels = 120, 118  #> Two counts that should match but don't.
assert n_train == n_labels, f"features ({n_train}) and labels ({n_labels}) differ"   #> If the condition is False, raises AssertionError with your message.
```

## Reading a traceback {B}

:::def
**Traceback**: the report Python prints when an exception is not handled: the chain of function calls that led to the error, ending with the error type and message.
:::


Read a traceback **from the bottom up**. The last line names the exception type and message. The lines above show the chain of calls that led there, ending with the line that failed. Part VIII expands on this, with dozens of real examples.

```py error title="A traceback through two functions"
def load_scores(record):  #> Define a function `load_scores` that takes `record`.
    return record["scores"]          #> Fails here: the dict has no "scores" key.

def average(record):  #> Define a function `average` that takes `record`.
    s = load_scores(record)          #> Called from here...
    return sum(s) / len(s)  #> Send `sum(s) / len(s)` back to the caller.

average({"score": [1, 2, 3]})       #> ...which was called from here. Note the typo: "score" vs "scores".
```

## Working with files {B}

:::def
**File path**: the location of a file, absolute (`/home/ada/data.csv`) or relative to the current working directory (`data/data.csv`).

**Working directory**: the folder a program treats as its starting point for relative paths. Check it with `Path.cwd()`.

**Encoding**: the rule that turns text into bytes on disk. UTF-8 is the modern standard.
:::


Always open files with a `with` block. It guarantees the file is closed even if an error occurs.

```py title="Writing and reading text files"
with open("notes.txt", "w", encoding="utf-8") as f:   #> Mode "w" = write (creates or overwrites). Always state the encoding.
    f.write("epoch,loss\n")                            #> `write` does not add newlines; include `\n` yourself.
    for epoch, loss in enumerate([0.9, 0.6, 0.45], 1):  #> Loop over `enumerate([0.9, 0.6, 0.45], 1)`, naming each item `epoch, loss`.
        f.write(f"{epoch},{loss}\n")  #> One CSV line per epoch.

with open("notes.txt", encoding="utf-8") as f:          #> Default mode "r" = read.
    header = f.readline().strip()                       #> Read one line; strip the trailing newline.
    rows = [line.strip().split(",") for line in f]      #> Iterating a file yields one line at a time (memory-friendly).
print(header, rows)  #> The header string and a list of [epoch, loss] string pairs.
with open("notes.txt", "a", encoding="utf-8") as f:     #> Mode "a" = append to the end.
    f.write("4,0.40\n")  #> Add one more line at the end.
```

### Paths with pathlib {B}

`pathlib` handles file paths portably across Windows, macOS and Linux.

```py title="Building, checking and listing paths"
from pathlib import Path  #> Import `Path` from `pathlib`.

root = Path("project")                       #> A Path object (no file is created yet).
data_dir = root / "data" / "raw"             #> `/` joins path parts correctly on every operating system.
data_dir.mkdir(parents=True, exist_ok=True)  #> Create the folders, including parents; no error if they exist.
(data_dir / "a.csv").write_text("x\n1\n")    #> Quick write without open().
(data_dir / "b.csv").write_text("x\n2\n")  #> A second small file.
print(sorted(p.name for p in data_dir.glob("*.csv")))   #> Find files matching a pattern.
p = data_dir / "a.csv"  #> A path to one file.
print(p.exists(), p.suffix, p.stem, p.parent)           #> Inspect parts of a path.
```

### JSON and CSV {B}

:::def
**CSV (comma-separated values)**: a plain-text table format: one row per line, values separated by commas (or semicolons or tabs).

**JSON**: a plain-text format for nested data made of objects (like dicts), arrays (like lists), strings, numbers, true/false and null.
:::


```py title="JSON for configs and results, CSV for tables"
import json, csv  #> Import the standard-library `json` module and the `csv` module.

results = {"model": "rf", "acc": 0.91, "params": {"n_estimators": 300}}  #> A nested dict to save.
with open("results.json", "w") as f:  #> Open a file for writing; it closes automatically.
    json.dump(results, f, indent=2)          #> Serialise a dict to a JSON file; indent makes it human-readable.
with open("results.json") as f:  #> Open it again for reading.
    loaded = json.load(f)                    #> Parse it back into Python dicts and lists.
print(loaded["params"]["n_estimators"])  #> Display the value of `loaded["params"]["n_estimators"]`.

rows = [{"id": 1, "group": "A"}, {"id": 2, "group": "B"}]  #> Rows as a list of dicts.
with open("groups.csv", "w", newline="") as f:          #> newline="" avoids blank lines on Windows.
    writer = csv.DictWriter(f, fieldnames=["id", "group"])  #> A writer that maps dict keys to columns.
    writer.writeheader()  #> Write the header line: id,group.
    writer.writerows(rows)  #> Write one line per dict.
with open("groups.csv", newline="") as f:  #> Re-open the file for reading.
    print(list(csv.DictReader(f)))                      #> Each row comes back as a dict of STRINGS.
```

:::tip Use pandas for real tables
The `csv` module is fine for small jobs, but for analysis you will use `pd.read_csv()` (Part III), which infers types, handles missing values and is much faster.
:::

## Modules, packages and imports {B}

:::def
**Module**: a single `.py` file containing Python code that can be imported.

**Package / library**: a collection of modules distributed together, such as NumPy or scikit-learn. Installed with `pip` or `conda`.

**Import**: making a module's contents available in your code: `import numpy as np`.
:::


A **module** is any `.py` file; a **package** is a folder of modules. `import` makes their contents available.

```py title="Import styles"
import math                          #> Import the module; use names as math.sqrt.
import numpy as np                   #> Import with an alias (the community-standard aliases: np, pd, plt, sns, tf).
from statistics import mean, median  #> Import specific names directly.
from collections import Counter as C #> Import and rename.
print(math.floor(2.7), np.pi > 3, mean([1, 2, 3]), median([3, 1, 2]), C("aab"))  #> Use a name from each import style.
```

:::warn Avoid `from module import *`
It dumps every name into your namespace, silently overwriting your own variables and making it impossible to tell where a name came from.
:::

Your own module is just a file. Suppose `utils.py` sits next to your script:

```py norun title="utils.py: a reusable module"
"""Helper functions for the project."""  #> A module docstring describes the whole file.

def zscore(xs):                         #> Any function defined here can be imported elsewhere.
    m = sum(xs) / len(xs)  #> Mean.
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5  #> Sample standard deviation.
    return [(x - m) / sd for x in xs]  #> Send `[(x - m) / sd for x in xs]` back to the caller.

if __name__ == "__main__":              #> True only when this file is run directly (python utils.py), not when imported.
    print(zscore([1, 2, 3]))            #> A good place for a quick self-test.
```

```py norun title="main.py: using it"
from utils import zscore                 #> Works when utils.py is in the same folder (or on the Python path).
print(zscore([10, 20, 30]))  #> [-1.0, 0.0, 1.0]
```

:::bug `ModuleNotFoundError` for your own file
Python searches the folder of the script being run, then installed packages. Run scripts from the project root, keep imported files beside them, or turn your project into a package with `pip install -e .`. Also check you didn't name your file `numpy.py`, `pandas.py` or `random.py`: it will shadow the real library.
:::

## Logging instead of print {I}

For long training runs, `logging` gives timestamps and severity levels, and can write to a file.

```py title="Basic logging setup"
import logging, sys  #> Import the `logging` module and the `sys` module (interpreter information).
logging.basicConfig(level=logging.INFO, stream=sys.stdout,        #> Show INFO and above, printed to the console.
                    format="%(asctime)s | %(levelname)s | %(message)s",  #> Each line shows the time, the level and the message.
                    datefmt="%H:%M:%S", force=True)                #> `force=True` replaces any earlier configuration.
log = logging.getLogger("train")  #> A named logger for this part of the program.
log.info("starting run with lr=%s", 0.001)                         #> Lazy %-style formatting is the logging convention.
log.warning("validation loss increased")  #> A WARNING-level message.
log.debug("not shown: DEBUG is below the INFO level")  #> Filtered out by level=INFO.
```

:::ex Try it yourself
1. Write `read_numbers(path)` that returns a list of floats from a text file, skipping lines that can't be converted (catch `ValueError`).
2. Save a dict of hyperparameters to `config.json` and load it back.
3. Use `Path.glob("**/*.csv")` to count CSV files in a folder tree.
:::
