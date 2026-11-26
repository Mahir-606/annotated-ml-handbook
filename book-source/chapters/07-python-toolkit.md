# The Pythonic toolkit

Python's standard library ships with tools that turn ten-line loops into one clear line. These are the ones that appear again and again in data and ML code.

## collections: smarter containers {I}

```py title="Counter, defaultdict, namedtuple, deque"
from collections import Counter, defaultdict, namedtuple, deque  #> Import `Counter`, `defaultdict`, `namedtuple` and `deque` from `collections`.

labels = ["cat", "dog", "cat", "bird", "cat", "dog"]  #> Labels to count.
counts = Counter(labels)                       #> Counts occurrences of each item.
print(counts, counts.most_common(1))           #> `most_common(k)` returns the top-k (item, count) pairs.
print(counts["fish"])                          #> Missing keys count as 0 instead of raising KeyError.

by_group = defaultdict(list)                   #> A dict that creates a default value (here an empty list) for missing keys.
for name, group in [("ana", "A"), ("bo", "B"), ("cy", "A")]:  #> Loop over `[("ana", "A"), ("bo", "B"), ("cy", "A")]`, naming each item `name, group`.
    by_group[group].append(name)               #> No need to check whether the key exists first.
print(dict(by_group))  #> Convert to a normal dict for printing: {'A': ['ana', 'cy'], 'B': ['bo']}.

Point = namedtuple("Point", ["x", "y"])         #> A tuple whose fields also have names.
p = Point(2, 5)  #> Create a named tuple.
print(p.x, p[1], p)  #> Access by name or by position.

window = deque(maxlen=3)                       #> A double-ended queue; with maxlen it keeps only the newest items.
for loss in [0.9, 0.7, 0.6, 0.55, 0.54]:  #> Loop over `[0.9, 0.7, 0.6, 0.55, 0.54]`, naming each item `loss`.
    window.append(loss)  #> When full, appending pushes out the oldest value.
print(list(window), sum(window) / len(window)) #> Moving average of the last 3 losses.
```

## itertools: loops without the loops {I}

```py title="Grids, combinations and chunks"
from itertools import product, combinations, permutations, chain, islice, accumulate  #> Import `product`, `combinations`, `permutations`, `chain`, `islice` and `accumulate` from `itertools`.

grid = {"lr": [0.1, 0.01], "depth": [3, 5]}  #> Candidate values for two hyperparameters.
for lr, depth in product(grid["lr"], grid["depth"]):    #> Cartesian product: every combination, a hyperparameter grid.
    print(f"lr={lr}, depth={depth}")  #> Four combinations in total.
print(list(combinations(["A", "B", "C"], 2)))           #> All unordered pairs, e.g. pairwise model comparisons.
print(len(list(permutations(range(4)))))                #> Ordered arrangements: 4! = 24.
print(list(chain([1, 2], (3, 4), {5})))                 #> Concatenate several iterables lazily.
print(list(islice(range(100), 5, 10)))                  #> Slice any iterator (works on generators too).
print(list(accumulate([1, 2, 3, 4])))                   #> Running totals: [1, 3, 6, 10].
```

## functools: partial and reduce {A}

```py title="Pre-filling arguments"
from functools import partial, reduce  #> Import `partial` and `reduce` from `functools`.

def scale(x, factor, offset=0):  #> Define a function `scale` that takes `x`, `factor` and `offset`.
    return x * factor + offset  #> Send `x * factor + offset` back to the caller.

to_percent = partial(scale, factor=100)        #> A new function with `factor` fixed at 100.
print(to_percent(0.257), to_percent(0.5, offset=1))  #> 25.7 and 51.0 (up to float rounding).
print(reduce(lambda a, b: a * b, [1, 2, 3, 4, 5]))   #> Fold a list into one value: ((((1*2)*3)*4)*5) = 120.
```

## Context managers {I}

:::def
**Context manager**: an object used with `with` that sets something up on entry and guarantees clean-up on exit, such as closing a file.
:::


Anything used with `with` is a context manager: it sets something up and guarantees clean-up. You can write your own with `contextlib`.

```py title="A timer you can wrap around any code"
import time  #> Import the standard-library `time` module.
from contextlib import contextmanager  #> Import `contextmanager` from `contextlib`.

@contextmanager  #> Turn the generator function below into a context manager.
def timer(label):  #> Define a function `timer` that takes `label`.
    start = time.perf_counter()        #> Code before `yield` runs when the `with` block starts.
    try:  #> Start a block whose errors we want to catch.
        yield                          #> The body of the `with` block runs here.
    finally:  #> This block always runs, whether or not an error happened.
        print(f"{label}: {time.perf_counter() - start:.4f} s")   #> Runs when the block ends, even after an error.

with timer("sum of squares"):  #> Everything indented below is timed.
    total = sum(i * i for i in range(300_000))  #> Some work to time.
```

## Randomness and reproducibility {B}

:::def
**Random seed**: a starting number for a pseudo-random generator. The same seed always produces the same "random" sequence, which makes results reproducible.

**Reproducibility**: the ability to obtain the same results by re-running the same code on the same data.
:::


ML is full of randomness: data shuffling, train/test splits, weight initialisation. **Seeding** makes it repeatable.

```py title="Seeded random numbers"
import random  #> Import Python's built-in `random` module.

random.seed(42)                         #> Fix the starting state of Python's random generator.
print(random.random(), random.randint(1, 6))   #> Float in [0, 1) and an integer from 1 to 6 inclusive.
items = list(range(10))  #> The numbers 0..9 as a list.
random.shuffle(items)                   #> Shuffle a list in place.
print(items, random.choice(items), random.sample(items, 3))   #> One random item; 3 distinct random items.
random.seed(42)  #> Reset to the same seed.
print(random.random())                  #> Same seed -> the very same "random" sequence.
```

:::research Seed everything, and report it
For a reproducible experiment you seed Python's `random`, NumPy's generator, and your framework (`random_state=` in scikit-learn, `tf.keras.utils.set_random_seed()` in TensorFlow). Then report results over several seeds (mean ± standard deviation), not one lucky run. [[ch:research-protocol]] shows how.
:::

## Dates and times {B}

```py title="datetime basics"
from datetime import datetime, timedelta, date  #> Import `datetime`, `timedelta` and `date` from `datetime`.

d = datetime(2026, 3, 14, 9, 30)                     #> Year, month, day, hour, minute.
print(d.isoformat(), d.strftime("%d %b %Y"))         #> ISO 8601 string; custom format (%d day, %b short month, %Y year).
print(d + timedelta(days=30))                        #> Date arithmetic with timedelta.
parsed = datetime.strptime("2026-01-05", "%Y-%m-%d") #> Parse a string with a known format.
print((d - parsed).days, date(2026, 3, 14).weekday())   #> Difference in days; weekday: Monday = 0.
```

## Regular expressions {I}

:::def
**Regular expression (regex)**: a pattern language for finding and extracting text, such as `\d+` for "one or more digits".
:::


Regular expressions find patterns in text: IDs, dates, numbers buried in strings. They are also how pandas' `.str.extract()` works.

```py title="Extracting structured values from messy text"
import re  #> Import the regular-expression module `re`.

text = "Sample S-104 measured 12.5 mg on 2026-02-01; sample S-221 measured 7 mg."  #> Messy text with IDs, doses and a date.
ids = re.findall(r"S-\d+", text)                    #> `\d+` = one or more digits. `r"..."` raw strings keep backslashes literal.
doses = re.findall(r"(\d+(?:\.\d+)?) mg", text)     #> A group (...) captures just the number; (?:...) groups without capturing.
date_m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text) #> `search` returns the first match or None.
print(ids, [float(x) for x in doses], date_m.groups())  #> IDs, doses converted to floats, and the captured date parts.
print(re.sub(r"\s+", " ", "too    many   spaces"))  #> Replace runs of whitespace with a single space.
```

## Measuring speed {I}

```py title="Timing small snippets"
import timeit  #> Import the `timeit` module.

loop_time = timeit.timeit("s = 0\nfor i in range(1000): s += i", number=2000)   #> Run the snippet 2000 times; total seconds.
builtin_time = timeit.timeit("sum(range(1000))", number=2000)  #> The same total using the built-in `sum`.
print(f"loop {loop_time:.3f}s vs sum() {builtin_time:.3f}s -> {loop_time / builtin_time:.1f}x faster")  #> Compare the two timings.
```

In Jupyter, `%timeit expression` does the same in one line, and `%%time` times a whole cell. The next part shows how NumPy makes numeric code tens to hundreds of times faster than pure Python loops.

:::ex Try it yourself
1. Use `Counter` to find the three most common words in a paragraph (lowercase it and split on whitespace first).
2. Use `itertools.product` to list all 12 combinations of `C in [0.1, 1, 10]` and `kernel in ["linear", "rbf", "poly", "sigmoid"]`.
3. Write a regex that extracts all e-mail addresses from a string.
:::
