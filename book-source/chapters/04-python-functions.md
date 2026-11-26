# Functions: reusable, testable logic

A function packages a piece of logic under a name so you can reuse it, test it and reason about it in isolation. Well-named small functions are the difference between an analysis you can trust and a notebook nobody can re-run.

## Defining and calling functions {B}

:::def
**Function**: a named, reusable block of code that takes inputs and can return an output. Defined with `def`, called with parentheses: `bmi(70, 1.75)`.

**Parameter vs argument**: a parameter is the name in the function definition (`weight_kg`); an argument is the actual value passed when calling it (`70`).

**Return value**: the result a function hands back with `return`. Without `return`, a function returns `None`.

**Docstring**: a string at the start of a function, class or module that documents it; shown by `help()`.
:::


```py title="Anatomy of a function"
def bmi(weight_kg, height_m):                 #> `def`, a name, parameters in parentheses, then a colon.
    """Return body-mass index (kg / m^2)."""  #> A *docstring*: the first string in the body documents the function.
    value = weight_kg / height_m ** 2         #> Local variable: exists only while the function runs.
    return round(value, 1)                    #> `return` sends a value back to the caller and exits the function.

print(bmi(70, 1.75))                          #> Call with *positional* arguments, matched by order.
print(bmi(height_m=1.6, weight_kg=55))        #> Call with *keyword* arguments, matched by name: order no longer matters.
print(bmi.__doc__)                            #> The docstring is stored on the function; `help(bmi)` shows it too.
```

A function without `return` returns `None`. A function can return several values at once; it actually returns a tuple you can unpack.

```py title="Returning several values"
def mean_and_std(values):  #> Define a function `mean_and_std` that takes `values`.
    n = len(values)  #> Number of values.
    mean = sum(values) / n  #> The arithmetic mean.
    var = sum((v - mean) ** 2 for v in values) / (n - 1)   #> Sample variance (divide by n-1).
    return mean, var ** 0.5                                 #> Returns the tuple (mean, std).

m, s = mean_and_std([2, 4, 4, 4, 5, 5, 7, 9])               #> Unpack the returned tuple into two names.
print(f"mean={m:.3f}, sd={s:.3f}")  #> Report both, rounded to 3 decimals.
```

## Default values, *args and **kwargs {I}

:::def
**Keyword argument**: an argument passed by parameter name, as in `train(model, lr=0.01)`. **Positional arguments** are matched by order instead.

***args and **kwargs**: parameters that collect any number of extra positional arguments (into a tuple) and keyword arguments (into a dict).
:::


```py title="Flexible signatures"
def train(model, epochs=10, lr=0.001, *, verbose=False):   #> Parameters with `=` have defaults. After `*`, arguments are keyword-only.
    return f"{model}: epochs={epochs}, lr={lr}, verbose={verbose}"  #> Send `f"{model}: epochs={epochs}, lr={lr}, verbose={verbose}"` back to the caller.

print(train("mlp"))                         #> Defaults fill in the rest.
print(train("mlp", 50, verbose=True))       #> `verbose` MUST be passed by name because it follows `*`.

def summarize(*values, **options):          #> `*values` collects extra positional args into a tuple; `**options` collects keyword args into a dict.
    print("values :", values)  #> The extra positional arguments, packed in a tuple.
    print("options:", options)  #> The extra keyword arguments, packed in a dict.

summarize(1, 2, 3, unit="cm", digits=2)  #> Three positional and two keyword arguments.
nums = [4, 5]  #> A list we will unpack into positional arguments.
cfg = {"unit": "kg"}  #> A dict we will unpack into keyword arguments.
summarize(*nums, **cfg)                     #> At call time, `*` and `**` *unpack* a list and a dict into arguments.
```

:::warn The mutable default argument trap
Default values are created **once**, when the function is defined, not on each call.
:::

```py title="Why def f(x, history=[]) is a bug"
def add_bad(item, history=[]):        #> This single list is shared by every call that uses the default.
    history.append(item)  #> Add the item to the list.
    return history  #> Send `history` back to the caller.

print(add_bad("a"))  #> First call: ['a'], as expected.
print(add_bad("b"))                   #> Surprise: ['a', 'b']. The list remembers the previous call.

def add_good(item, history=None):     #> Correct pattern: use None as the default...
    if history is None:  #> Run the indented block only if `history is None` is true.
        history = []                  #> ...and create a fresh list inside the function.
    history.append(item)  #> Add the item to the list.
    return history  #> Send `history` back to the caller.

print(add_good("a"), add_good("b"))  #> Each call gets its own fresh list: ['a'] and ['b'].
```

## Scope: where names live {I}

:::def
**Scope**: the region of a program where a name is visible. Names created inside a function are **local** to it; names at the top level of a file are **global**.
:::


Python looks names up in the order **L**ocal → **E**nclosing → **G**lobal → **B**uilt-in (the *LEGB* rule).

```py title="Local vs global names"
threshold = 0.5                       #> A *global* name, defined at the top level of the script.

def classify(p):  #> Define a function `classify` that takes `p`.
    label = 1 if p >= threshold else 0   #> Reading a global inside a function is allowed...
    return label  #> Send `label` back to the caller.

def bump():  #> Define a function `bump` with no arguments.
    threshold = 0.9                   #> ...but ASSIGNING creates a new LOCAL name; the global is untouched.
    return threshold  #> Send `threshold` back to the caller.

print(classify(0.7), bump(), threshold)  #> 1, 0.9, and the global threshold is still 0.5.
```

:::tip Pass data in, return data out
Functions that read and modify globals are hard to test and reuse. Prefer functions whose output depends only on their arguments. Avoid the `global` keyword in analysis code.
:::

## Lambda functions and functions as values {I}

:::def
**Lambda**: a small anonymous function written in one expression. *Example:* `lambda x: x ** 2`.

**Higher-order function**: a function that takes another function as an argument or returns one, such as `sorted(key=...)`, `map`, `filter`.
:::


Functions are ordinary objects: you can store them in variables, put them in lists and pass them to other functions. A `lambda` is a small anonymous function written in one expression.

```py title="Passing functions around"
square = lambda x: x ** 2                       #> Equivalent to `def square(x): return x ** 2`.
print(square(7))  #> Call the lambda like any function: 49.
people = [("Ana", 31), ("Bo", 25), ("Cy", 40)]  #> A list of (name, age) tuples.
print(sorted(people, key=lambda p: p[1]))       #> Sort tuples by their second element (age).
print(list(map(lambda x: x * 10, [1, 2, 3])))   #> `map` applies a function to every item (a comprehension is often clearer).
print(list(filter(lambda x: x > 1, [0, 1, 2, 3])))  #> `filter` keeps items where the function returns True.

def apply_twice(f, x):                          #> A *higher-order* function: takes a function as an argument.
    return f(f(x))  #> Send `f(f(x))` back to the caller.
print(apply_twice(square, 3))                   #> square(square(3)) = 81.
```

## Closures and decorators {A}

:::def
**Closure**: a function that remembers variables from the scope in which it was created.

**Decorator**: a function that takes a function and returns a modified version; applied with `@name` above a definition.
:::


A **closure** is a function that remembers variables from the scope where it was created. A **decorator** is a function that takes a function and returns an enhanced version; the `@name` syntax applies it.

```py title="A timing decorator"
import time  #> Import the standard-library `time` module.
from functools import wraps                 #> `wraps` copies the original function's name and docstring onto the wrapper.

def timed(func):                            #> The decorator receives the function to wrap.
    @wraps(func)  #> Decorate the wrapper so it keeps `func`'s name and docstring.
    def wrapper(*args, **kwargs):           #> Accept any arguments so it can wrap any function.
        start = time.perf_counter()         #> High-resolution timer.
        result = func(*args, **kwargs)      #> Call the original function.
        elapsed = time.perf_counter() - start  #> Seconds elapsed during the call.
        print(f"{func.__name__} took {elapsed * 1000:.2f} ms")  #> Report the function's name and its run time in milliseconds.
        return result                       #> Return the original result unchanged.
    return wrapper  #> Send `wrapper` back to the caller.

@timed                                      #> Same as writing: slow_sum = timed(slow_sum)
def slow_sum(n):  #> Define a function `slow_sum` that takes `n`.
    return sum(i * i for i in range(n))  #> Send `sum(i * i for i in range(n))` back to the caller.

total = slow_sum(200_000)                   #> Underscores in numbers are allowed for readability.
print(total)  #> Display the value of `total`.
```

```py title="Caching expensive results"
from functools import lru_cache             #> A built-in decorator that memoises results.

@lru_cache(maxsize=None)                    #> Remember every (argument -> result) pair.
def fib(n):  #> Define a function `fib` that takes `n`.
    return n if n < 2 else fib(n - 1) + fib(n - 2)  #> Send `n if n < 2 else fib(n - 1) + fib(n - 2)` back to the caller.

print(fib(80))                              #> Instant; without the cache this recursion would take years.
print(fib.cache_info())  #> Cache statistics: hits, misses and current size.
```

## Generators: producing values lazily {I}

:::def
**Generator**: a function containing `yield` that produces a sequence of values one at a time, pausing between them. It saves memory for large or infinite sequences.
:::


A function containing `yield` becomes a **generator**. Each `yield` hands out one value and pauses the function until the next value is requested. Generators let you process data that does not fit in memory, and they are the idea behind Keras data pipelines.

```py title="A mini-batch generator"
def batches(data, batch_size):  #> Define a function `batches` that takes `data` and `batch_size`.
    for start in range(0, len(data), batch_size):   #> Step through the data in chunks.
        yield data[start:start + batch_size]        #> Hand out one chunk, then pause here.

for b in batches(list(range(10)), 4):               #> The loop asks for values one at a time.
    print(b)  #> Display the value of `b`.
squares = (x * x for x in range(5))                 #> A *generator expression*: like a list comprehension with ().
print(sum(squares))                                 #> Consumed lazily by sum(); no list is ever built.
```

## Type hints and docstrings {I}

:::def
**Type hint**: an annotation stating the expected type of a parameter or return value, such as `def f(x: float) -> list[float]`. Not enforced at run time.
:::


Type hints document what a function expects and returns. Python does not enforce them at run time, but editors and tools such as `mypy` use them to catch mistakes early.

```py title="A well-documented function"
def normalize(values: list[float], eps: float = 1e-12) -> list[float]:  #> Type hints: `values` is a list of floats, `eps` a float; `-> list[float]` is the return type.
    """Scale values to the range [0, 1].

    Parameters
    ----------
    values : list of float
        Raw measurements.
    eps : float, default 1e-12
        Guards against division by zero when all values are equal.

    Returns
    -------
    list of float
    """                                                #> This "NumPy docstring" layout is used by NumPy, pandas and scikit-learn.
    lo, hi = min(values), max(values)  #> Smallest and largest values.
    return [(v - lo) / (hi - lo + eps) for v in values]  #> Send `[(v - lo) / (hi - lo + eps) for v in values]` back to the caller.

print(normalize([10.0, 15.0, 20.0]))  #> [0.0, 0.5, 1.0] (up to tiny rounding from eps).
print(normalize.__annotations__)                       #> The hints are stored on the function.
```

:::research Functions make analyses reproducible
Wrap each stage of an analysis (load, clean, featurize, train, evaluate) in a function with clear inputs and outputs. You can then re-run the whole pipeline with a different seed or dataset by changing one argument, which is what reviewers ask for.
:::

:::ex Try it yourself
1. Write `def describe(values)` that returns a dict with `n`, `mean`, `min` and `max`.
2. Write a decorator `@count_calls` that prints how many times a function has been called.
3. Write a generator `def running_mean(xs)` that yields the mean of the items seen so far.
:::
