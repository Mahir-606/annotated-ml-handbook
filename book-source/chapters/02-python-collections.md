# Data structures: lists, tuples, dictionaries, sets

Real data comes in groups: a column of measurements, a record of a patient, a set of unique labels. Python's four built-in collections cover nearly every case, and understanding them makes NumPy and pandas feel familiar later.

| Type | Literal | Ordered | Mutable | Duplicates | Typical use |
|---|---|---|---|---|---|
| `list` | `[1, 2, 3]` | yes | yes | yes | a sequence you will grow, sort or edit |
| `tuple` | `(1, 2, 3)` | yes | no | yes | a fixed record, e.g. `(lat, lon)`, a shape `(rows, cols)` |
| `dict` | `{"a": 1}` | insertion order | yes | keys unique | lookup by name: configs, hyperparameters, JSON |
| `set` | `{1, 2, 3}` | no | yes | no | uniqueness, membership tests, set algebra |

## Lists {B}

:::def
**List**: an ordered, **mutable** collection of items in square brackets. Items can be of any type and can repeat. *Example:* `scores = [88, 92, 79]`.

**Index**: the position of an item in a sequence, counting from 0 at the left. Negative indices count from the right: `x[-1]` is the last item.

**Mutable / immutable**: a mutable object can be changed after it is created (lists, dicts, sets, NumPy arrays, DataFrames); an immutable one cannot (numbers, strings, tuples).
:::


```py title="Creating, indexing and modifying a list"
scores = [88, 92, 79, 95]          #> Square brackets create a list. Items can be any type, even mixed.
print(scores[0], scores[-1])       #> Indexing works exactly like strings: first and last item.
scores[2] = 81                     #> Lists are *mutable*: you can replace an item in place.
scores.append(70)                  #> Add one item to the end.
scores.extend([99, 60])            #> Add every item from another iterable to the end.
scores.insert(1, 100)              #> Insert 100 at index 1, shifting the rest right.
print(scores)  #> Display the value of `scores`.
last = scores.pop()                #> Remove AND return the last item (60). `pop(i)` removes index i.
scores.remove(100)                 #> Remove the first item equal to 100 (by value, not index).
print(last, scores)  #> Show the popped value (60) and the list without 100 and 60.
print(len(scores), sum(scores), max(scores), 95 in scores)  #> Built-ins work on any list of numbers.
```

### Slicing {B}

:::def
**Slice**: a sub-sequence selected with `[start:stop:step]`. `start` is included and `stop` is excluded. *Example:* `x[1:4]` gives items 1, 2 and 3.
:::


Slicing `[start:stop:step]` returns a new list. The same syntax powers NumPy arrays and pandas, so it is worth mastering now.

```py title="Slices"
nums = list(range(10))     #> `range(10)` produces 0..9; `list()` materialises it: [0, 1, ..., 9].
print(nums[2:5])           #> Items at index 2, 3, 4 (stop is excluded).
print(nums[:3], nums[-3:]) #> First three, last three.
print(nums[::2])           #> Every second item: step = 2.
print(nums[::-1])          #> Reversed copy.
nums[1:3] = [10, 20]       #> Slice assignment replaces a section in place.
print(nums)  #> Display the value of `nums`.
```

### Sorting {B}

```py title="sorted() vs .sort()"
names = ["mia", "Ola", "zed", "ben"]  #> A list of names with mixed capitalisation.
print(sorted(names))                       #> `sorted()` returns a NEW sorted list. Uppercase sorts before lowercase!
print(sorted(names, key=str.lower))        #> `key` = function applied to each item before comparing: case-insensitive.
print(sorted(names, key=len, reverse=True))  #> Sort by length, longest first.
names.sort()                               #> `.sort()` sorts IN PLACE and returns None.
print(names)  #> Display the value of `names`.
```

:::warn `x = x.sort()` destroys your list
Because `.sort()` returns `None`, writing `scores = scores.sort()` replaces your data with `None`. Use either `scores.sort()` on its own line or `scores = sorted(scores)`.
:::

## Tuples {B}

:::def
**Tuple**: an ordered, **immutable** collection in round brackets. Use it for fixed groups of values. *Example:* `point = (3.0, 4.0)`, an array's `shape` such as `(100, 4)`.

**Unpacking**: assigning the items of a sequence to several names at once. *Example:* `x, y = point`.
:::


A tuple is an immutable list. Use it for things that belong together and should not change, such as coordinates, RGB colours, or the `shape` of an array.

```py title="Tuples and unpacking"
point = (3.0, 4.0)                 #> Parentheses create a tuple; actually the comma does: `3.0, 4.0` is also a tuple.
x, y = point                       #> *Unpacking*: assign each element to its own name.
print(x, y, len(point))  #> Both coordinates, and the tuple's length (2).
single = (5,)                      #> A one-element tuple needs a trailing comma; `(5)` is just the number 5.
print(type(single), type((5)))  #> Shows `tuple` for `(5,)` but `int` for `(5)`.
a, b = 1, 2  #> Assign two names at once.
a, b = b, a                        #> Swap two variables without a temporary: builds a tuple, then unpacks it.
print(a, b)  #> Now `a` is 2 and `b` is 1.
first, *rest = [10, 20, 30, 40]    #> Star-unpacking collects "everything else" into a list.
print(first, rest)  #> `first` is 10; `rest` is the list [20, 30, 40].
```

```py error title="Tuples cannot be changed"
shape = (28, 28)          #> E.g. the height and width of an image.
shape[0] = 32             #> Item assignment on a tuple raises TypeError. Create a new tuple instead.
```

## Dictionaries {B}

:::def
**Dictionary (dict)**: a mutable collection of **key–value pairs** in curly braces, for looking up a value by its key. Keys must be unique and immutable. *Example:* `params = {"C": 1.0, "kernel": "rbf"}`; `params["C"]` gives `1.0`.

**Key and value**: in a dictionary, the key is the label you look up (often a string) and the value is the data stored under it.
:::


A dictionary maps **keys** to **values**. Lookup by key is extremely fast regardless of size. Keys must be immutable (strings, numbers, tuples).

```py title="Building and querying a dictionary"
params = {"model": "svm", "C": 1.0, "kernel": "rbf"}   #> `key: value` pairs inside curly braces.
print(params["C"])                       #> Look up a value by its key.
params["gamma"] = 0.1                    #> Add a new key, or overwrite an existing one.
params["C"] = 10.0  #> Assigning to an existing key overwrites its value.
print(params.get("degree"))              #> `.get` returns None instead of crashing if the key is missing...
print(params.get("degree", 3))           #> ...or a default value you supply.
print("kernel" in params)                #> `in` checks KEYS, not values.
del params["gamma"]                      #> Remove a key.
for key, value in params.items():        #> `.items()` yields (key, value) pairs: the most common loop.
    print(f"{key:>7} -> {value}")  #> Right-align the key in 7 characters, then show its value.
print(list(params.keys()), list(params.values()))  #> All keys and all values, converted to lists for printing.
```

```py error title="The KeyError"
params = {"model": "svm"}  #> A dictionary with only one key.
print(params["learning_rate"])   #> A missing key raises KeyError. Use `.get()` or check `in` first.
```

### Nested structures {I}

Dictionaries and lists nest freely. This is exactly the shape of JSON data and of many configuration files.

```py title="A nested record"
experiment = {  #> Start a dictionary that spans several lines.
    "name": "baseline",  #> A plain string value.
    "metrics": {"acc": 0.91, "f1": 0.88},          #> A dict inside a dict.
    "seeds": [0, 1, 2],                             #> A list inside a dict.
}  #> Close the dictionary opened above.
print(experiment["metrics"]["f1"])                   #> Chain lookups from the outside in.
experiment["seeds"].append(3)                        #> Modify the inner list directly.
print(experiment["seeds"])  #> Display the value of `experiment["seeds"]`.
merged = {**experiment["metrics"], "auc": 0.95}      #> `**` unpacks a dict into a new one; then add a key.
print(merged)  #> Display the value of `merged`.
print(experiment["metrics"] | {"acc": 0.93})         #> `|` merges dicts (right side wins on clashes). Python 3.9+.
```

## Sets {B}

:::def
**Set**: an unordered, mutable collection of **unique** items in curly braces, with fast membership tests and set algebra (union, intersection, difference). *Example:* `{"cat", "dog"}`.
:::


```py title="Uniqueness and set algebra"
labels = ["cat", "dog", "cat", "bird", "dog"]  #> A list with repeated labels.
unique = set(labels)                    #> Converting to a set drops duplicates. Order is NOT preserved.
print(unique, len(unique))  #> Three unique labels remain.
a = {1, 2, 3, 4}  #> A set literal: curly braces with single values (no colons).
b = {3, 4, 5}  #> A second set that overlaps the first in 3 and 4.
print(a | b, a & b, a - b, a ^ b)       #> Union, intersection, difference, symmetric difference.
print(3 in a)                           #> Membership tests on sets are very fast even for millions of items.
train_ids, test_ids = {1, 2, 3}, {3, 4}  #> Sample IDs in a training set and a test set.
print("Overlap:", train_ids & test_ids) #> A real use: detect samples leaking between train and test sets!
```

:::warn `{}` is an empty dict, not a set
Write `set()` for an empty set. `{}` creates an empty dictionary.
:::

## Comprehensions {B}

:::def
**Comprehension**: a compact expression that builds a new list, dict or set from an iterable, optionally with a filter. *Example:* `[x ** 2 for x in nums if x > 0]`.
:::


A comprehension builds a new collection from an existing one in a single readable expression. They replace most simple `for` loops that build lists.

```py title="List, dict and set comprehensions"
nums = [1, 2, 3, 4, 5, 6]  #> The input list.
squares = [n ** 2 for n in nums]                  #> "n squared, for each n in nums".
evens = [n for n in nums if n % 2 == 0]           #> Add a filter with `if` at the end.
labels = ["even" if n % 2 == 0 else "odd" for n in nums]  #> An if/else *expression* goes BEFORE the `for`.
print(squares, evens, labels, sep="\n")           #> `sep="\n"` prints each argument on its own line.
word_len = {w: len(w) for w in ["numpy", "pandas", "keras"]}  #> Dict comprehension: `key: value for ...`.
print(word_len)  #> Display the value of `word_len`.
first_letters = {w[0] for w in ["apple", "avocado", "banana"]}  #> Set comprehension: unique first letters.
print(first_letters)  #> Display the value of `first_letters`.
pairs = [(i, j) for i in range(3) for j in range(2)]  #> Two `for`s = nested loops (outer loop first).
print(pairs)  #> Display the value of `pairs`.
```

## Mutability, references and copies {I}

:::def
**Reference**: a name pointing to an object. Assignment (`b = a`) creates a second reference to the same object, not a copy.

**Shallow copy / deep copy**: a shallow copy duplicates the outer container but shares the objects inside it; a deep copy (`copy.deepcopy`) duplicates everything, recursively.
:::


This is the single most common source of baffling bugs. **Assignment never copies.** It makes a second name for the same object.

```py title="Two names, one list"
a = [1, 2, 3]  #> Create a list and name it `a`.
b = a                    #> `b` is NOT a copy. Both names point at the same list.
b.append(4)              #> Modifying through `b`...
print(a)                 #> ...changes what `a` sees, because it is the same object.
print(a is b)            #> `is` confirms they are identical objects.
c = a.copy()             #> A real (shallow) copy. Also: `list(a)` or `a[:]`.
c.append(5)  #> Change only the copy.
print(a, c, a is c)      #> Now `a` is unaffected.
```

```py title="Shallow vs deep copies of nested data"
import copy                           #> Standard-library module for copying.
grid = [[0, 0], [0, 0]]  #> A list of two lists: a 2x2 grid.
shallow = grid.copy()                 #> Copies the OUTER list only; the inner lists are still shared.
deep = copy.deepcopy(grid)            #> Recursively copies everything.
grid[0][0] = 99                       #> Change an inner list.
print(shallow[0][0], deep[0][0])      #> The shallow copy sees 99; the deep copy still has 0.
bad = [[0] * 2] * 2                   #> Classic trap: `* 2` repeats a reference to the SAME inner list.
bad[0][0] = 1  #> Change the first element of the first row...
print(bad)                            #> Both rows changed! Build grids with a comprehension instead:
good = [[0] * 2 for _ in range(2)]    #> Each iteration creates a fresh inner list.
good[0][0] = 1  #> Change one cell: only the first row is affected.
print(good)  #> Display the value of `good`.
```

:::bug "I changed one variable and another one changed too"
You have two names for one mutable object (a list, dict, NumPy array or DataFrame). Make an explicit copy: `.copy()` for lists, dicts, NumPy arrays and DataFrames, or `copy.deepcopy()` for nested structures.
:::

:::ex Try it yourself
1. From `words = ["Data", "science", "IS", "fun", "data"]`, build the set of lowercase unique words.
2. Build a dict mapping each number 1–10 to `"fizz"` if divisible by 3, else the number itself.
3. Explain why `t = ([1, 2], 3); t[0].append(9)` works even though tuples are immutable.
:::
