# Control flow: decisions and loops

Programs become useful when they can choose between actions and repeat work. Python marks which lines belong to an `if`, `for` or `while` by **indentation** (four spaces by convention), not by braces.

## if, elif, else {B}

:::def
**Control flow**: the order in which a program's statements run, changed by decisions (`if`) and repetition (loops).

**Conditional statement**: code that runs only when a condition is true: `if`, followed by optional `elif` and `else` branches.

**Indentation**: the spaces at the start of a line. In Python they define which lines belong to a block (4 spaces by convention).
:::


```py title="Choosing a branch"
p_value = 0.032                         #> A value to test.
if p_value < 0.01:                      #> The condition is followed by a colon.
    verdict = "strong evidence"         #> Indented lines run only when the condition is True.
elif p_value < 0.05:                    #> `elif` = "else if": checked only if earlier conditions were False.
    verdict = "moderate evidence"  #> Runs when 0.01 <= p < 0.05 (our case, p = 0.032).
else:                                   #> Runs when no condition above was True.
    verdict = "insufficient evidence"  #> Runs when p >= 0.05.
print(verdict)  #> Display the value of `verdict`.
label = "pass" if p_value < 0.05 else "fail"   #> One-line *conditional expression* for simple choices.
print(label)  #> Display the value of `label`.
```

:::bug `IndentationError: expected an indented block`
Every line ending in `:` must be followed by an indented block. Mixing tabs and spaces also triggers indentation errors. Configure your editor to insert 4 spaces when you press Tab.
:::

## for loops {B}

:::def
**Loop**: code that repeats. A `for` loop runs once for each item of an iterable; a `while` loop repeats while a condition is true.

**Iterable**: any object you can loop over: lists, tuples, strings, dicts, sets, `range`, files, arrays, DataFrame columns.
:::


A `for` loop visits each item of any *iterable*: a list, string, dict, range, file, DataFrame column, and so on.

```py title="Looping patterns you will use daily"
fruits = ["apple", "banana", "cherry"]  #> A list to loop over.
for fruit in fruits:                         #> `fruit` takes each value in turn.
    print(fruit.upper())  #> This indented line runs once per item.
for i in range(3):                           #> range(3) gives 0, 1, 2. range(start, stop, step) also exists.
    print("iteration", i)  #> Prints iteration 0, 1, 2.
for i, fruit in enumerate(fruits, start=1):  #> `enumerate` gives (index, item) pairs; `start` sets the first index.
    print(i, fruit)  #> Index (starting at 1) and item.
prices = [1.2, 0.5, 3.0]  #> A second list, aligned with `fruits` by position.
for fruit, price in zip(fruits, prices):     #> `zip` walks several sequences in parallel, stopping at the shortest.
    print(f"{fruit}: ${price:.2f}")  #> The literal `$` followed by the price with 2 decimals.
```

:::warn Don't write `for i in range(len(x))` just to get items
`for item in x:` is clearer. If you need the index too, use `enumerate(x)`. If you need two lists side by side, use `zip(a, b)`.
:::

## while loops, break and continue {B}

A `while` loop repeats as long as its condition is True. It suits situations where you don't know the number of iterations in advance, such as iterating until convergence.

```py title="Iterate until convergence"
x = 10.0                       #> Starting guess for the square root of 2 (Newton's method).
steps = 0  #> A counter for the number of iterations.
while abs(x * x - 2) > 1e-10:  #> Keep going while the error is larger than our tolerance.
    x = (x + 2 / x) / 2        #> Newton update: average x and 2/x.
    steps += 1                 #> `+=` is shorthand for `steps = steps + 1`.
print(f"sqrt(2) ~ {x:.10f} after {steps} steps")  #> Show the result to 10 decimal places and how many steps it took.
```

```py title="break, continue and loop-else"
for n in [3, 8, -1, 12, 5]:  #> Loop over `[3, 8, -1, 12, 5]`, naming each item `n`.
    if n < 0:  #> Run the indented block only if `n < 0` is true.
        print("negative value found, stopping")  #> Reached when n is -1.
        break                  #> Leave the loop immediately.
    if n > 10:  #> Run the indented block only if `n > 10` is true.
        continue               #> Skip the rest of this iteration and go to the next item.
    print("processing", n)  #> Runs for 3 and 8 (12 is skipped by `continue`; the loop stops at -1).
for n in [2, 4, 6]:  #> Loop over `[2, 4, 6]`, naming each item `n`.
    if n % 2 == 1:  #> Run the indented block only if `n % 2 == 1` is true.
        break  #> Leave the loop immediately.
else:                          #> A loop's `else` runs only if the loop finished WITHOUT `break`.
    print("all numbers were even")  #> Printed because the loop never hit `break`.
```

:::bug My `while` loop never ends
The condition never becomes False: usually you forgot to update the variable inside the loop. In Jupyter press the stop button (or *Interrupt kernel*); in a terminal press `Ctrl+C`. Add a safety cap such as `while err > tol and steps < 10_000:`.
:::

## match: structural pattern matching {I}

Python 3.10 added `match`, a clean way to branch on the *shape* of data.

```py title="Dispatching on a value"
def describe(task):  #> Define a function `describe` that takes `task`.
    match task:                                    #> Compare `task` against each `case` pattern in order.
        case "regression" | "forecasting":        #> `|` means "either of these".
            return "predict a number"  #> Send `"predict a number"` back to the caller.
        case "classification":  #> Matches the exact string "classification".
            return "predict a category"  #> Send `"predict a category"` back to the caller.
        case {"type": "clustering", "k": k}:      #> Match a dict shape and capture the value of "k".
            return f"group into {k} clusters"  #> Send `f"group into {k} clusters"` back to the caller.
        case _:                                    #> `_` is the wildcard: matches anything else.
            return "unknown task"  #> Send `"unknown task"` back to the caller.
print(describe("forecasting"))  #> Matched by the first case.
print(describe({"type": "clustering", "k": 4}))  #> Matched by the dict pattern; k is captured as 4.
print(describe("translation"))  #> No pattern matches, so the wildcard case answers.
```

## Iterables, iterators and why loops are lazy {A}

:::def
**Iterator**: an object that produces the items of an iterable one at a time and remembers where it is. It can be consumed only once.

**Lazy evaluation**: computing values only when they are requested, instead of all at once. `range`, `zip`, `map` and generators are lazy.
:::


`range`, `zip`, `enumerate`, `map`, dictionary views and file objects don't build a full list in memory; they produce items one at a time when asked. That is why `range(10**12)` is instant.

```py title="Lazy objects"
r = range(10**12)               #> No trillion-item list is created; `range` computes values on demand.
print(r[5], len(r))             #> Still supports indexing and len().
z = zip([1, 2], ["a", "b"])  #> Pair up two lists lazily.
print(z)                        #> Printing a zip shows the object, not its contents...
print(list(z))                  #> ...convert to a list to see the pairs.
print(list(z))                  #> An iterator is exhausted after one pass: the second list is empty!
```

:::warn One pass only
`zip`, `map`, `filter` and generator objects can be consumed only once. If you need the values twice, store them: `pairs = list(zip(a, b))`.
:::

:::ex Try it yourself
1. Loop over `range(1, 31)` and print "Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both, else the number.
2. Use `zip` to compute the dot product of `[1, 2, 3]` and `[4, 5, 6]` in a loop. (Answer: 32.)
3. Write a `while` loop that halves `lr = 1.0` until it is below `0.01` and counts how many halvings it took.
:::
