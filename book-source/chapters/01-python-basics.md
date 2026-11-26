@part 1

# Values, variables and types

Everything a program does comes down to storing values, combining them and making decisions about them. This chapter covers the atoms: numbers, text, true/false values, and the names you give them.

## Variables: names for values {B}

:::def
**Variable**: a name that refers to a value stored in the computer's memory. *Example:* `age = 29`.

**Value**: a piece of data, such as `29`, `3.14` or `"Ada"`.

**Data type (type)**: the kind of a value, which decides what you can do with it: `int`, `float`, `str`, `bool`, `NoneType` and others. Check it with `type(x)`.

**Assignment**: binding a name to a value with `=`. `x = x + 1` computes the right side first, then re-points `x` to the result.
:::


A **variable** is a name that refers to a value stored in memory. You create one with `=`, the *assignment operator*. It does not mean "equals" as in maths; read it as "now refers to".

```py title="Creating and reusing variables"
age = 29                     #> Create the name `age` and point it at the integer value 29.
height_m = 1.74              #> A number with a decimal point is a *float* (floating-point number).
name = "Ada"                 #> Text in quotes is a *string*. Single or double quotes both work.
is_student = True            #> `True` and `False` (capitalised) are the two *Boolean* values.
age = age + 1                #> Right side is evaluated first (29 + 1), then `age` is re-pointed to 30.
print(name, age, height_m, is_student)  #> Print several values in one call; Python inserts spaces.
print(type(age), type(height_m), type(name), type(is_student))  #> `type()` reveals what kind of value each is.
```

**Naming rules.** Names may contain letters, digits and underscores, cannot start with a digit, and are case-sensitive (`Age` and `age` are different). By convention Python uses `snake_case` for variables and functions. Avoid shadowing built-in names such as `list`, `str`, `sum`, `max` or `type`.

:::warn Don't name a variable `list` or `sum`
`sum = 10` silently replaces the built-in `sum()` function for the rest of your session. Later, `sum([1, 2])` fails with `TypeError: 'int' object is not callable`. If that happens, rename your variable and restart the kernel.
:::

## Numbers and arithmetic {B}

:::def
**Integer (int)**: a whole number with no decimal point, of unlimited size. *Example:* `42`, `-7`.

**Floating-point number (float)**: a number with a decimal point, stored approximately in binary with about 15–17 significant digits. *Example:* `3.14`, `1e-5`.

**Operator**: a symbol that performs an operation on values: `+ - * / // % **` for arithmetic, `== != < >` for comparison, `and or not` for logic.

**Expression**: any piece of code that produces a value, such as `2 * (x + 1)` or `len(name) > 3`.
:::


Python has two everyday numeric types: `int` (whole numbers, unlimited size) and `float` (decimals, about 15–17 significant digits).

```py title="Arithmetic operators"
a, b = 17, 5                 #> *Tuple unpacking*: assigns 17 to `a` and 5 to `b` in one line.
print(a + b, a - b, a * b)   #> Addition, subtraction, multiplication.
print(a / b)                 #> True division **always** returns a float: 3.4.
print(a // b)                #> Floor division: rounds down to a whole number, giving 3.
print(a % b)                 #> Modulo: the remainder after division (17 = 3*5 + 2), giving 2.
print(a ** 2)                #> Exponentiation: 17 squared. (`^` is NOT power in Python; it is bitwise XOR.)
print(-7 // 2, -7 % 2)       #> Floor division rounds toward minus infinity, so -7 // 2 is -4, and the remainder is 1.
print(round(3.14159, 2), abs(-4), max(3, 9, 2), min(3, 9, 2))  #> Useful built-in numeric functions.
```

### Floating-point surprises {I}

Floats are stored in binary, and most decimal fractions (like 0.1) cannot be represented exactly. This matters when you compare results.

```py title="Never compare floats with =="
import math                         #> The standard-library `math` module: sqrt, log, isclose, pi, and more.
x = 0.1 + 0.2                       #> Looks like it should be 0.3...
print(x)                            #> ...but the stored value is 0.30000000000000004.
print(x == 0.3)                     #> So exact comparison is False.
print(math.isclose(x, 0.3))         #> `isclose` compares with a tolerance: the correct way.
print(math.sqrt(16), math.log(math.e), math.pi)  #> Square root, natural logarithm, and the constant pi.
print(float("inf") > 1e308, math.isnan(float("nan")))  #> Infinity and NaN ("not a number") exist as floats.
```

:::bug Your model's loss or accuracy is `nan`
`nan` spreads through every calculation it touches (`nan + 1` is `nan`). It usually comes from missing values in the data, dividing by zero, or taking `log(0)`. Part VIII has a full checklist.
:::

## Strings: working with text {B}

:::def
**String (str)**: an ordered, immutable sequence of characters written in quotes. *Example:* `"Machine Learning"`.

**Method**: a function that belongs to an object and is called with a dot: `text.upper()`, `text.split(",")`.

**f-string**: a string prefixed with `f` in which expressions inside `{}` are evaluated and inserted. *Example:* `f"Accuracy: {acc:.2%}"`.
:::


A string is an ordered, **immutable** sequence of characters. Immutable means you cannot change a string in place; every "modification" creates a new string.

```py title="Indexing, slicing and common methods"
s = "Machine Learning"            #> A 16-character string (the space counts).
print(s[0], s[-1])                #> Index from 0 at the left, or from -1 at the right: 'M' and 'g'.
print(s[0:7])                     #> Slice `[start:stop]` includes start, excludes stop: characters 0..6, 'Machine'.
print(s[8:], s[:7], s[::-1])      #> Omitted start/stop mean "from the beginning/to the end"; step -1 reverses.
print(s.lower(), s.upper())       #> Methods return NEW strings; `s` itself is unchanged.
print(s.replace("Machine", "Deep"))  #> Replace every occurrence of a substring.
print(s.split(" "))               #> Split on a separator into a list of pieces.
print("-".join(["a", "b", "c"]))  #> `join` is the reverse: glue a list of strings with a separator.
print("  padded  ".strip() + "|") #> `strip()` removes leading/trailing whitespace; essential when cleaning data.
print("Learn" in s, s.startswith("Mach"), s.find("L"), len(s))  #> Membership, prefix test, position, length.
```

### f-strings: the modern way to format {B}

Put an `f` before the opening quote and any expression inside `{}` is evaluated and inserted. A format specification after a colon controls how it looks.

```py title="Formatting numbers for reports"
model, acc, n = "RandomForest", 0.934567, 12500    #> Three values to report.
print(f"{model} accuracy: {acc}")                  #> Plain insertion.
print(f"{model} accuracy: {acc:.2f}")              #> `.2f` = fixed-point with 2 decimals: 0.93.
print(f"{model} accuracy: {acc:.1%}")              #> `.1%` multiplies by 100 and adds a percent sign: 93.5%.
print(f"Samples: {n:,}")                           #> `,` inserts thousands separators: 12,500.
print(f"|{model:<14}|{acc:>8.3f}|")                #> `<14` left-aligns in 14 chars; `>8` right-aligns in 8.
print(f"{acc=}")                                   #> `=` prints the expression AND its value: great for debugging.
print(f"{1e-5:.2e}")                               #> Scientific notation with 2 decimals.
```

## Booleans, comparisons and logic {B}

:::def
**Boolean (bool)**: one of the two truth values `True` and `False`. Comparisons such as `x > 5` produce Booleans.

**Truthy and falsy**: how a value behaves in an `if`: `0`, `0.0`, `""`, `[]`, `{}` and `None` count as false (falsy); everything else counts as true (truthy).
:::


Comparisons produce `True` or `False`. Combine them with `and`, `or`, `not`.

```py title="Comparisons and logical operators"
x = 7                                   #> A test value.
print(x > 5, x == 7, x != 7)            #> Greater than, equal to (two `=`!), not equal to.
print(3 < x <= 10)                      #> Python allows chained comparisons: same as 3 < x and x <= 10.
print(x > 5 and x % 2 == 0)             #> `and` is True only if both sides are True. 7 is odd, so False.
print(x > 5 or x % 2 == 0)              #> `or` is True if at least one side is True.
print(not x > 5)                        #> `not` flips True and False.
print(bool(0), bool(""), bool([]), bool(None))  #> "Falsy" values: zero, empty string, empty list, None.
print(bool(42), bool("hi"), bool([0]))  #> Everything else is "truthy", even a list containing a zero.
```

:::tip `is` versus `==`
`==` asks "do these have equal value?" `is` asks "are these the very same object in memory?" Use `is` only for `None` (`if result is None:`), and `==` for everything else.
:::

## Converting between types {B}

```py title="Casting and the errors it can raise"
print(int("42") + 1)          #> Convert a numeric string to an int before doing maths with it.
print(float("3.5") * 2)       #> Strings containing decimals convert with `float()`.
print(str(3.5) + " kg")       #> Convert numbers to text before joining them to other text.
print(int(3.99))              #> `int()` truncates toward zero: 3, not 4. Use `round()` to round.
print(int(float("3.7")))      #> A decimal string must go through `float` first; `int("3.7")` would fail.
```

```py error title="Mixing text and numbers"
age = 30                       #> An integer.
print("Age: " + age)           #> `+` cannot join a string and an int. Fix: "Age: " + str(age), or an f-string.
```

## `None`: the value for "nothing" {B}

:::def
**None**: Python's single object meaning "no value". Functions without a `return` give back `None`. Test for it with `is None`.
:::


`None` is Python's single "no value here" object. Functions without a `return` give back `None`, and it is often used as a default meaning "not provided".

```py title="Checking for None"
result = None                     #> Nothing computed yet.
if result is None:                #> The idiomatic test for None uses `is`.
    print("No result yet")  #> Runs only because `result` is still None.
print(type(None))                 #> `None` has its own type, NoneType.
```

## Comments, input and getting help {B}

```py norun title="Comments, reading input, built-in help"
# A comment starts with # and runs to the end of the line.   #> Comments are ignored by Python; write them for humans.
name = input("Your name: ")       #> `input()` pauses, waits for the user to type, and ALWAYS returns a string.
age = int(input("Your age: "))    #> Convert immediately if you need a number.
help(len)                         #> Print the documentation of any function, class or module.
dir("text")                       #> List every attribute and method available on an object.
```

:::ex Try it yourself
1. Store the radius `r = 2.5` and print the circle's area to 3 decimal places using `math.pi` and an f-string.
2. Given `email = "  Ada.Lovelace@Example.COM "`, produce `"ada.lovelace@example.com"` in one line.
3. Predict, then check: `7 // -2`, `bool("False")`, `"3" * 3`.
:::
