# Classes and objects

Every value in Python is an **object**: it has data (*attributes*) and behaviour (*methods*). A **class** is the blueprint you write to create your own kind of object. You don't need classes for small scripts, but you need to read them, because scikit-learn models, Keras layers and pandas DataFrames are all objects with the same anatomy.

## Your first class {B}

:::def
**Object**: a bundle of data (attributes) and behaviour (methods). Every value in Python is an object.

**Class**: a blueprint for creating objects of a new type. Defined with `class Name:`.

**Instance**: one object created from a class. *Example:* `ds = Dataset(...)` makes `ds` an instance of `Dataset`.

**Attribute**: a value stored on an object, accessed with a dot: `ds.name`.

**Constructor (`__init__`)**: the special method that runs automatically when a new instance is created, usually to set its attributes.

**self**: the conventional name of a method's first parameter: the object the method was called on.
:::


```py title="A class that stores and summarises measurements"
class Dataset:                                    #> `class` + a CapitalisedName defines a new type.
    """A tiny container for numeric samples."""  #> The class docstring.

    def __init__(self, name, values):             #> `__init__` runs automatically when an object is created.
        self.name = name                          #> `self` is the object being built; `self.name` is an *attribute*.
        self.values = list(values)                #> Store a copy so outside changes don't affect us.

    def mean(self):                               #> A *method*: a function defined in a class. Its first parameter is always `self`.
        return sum(self.values) / len(self.values)  #> Send `sum(self.values) / len(self.values)` back to the caller.

    def add(self, x):  #> Define the method `add`; besides `self` (the object) it takes `x`.
        self.values.append(x)                     #> Methods can change the object's state.
        return self                               #> Returning `self` allows chaining: ds.add(1).add(2)

ds = Dataset("heights", [1.62, 1.75, 1.80])        #> Create an *instance*; Python calls __init__ for you.
print(ds.name, ds.values)                         #> Access attributes with a dot.
print(round(ds.mean(), 3))                        #> Call methods with a dot; `self` is passed automatically.
ds.add(1.70).add(1.68)  #> Chained calls work because `add` returns `self`.
print(len(ds.values), round(ds.mean(), 3))  #> Now 5 values, and an updated mean.
```

:::bug `TypeError: mean() takes 0 positional arguments but 1 was given`
You forgot `self` in a method definition. Every regular method must take `self` as its first parameter, because Python passes the object in automatically.
:::

## Special ("dunder") methods {I}

:::def
**Dunder method**: a method whose name starts and ends with double underscores, such as `__repr__` or `__add__`, that lets objects work with Python's built-in syntax.
:::


Methods with double underscores let your objects work with Python's built-in syntax: `print`, `len`, `+`, `==`, `[]`, and so on.

```py title="Making objects print, measure and compare nicely"
class Vector2:  #> Define a class `Vector2`.
    def __init__(self, x, y):  #> The constructor: runs automatically when a new object is created; it takes `x` and `y`.
        self.x, self.y = x, y  #> Store both coordinates as attributes.

    def __repr__(self):                       #> What you see when you print the object or inspect it in a notebook.
        return f"Vector2({self.x}, {self.y})"  #> Send `f"Vector2({self.x}, {self.y})"` back to the caller.

    def __add__(self, other):                 #> Defines `v + w`.
        return Vector2(self.x + other.x, self.y + other.y)  #> Send `Vector2(self.x + other.x, self.y + other.y)` back to the caller.

    def __mul__(self, k):                     #> Defines `v * k` for a number k.
        return Vector2(self.x * k, self.y * k)  #> Send `Vector2(self.x * k, self.y * k)` back to the caller.

    def __abs__(self):                        #> Defines `abs(v)`: here, the vector's length.
        return (self.x ** 2 + self.y ** 2) ** 0.5  #> Send `(self.x ** 2 + self.y ** 2) ** 0.5` back to the caller.

    def __eq__(self, other):                  #> Defines `==`. Without it, == compares identity.
        return (self.x, self.y) == (other.x, other.y)  #> Send `(self.x, self.y) == (other.x, other.y)` back to the caller.

v, w = Vector2(3, 4), Vector2(1, 1)  #> Two vectors.
print(v + w, v * 2, abs(v), v == Vector2(3, 4))  #> Uses __add__, __mul__, __abs__ and __eq__, with __repr__ for display.
```

## Inheritance {I}

:::def
**Inheritance**: creating a new class (subclass) from an existing one (parent or base class), reusing its code and overriding what differs.
:::


A class can **inherit** from another, reusing its code and overriding what differs. scikit-learn is organised this way: every estimator inherits from `BaseEstimator`.

```py title="A base model and two subclasses"
class Model:  #> Define a class `Model`.
    def __init__(self, name):  #> The constructor: runs automatically when a new object is created; it takes `name`.
        self.name = name  #> Store the name on the object.
        self.fitted = False  #> Not fitted yet.

    def fit(self, X, y):  #> Define the method `fit`; besides `self` (the object) it takes `X` and `y`.
        self.fitted = True  #> Mark the model as fitted.
        return self  #> Return the object itself, so calls can be chained (the scikit-learn convention).

    def predict(self, X):  #> Define the method `predict`; besides `self` (the object) it takes `X`.
        raise NotImplementedError("subclasses must implement predict")   #> Forces subclasses to provide their own version.

class MeanRegressor(Model):                     #> `(Model)` means MeanRegressor inherits from Model.
    def fit(self, X, y):  #> Define the method `fit`; besides `self` (the object) it takes `X` and `y`.
        self.mean_ = sum(y) / len(y)            #> Learned attributes end in `_`, the scikit-learn convention.
        return super().fit(X, y)                #> `super()` calls the parent's version (sets fitted=True).

    def predict(self, X):  #> Define the method `predict`; besides `self` (the object) it takes `X`.
        return [self.mean_ for _ in X]          #> Always predicts the training mean: a *baseline* model.

class MajorityClassifier(Model):  #> Define a class `MajorityClassifier` that inherits from `Model`.
    def fit(self, X, y):  #> Define the method `fit`; besides `self` (the object) it takes `X` and `y`.
        self.majority_ = max(set(y), key=y.count)   #> Most frequent label.
        return super().fit(X, y)  #> Send `super().fit(X, y)` back to the caller.

    def predict(self, X):  #> Define the method `predict`; besides `self` (the object) it takes `X`.
        return [self.majority_] * len(X)  #> Send `[self.majority_] * len(X)` back to the caller.

reg = MeanRegressor("baseline").fit([[1], [2], [3]], [10, 20, 60])  #> Create and fit in one line; the training mean of y is 30.
clf = MajorityClassifier("baseline").fit([[0]] * 5, ["a", "b", "b", "c", "b"])  #> "b" is the most frequent label.
print(reg.predict([[4], [5]]), reg.fitted)  #> Predicts 30.0 for every input; `fitted` is True.
print(clf.predict([[0], [0]]))  #> Always "b".
print(isinstance(reg, Model), isinstance(reg, MajorityClassifier))   #> An instance of a subclass is also an instance of its parent.
```

:::key This is the scikit-learn API in miniature
`fit(X, y)` learns from data and returns `self`; `predict(X)` uses what was learned; learned attributes end with an underscore (`coef_`, `classes_`). Once you have seen this pattern, every model in Part V will feel the same.
:::

## Properties, class attributes and static methods {A}

```py title="Controlled attributes"
class Experiment:  #> Define a class `Experiment`.
    count = 0                                   #> A *class attribute*: shared by all instances.

    def __init__(self, lr):  #> The constructor: runs automatically when a new object is created; it takes `lr`.
        self.lr = lr                            #> Goes through the property setter below, so it is validated.
        Experiment.count += 1  #> Update the shared class attribute every time an object is created.

    @property  #> Make the next method readable like an attribute.
    def lr(self):                               #> `@property` makes a method look like an attribute: exp.lr
        return self._lr  #> Send `self._lr` back to the caller.

    @lr.setter  #> Define what happens when someone assigns to `lr`.
    def lr(self, value):                        #> Runs on assignment: exp.lr = 0.1
        if value <= 0:  #> Run the indented block only if `value <= 0` is true.
            raise ValueError("learning rate must be positive")  #> Reject the value with a clear message.
        self._lr = value                        #> The real value lives in a "private" attribute (leading underscore).

    @staticmethod  #> The next method needs neither the object nor the class.
    def default_seeds():                        #> A *static method* needs neither self nor the class.
        return [0, 1, 2]  #> Send `[0, 1, 2]` back to the caller.

    @classmethod  #> The next method receives the class (`cls`) instead of an object.
    def fast(cls):                              #> A *class method* receives the class: handy as an alternative constructor.
        return cls(lr=0.1)  #> Send `cls(lr=0.1)` back to the caller.

e = Experiment(0.01)  #> First object: count becomes 1.
f = Experiment.fast()  #> Second object, via the class-method constructor: count becomes 2.
print(e.lr, f.lr, Experiment.count, Experiment.default_seeds())  #> 0.01, 0.1, 2 and [0, 1, 2].
try:  #> Start a block whose errors we want to catch.
    e.lr = -1                                   #> The setter rejects invalid values.
except ValueError as err:  #> Catch the error raised by the setter.
    print("Rejected:", err)  #> Show the setter's message.
```

## Dataclasses: classes without boilerplate {I}

:::def
**Dataclass**: a class decorated with `@dataclass`, which automatically writes `__init__`, `__repr__` and `__eq__` from the annotated fields.
:::


For classes that mainly hold data, such as a run configuration, `@dataclass` writes `__init__`, `__repr__` and `__eq__` for you.

```py title="A configuration object"
from dataclasses import dataclass, field, asdict  #> Import `dataclass`, `field` and `asdict` from `dataclasses`.

@dataclass  #> Generate __init__, __repr__ and __eq__ from the annotated fields below.
class TrainConfig:  #> Define a class `TrainConfig`.
    model: str                                   #> A required field (no default). The type hint is required syntax here.
    lr: float = 1e-3                             #> Fields with defaults must come after fields without.
    epochs: int = 20  #> Another field with a default.
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])   #> Mutable defaults need `default_factory`.

cfg = TrainConfig("cnn", epochs=50)  #> Create a config, overriding one default.
print(cfg)                                       #> A readable __repr__ for free.
print(asdict(cfg))                               #> Convert to a dict, e.g. to save as JSON next to your results.
print(cfg == TrainConfig("cnn", epochs=50))      #> Field-by-field equality for free.
```

:::research Store every run's configuration
Save the configuration object (as JSON) alongside each experiment's results. Months later you will know exactly which learning rate, seed and data version produced the number in Table 2.
:::

:::ex Try it yourself
1. Write a class `Counter` with methods `add(item)` and `most_common()`.
2. Give `Dataset` a `__len__` method so that `len(ds)` works.
3. Subclass `Model` as `MedianRegressor`.
:::
