"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


# - mul
def mul(x: float, y: float) -> float:
    """Multiply two numbers x and y and return the product.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The product of x and y.

    """
    return x * y


# - id
def id(x: float) -> float:
    """Returns the input unchanged. (identity)

    Args:
    ----
        x: Passed number.

    Returns:
    -------
        The input number unchanged.

    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Add two numbers x and y and return the sum.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates the input number.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        The negated result of the input number.

    """
    return -x


# - lt
def lt(x: float, y: float) -> float:
    """Checks if x is less than y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if x is less than y. False otherwise.

    """
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """Checks if x equals y.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if x equals y. False otherwise.

    """
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The larger of x and y.

    """
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value. (within 1e-2)

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if | x - y | < 1e-2. False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


# - sigmoid
def sigmoid(x: float) -> float:
    """Calculates the sigmoid function of the input x.

    Args:
    ----
        x: input number.

    Returns:
    -------
        sigmoid(x)

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Applies the relu activation function to the input number.

    Args:
    ----
        x: input number

    Returns:
    -------
        x if x > 0. 0 otherwise

    """
    return x if x > 0 else 0.0


EPS = 1e-6


# - log
def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x: input number

    Returns:
    -------
        natural logarithm of x

    """
    return math.log(x + EPS)


# - exp
def exp(x: float) -> float:
    """Calculates the exponential function

    Args:
    ----
        x: input number

    Returns:
    -------
        exponential of x

    """
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg y.

    Args:
    ----
        x: input number
        y: the argument

    Returns:
    -------
        the derivative of log times a second arg y

    """
    return y / (x + EPS)


# - inv
def inv(x: float) -> float:
    """Takes the reciprocal of x.

    Args:
    ----
        x: input number

    Returns:
    -------
        reciprocal of x

    """
    return 1.0 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg y.

    Args:
    ----
        x: input number
        y: argument

    Returns:
    -------
        derivative of reciprocal times y

    """
    return -(1.0 / x**2) * y


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu times a second arg y.

    Args:
    ----
        x: input number
        y: argument

    Returns:
    -------
        derivative of relu times y

    """
    return y if x > 0 else 0.0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(
    func: Callable[[float], float],
) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        func: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns an iterable of floats.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of two iterables.

    Args:
    ----
        func: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes two iterables of floats and returns an iterable of floats.

    """

    def inner(
        iterable1: Iterable[float], iterable2: Iterable[float]
    ) -> Iterable[float]:
        """Applies a given function to each element of two iterables.

        Args:
        ----
            iterable1: First iterable of float values.
            iterable2: Second iterable of float values.

        Returns:
        -------
            An iterable containing the results of applying func to corresponding elements.

        """
        return [func(x, y) for x, y in zip(iterable1, iterable2)]

    return inner


def reduce(
    func: Callable[[float, float], float],
    initial: Optional[float] = None,
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        func: A function that takes two floats and returns a float.
        initial: An optional initial value for the reduction.

    Returns:
    -------
        A function that takes an iterable of floats and returns a single float.

    """

    def inner(iterable: Iterable[float]) -> float:
        """Reduces an iterable to a single value using a given function.

        Args:
        ----
            iterable: An iterable of float values.

        Returns:
        -------
            The result of reducing the iterable using func.

        """
        result = initial if initial is not None else 0.0
        for x in iterable:
            result = func(result, x)
        return result

    return inner


def negList(x: Iterable[float]) -> Iterable[float]:
    """Negates all elements of a list x.

    Args:
    ----
        x: An iterable of float values.

    Returns:
    -------
        An iterable containing the negated values of the input.

    """
    return map(neg)(x)


def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Adds two lists x and y element-wise.

    Args:
    ----
        x: First iterable of float values.
        y: Second iterable of float values.

    Returns:
    -------
        An iterable containing the element-wise sums of the inputs.

    """
    return zipWith(add)(x, y)


def sum(x: Iterable[float]) -> float:
    """Sums all elements of a list x.

    Args:
    ----
        x: An iterable of float values.

    Returns:
    -------
        The sum of all elements in the input iterable.

    """
    return reduce(add, initial=0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Multiplies all elements of a list x.

    Args:
    ----
        x: An iterable of float values.

    Returns:
    -------
        The product of all elements in the input iterable.

    """
    return reduce(mul, initial=1.0)(x)
