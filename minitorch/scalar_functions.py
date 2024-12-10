from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Compute the backward pass."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Compute the forward pass."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), f"Expected return type float got {type(c)}"

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for log."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
# Implement the following functions for the ScalarFunction class:
# - Mul: Multiplication function $f(x, y) = x * y$
# - Inv: Inverse function $f(x) = 1/x$
# - Neg: Negation function $f(x) = -x$
# - Sigmoid: Sigmoid function $f(x) = 1 / (1 + exp(-x))$
# - ReLU: Rectified Linear Unit function $f(x) = max(0, x)$
# - Exp: Exponential function $f(x) = exp(x)$
# - lt: Less than function $f(x, y) = x < y$
# - eq: Equality function $f(x, y) = x == y$


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute backward pass for multiplication."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for inverse."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for inverse."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for negation."""
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for negation."""
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for sigmoid."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for sigmoid."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    """Rectified Linear Unit function $f(x) = max(0, x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for ReLU."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for ReLU."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute forward pass for exponential."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute backward pass for exponential."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute forward pass for less than."""
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute backward pass for less than."""
        a, b = ctx.saved_values
        return 0, 0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute forward pass for equality."""
        ctx.save_for_backward(a, b)
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute backward pass for equality."""
        a, b = ctx.saved_values
        return 0, 0
