from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from minitorch.operators import mul, log, log_back, relu
from minitorch.autodiff import Context
from minitorch.operators import inv, neg, exp, lt

if TYPE_CHECKING:
    from typing import Tuple

    from minitorch.scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        from minitorch.scalar import Scalar
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a,b)
        return mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        x, y = ctx.saved_values
        return y*d_output, x*d_output


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return d_output * (-1/pow(a, 2))


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a,)
        return neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -1*d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # sigmoid(a) = 1 / (1 + exp(-a))
        s = 1.0 / (1.0 + exp(-a))
        ctx.save_for_backward(s,)  # save sigmoid output for backward
        return s

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (s,) = ctx.saved_values
        # derivative: s * (1 - s)
        return d_output * s * (1 - s)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a,)
        return relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    "Exponential function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        val = exp(a)
        ctx.save_for_backward(val,)
        return float(val)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (val,) = ctx.saved_values
        return float(d_output * val)  # derivative of exp(a) is exp(a)


class LT(ScalarFunction):
    "Less-than function: 1.0 if x < y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Not differentiable; gradient is zero everywhere
        return 0.0, 0.0



class EQ(ScalarFunction):
    "Equal function: 1.0 if x == y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Not differentiable; gradient is zero everywhere
        return 0.0, 0.0
