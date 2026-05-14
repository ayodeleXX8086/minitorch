from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

from typing_extensions import Protocol

from minitorch import operators
from minitorch.tensor_data import (
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from minitorch.tensor import Tensor
    from minitorch.tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        return SimpleOps.zip(operators.mul)(a, b)

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_size = int(operators.reduce(operators.mul, start=1)(out_shape))
        out_index = [0] * len(out_shape)
        in_index = [0]*len(in_shape)
        for i in range(out_size):
            to_index(i, out_shape, out_index)
            offset = len(out_shape) - len(in_index)
            for j in range(len(in_index)):
                if in_shape[j]==1:
                    in_index[j]=0
                else:
                    in_index[j]=out_index[j+offset]
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        # Compute total number of elements in output
        out_size = int(operators.reduce(operators.mul, start=1)(out_shape))

        # Indices
        out_index = [0] * len(out_shape)
        a_index = [0] * len(a_shape)
        b_index = [0] * len(b_shape)

        for i in range(out_size):
            # Step 1: get output index
            to_index(i, out_shape, out_index)

            offset = len(out_shape) - len(a_shape)
            # Step 2: map to a_index (with broadcasting)
            for d in range(len(a_shape)):
                if a_shape[d] == 1:
                    a_index[d] = 0
                else:
                    a_index[d] = out_index[d+offset]

            offset = len(out_shape) - len(b_shape)
            # Step 3: map to b_index (with broadcasting)
            for d in range(len(b_shape)):
                if b_shape[d] == 1:
                    b_index[d] = 0
                else:
                    b_index[d] = out_index[d+offset]

            # Step 4: compute positions
            pos_out = index_to_position(out_index, out_strides)
            pos_a = index_to_position(a_index, a_strides)
            pos_b = index_to_position(b_index, b_strides)

            # Step 5: apply function
            out[pos_out] = fn(a_storage[pos_a], b_storage[pos_b])
    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:

        # Total number of elements in output
        out_size = int(operators.reduce(operators.mul, 1)(out_shape))

        # Temporary index holders
        out_index = [0] * len(out_shape)
        a_index = [0] * len(a_shape)

        for i in range(out_size):
            # Convert flat index -> multidimensional index for output
            to_index(i, out_shape, out_index)

            # Copy out_index into a_index
            for d in range(len(a_shape)):
                a_index[d] = out_index[d]

            # Initialize accumulator with first element along reduce_dim
            a_index[reduce_dim] = 0
            acc = a_storage[index_to_position(a_index, a_strides)]

            # Loop over the reduce dimension
            for r in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = r
                pos = index_to_position(a_index, a_strides)
                acc = fn(acc, a_storage[pos])

            # Write result to output
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
