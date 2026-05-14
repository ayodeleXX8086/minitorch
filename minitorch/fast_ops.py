from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from minitorch.tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from minitorch.tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from minitorch.tensor import Tensor
    from minitorch.tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_size = 1
        for s in out_shape:
            out_size *= s

        for i in prange(out_size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            in_index = np.zeros(len(in_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            offset = len(out_shape) - len(in_shape)
            for j in range(len(in_shape)):
                if in_shape[j] == 1:
                    in_index[j] = 0
                else:
                    in_index[j] = out_index[j + offset]
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)


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
        out_size = 1
        for s in out_shape:
            out_size *= s

        for i in prange(out_size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            b_index = np.zeros(len(b_shape), dtype=np.int32)

            to_index(i, out_shape, out_index)

            a_offset = len(out_shape) - len(a_shape)
            for d in range(len(a_shape)):
                if a_shape[d] == 1:
                    a_index[d] = 0
                else:
                    a_index[d] = out_index[d + a_offset]

            b_offset = len(out_shape) - len(b_shape)
            for d in range(len(b_shape)):
                if b_shape[d] == 1:
                    b_index[d] = 0
                else:
                    b_index[d] = out_index[d + b_offset]

            pos_out = index_to_position(out_index, out_strides)
            pos_a = index_to_position(a_index, a_strides)
            pos_b = index_to_position(b_index, b_strides)
            out[pos_out] = fn(a_storage[pos_a], b_storage[pos_b])

    return njit(parallel=True)(_zip)


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
        out_size = 1
        for s in out_shape:
            out_size *= s

        for i in prange(out_size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)

            for d in range(len(a_shape)):
                a_index[d] = out_index[d]

            a_index[reduce_dim] = 0
            acc = a_storage[index_to_position(a_index, a_strides)]

            for r in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = r
                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])

            out[index_to_position(out_index, out_strides)] = acc

    return njit(parallel=True)(_reduce)


def _tensor_matrix_multiply(
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
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for n in prange(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                acc = 0.0
                for k in range(a_shape[2]):
                    a_pos = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]
                    b_pos = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]
                    acc += a_storage[a_pos] * b_storage[b_pos]
                out[n * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
