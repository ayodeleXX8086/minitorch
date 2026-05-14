from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    plus_values_lst=list(vals)
    neg_values_lst = list(vals)
    plus_values_lst[arg]+=epsilon
    neg_values_lst[arg]-=epsilon
    result = (f(*plus_values_lst)-f(*neg_values_lst))/(2*epsilon)
    return result


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # visited = set()
    # order = []
    # def dfs(node):
    #     if node.unique_id in visited or node.is_constant():
    #         return
    #     visited.add(node.unique_id)
    #     for parent in node.parents:
    #         dfs(parent)
    #
    #     order.append(node)
    #
    # dfs(variable)
    # """
    # b=a*a*a
    # ...
    # where (a*a)=z
    # b=a*(a*a)
    # b=[a, z]
    # z=[a]
    # [b, z, a]
    # """
    # return reversed(order)
    visited = set()
    order = []
    stack = [(variable, False)]  # (node, processed_flag)

    while stack:
        node, processed = stack.pop()

        if node.unique_id in visited or node.is_constant():
            continue

        if processed:
            visited.add(node.unique_id)
            order.append(node)
        else:
            # Post-order: process after parents
            stack.append((node, True))
            for parent in node.parents:
                if parent.unique_id not in visited and not parent.is_constant():
                    stack.append((parent, False))

    return reversed(order)

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    lst: List[Variable] = list(topological_sort(variable))
    grads = {variable.unique_id: deriv}
    for dep in lst:
        if dep.unique_id not in grads:
            continue
        d_output = grads[dep.unique_id]
        if dep.is_leaf():
            dep.accumulate_derivative(d_output)
        else:
            for node, grad in dep.chain_rule(d_output):
                grads[node.unique_id]=grads.get(node.unique_id, 0)+grad



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values


if __name__ == "__main__":
    inv_func = lambda x: 1/x
    derivative = lambda x: -1/(x*x)
    x = 3
    print(inv_func(x))
    print(central_difference(inv_func, x,), derivative(x))
    print(abs(central_difference(inv_func, x, )-derivative(x))<1e-6)
