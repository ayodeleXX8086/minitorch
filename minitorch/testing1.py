from minitorch import TensorData, SimpleOps, Tensor, TensorBackend

tensor = Tensor(TensorData([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]), backend=TensorBackend(SimpleOps))
tensor2 = Tensor(TensorData([5,6,7,8], [1,4]), backend=TensorBackend(SimpleOps))


# def func(a, b):
#     return a * b
#
# print(tensor.to_string())
# result = tensor.reduce_add(0)
# print(result.to_string())
print(tensor.to_string())
print(tensor2.to_string())
result1 = tensor.backend.add_zip(tensor, tensor2)
print(result1.to_string())
# result = tensor.reduce_add(1)
# print(result.to_string())