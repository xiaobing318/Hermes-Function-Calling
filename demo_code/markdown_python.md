```python
import numpy as np

def matrix_multiply(matrix_a, matrix_b):
    """
    计算两个矩阵的乘法。

    参数:
    matrix_a (np.ndarray): 第一个矩阵。
    matrix_b (np.ndarray): 第二个矩阵。

    返回:
    np.ndarray: 结果矩阵。
    """
    try:
        # 使用 numpy 的 dot 函数来计算矩阵乘法
        return np.dot(matrix_a, matrix_b)
    except ValueError as e:
        # 如果矩阵维度不匹配，则抛出错误
        print(f"Error: {e}")


# 定义两个矩阵
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[2, 0], [1, 2]])
# 计算乘积
result = matrix_multiply(matrix_a, matrix_b)
# 输出结果
if result is not None:
    print("矩阵 A:")
    print(matrix_a)
    print("矩阵 B:")
    print(matrix_b)
    print("矩阵乘积 A * B:")
    print(result)
```