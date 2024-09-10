```python
import matplotlib.pyplot as plt
import numpy as np


"""
绘制函数 f(x) = x^2 的图像。
"""
# 创建 x 值的范围：从 -10 到 10，总共 400 个点
x = np.linspace(-10, 10, 400)
# 计算每个 x 点的 f(x)
y = x ** 2

# 创建图形和轴
fig, ax = plt.subplots()
# 绘制 x 和 y
ax.plot(x, y)

# 设置图表标题和轴标签
ax.set_title("$f(x) = x^2$")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
# 显示图像
plt.show()
```