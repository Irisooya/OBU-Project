import matplotlib.pyplot as plt

# 数据表中的数据
tiou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.7, 0.7]
area1 = [0.76, 0.67, 0.57, 0.48, 0.38, 0.28, 0.19, 30.04]
area2 = [0.77, 0.70, 0.64, 0.56, 0.46, 0.37, 0.25, 35.70]
area3 = [0.82, 0.78, 0.74, 0.68, 0.59, 0.50, 0.37, 62.28]

# 只取前8个元素，因为 tiou 有10个元素，而 area1/area2/area3 只有8个元素
tiou = tiou[:8]

# 绘制折线图
plt.plot(tiou, area1, label='Area1')
plt.plot(tiou, area2, label='Area2')
plt.plot(tiou, area3, label='Area3')

# 添加标签和标题
plt.xlabel('tiou')
plt.ylabel('Area Value')
plt.title('Line Graph of Areas')

# 添加图例
plt.legend()

# 显示图形
plt.show()
