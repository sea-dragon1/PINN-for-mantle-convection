import torch
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 选择GPU或CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 从文件加载已经训练完成的模型
model_loaded = torch.load('model.pth', map_location=device)
model_loaded.eval()  # 设置模型为evaluation状态

# 生成时空网格
h = 0.01
k = 0.01
x = torch.arange(-1, 1, h)
y = torch.arange(-1, 1, h)
t = torch.arange(0, 1, k)
X = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T
X = X.to(device)

# 计算该时空网格对应的预测值
with torch.no_grad():
    T_pred = model_loaded(X).reshape(len(x), len(y), len(t)).cpu().numpy()

# 绘制计算结果
sns.set_style('whitegrid')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection = '3d')
# sc = ax.scatter(x, y, t, c=T_pred, cmap='vididis')

# fig.colorbar(sc, ax=ax, label='T(x, y, t)')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Scatter Plot with Seaborn')

xnumpy = x.numpy()
plt.plot(xnumpy, T_pred[:, 0], 'o', markersize=1)
plt.plot(xnumpy, T_pred[:, 20], 'o', markersize=1)
plt.plot(xnumpy, T_pred[:, 40], 'o', markersize=1)
plt.figure(figsize=(5, 3), dpi=300)
# sns.heatmap(T_pred, cmap='jet')
plt.show()
