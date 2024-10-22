import math
import torch
import numpy as np
from network import Network

# Heat tansfer and energy Conservation equation
# \partial T / \partial t + u\cdot \nabla T - \nabla^2 T = \rho H(x,y)
# rectangle
# Boundary condition: T(0,y,t) = 0, T(x,0,t) = 0,
# Initial condition:　T(x,y,1) = T_pred9

class PINN:
    def __init__(self, epoch):

        self.epoch = epoch


        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('--------GPU运行中---------')
        else:
            device = torch.device('cpu')
            print('--------CPU运行中---------')

        # 定义神经网络
        self.model = Network(
            input_size = 3,
            hidden_size = 16,
            output_size = 1,
            depth = 8,
            act = torch.nn.Tanh
        ).to(device)

        self.h = 0.1 # 设置空间步长
        self.k = 0.1 # 设置时间步长
        x = torch.arange(-1, 1 + self.h, self.h) # 在[-1,1]区间上均匀取值，记为x
        y = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为y
        t = torch.arange(0, 1 + self.k, self.k) # 在[0,1]区间均匀取值，记为t

        # 将t和x组合， 形成时间空间网格，记录在张量X_inside中
        self.X_inside = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T

        # 边界处的时空坐标
        bcx1 = torch.stack(torch.meshgrid(x[0], y, t)).reshape(3, -1).T
        bcx2 = torch.stack(torch.meshgrid(x[-1], y, t)).reshape(3, -1).T
        bcy1 = torch.stack(torch.meshgrid(x, y[0], t)).reshape(3, -1).T
        bcy2 = torch.stack(torch.meshgrid(x, y[-1], t)).reshape(3, -1).T
        ic = torch.stack(torch.meshgrid(x, y, t[10])).reshape(3, -1).T
        self.X_boundary = torch.cat([bcx1, bcx2, bcy1, bcy2, ic])


        # 边界处的T值
        # 初边值条件 T(-1,y,t)=0, T(1,y,t)=0, T(x,y,1)=T_pred9
        # !!!需要针对具体方程重写
        T_bcx1 = torch.zeros(len(bcx1)) # x=-1处的第一类边界条件 T=0
        T_bcx2 = torch.zeros(len(bcx2)) # x=1 处的第二类边界条件 T=0
        T_bcy1 = torch.zeros(len(bcy1))  # y=-1处的第一类边界条件 T=0
        T_bcy2 = torch.zeros(len(bcy2))  # y=1 处的第二类边界条件 T=0
        T_ic = torch.from_numpy(np.loadtxt('T_pred11.txt').reshape(-1,1)).float().squeeze()
        self.T_boundary = torch.cat([T_bcx1 ,T_bcx2, T_bcy1, T_bcy2, T_ic])# 将所有边界处的T值整合为一个张量
        self.T_boundary = self.T_boundary.unsqueeze(1)

        # 速度场u
        self.u_inside = torch.ones(self.X_inside.size()[0], 2)

        # 将数据放入GPU
        self.X_inside = self.X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)
        self.T_boundary = self.T_boundary.to(device)
        self.u_inside = self.u_inside.to(device)

        self.X_inside.requires_grad = True # 设置：需要计算对X的梯度

        # 设置MSE
        self.MSE = torch.nn.MSELoss()

        # 定义迭代序号，记录调用loss次数
        self.iter = 1

        # 设置优化器
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr = 1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe',
        )

        # 设置Adam优化器
        self.adam = torch.optim.Adam(self.model.parameters())

    # 损失函数
    # ！！！ 针对具体方程，损失函数需要重写
    def loss_fun(self):
        # 导数归零
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        # 第一部分loss: 边界条件
        T_pred_boundary = self.model(self.X_boundary)
        loss_boundary = self.MSE(T_pred_boundary, self.T_boundary)

        # 第二部分loss：内点吻合方程（物理性）
        T_inside = self.model(self.X_inside)

        # 使用自动微分方法得到T对X的导数
        dT_dxyt = torch.autograd.grad(
            inputs = self.X_inside,
            outputs = T_inside,
            grad_outputs = torch.ones_like(T_inside),
            retain_graph = True,
            create_graph = True
        )[0]
        dT_dx = dT_dxyt[:, 0]
        dT_dy = dT_dxyt[:, 1]
        dT_dt = dT_dxyt[:, 2]

        # 使用自动微分求U对X的二阶导数
        dT_dXX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=dT_dxyt,
            grad_outputs=torch.ones_like(dT_dxyt),
            retain_graph=True,
            create_graph=True
        )[0]
        dT_dxx = dT_dXX[:, 0]
        dT_dyy = dT_dXX[:, 0]

        # loss_eqution MSE中其实就是 已知的偏微分方程，此处为Burgers方程
        pde_func = dT_dt + torch.matmul(self.u_inside, dT_dxyt[:,:2].T)
        loss_eqution = self.MSE(pde_func,  dT_dxx + dT_dyy)


        # 最终loss
        loss = loss_eqution + loss_boundary

        # loss反向传播， 用于给优化器提供梯度信息
        loss.backward()

        # 每计算100次loss在控制台上输出
        if self.iter % 1 == 0:
            print(self.iter, loss.item())
        self.iter += 1
        return loss

    # 训练
    def train(self):
        self.model.train()

        # 首先运行5000步Adam优化器
        print("采用Adam优化器")
        for i in range(self.epoch):
            self.adam.step(self.loss_fun)
        # 然后运行lbfgs优化器
        print("采用lbfgs优化器")
        self.lbfgs.step(self.loss_fun)

# 实例化PINN
iter = 2

pinn = PINN(iter)

# 开始训练
pinn.train()

# 将模型保存到文件
torch.save(pinn.model, 'model_retro11.pth')