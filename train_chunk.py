import math
import torch
import numpy as np
import os
import time
import pandas as pd
from network import Network


# Heat tansfer and energy Conservation equation
# \partial T / \partial t + u\cdot \nabla T - \nabla^2 T = \rho H(x,y)
# rectangle
# Boundary condition: T(0,y,t) = 0, T(x,0,t) = 0,
# Initial condition:　T(x,y,0) = T_0(x,y,0)

class PINN:
    def __init__(self, epoch):

        self.epoch = epoch

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('--------GPU运行中---------')
        else:
            self.device = torch.device('cpu')
            print('--------CPU运行中---------')

        # 定义神经网络
        self.model = Network(
            input_size = 3,
            hidden_size = 16,
            output_size = 1,
            depth = 8,
            act = torch.nn.Tanh
        ).to(self.device)

        self.h = 0.02 # 设置空间步长
        self.k = 0.02 # 设置时间步长
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
        ic = torch.stack(torch.meshgrid(x, y, t[0])).reshape(3, -1).T
        self.X_boundary = torch.cat([bcx1, bcx2, bcy1, bcy2, ic])

        # 边界处的T值
        # 初边值条件 T(-1,y,t)=0, T(1,y,t)=0, T(x,y,1)=-sin(pi*x)-sin(pi*y)
        # !!!需要针对具体方程重写
        T_bcx1 = torch.zeros(len(bcx1)) # x=-1处的第一类边界条件 T=0
        T_bcx2 = torch.zeros(len(bcx2)) # x=1 处的第二类边界条件 T=0
        T_bcy1 = torch.zeros(len(bcy1))  # y=-1处的第一类边界条件 T=0
        T_bcy2 = torch.zeros(len(bcy2))  # y=1 处的第二类边界条件 T=0
        T_ic = -torch.sin(math.pi * ic[:, 0]) -torch.sin(math.pi * ic[:, 1])# t=0的初值条件 T=-sin(pi*x)-sin(pi*y)
        self.T_boundary = torch.cat([T_bcx1 ,T_bcx2, T_bcy1, T_bcy2, T_ic])# 将所有边界处的T值整合为一个张量
        self.T_boundary = self.T_boundary.unsqueeze(1)

        # 速度场u
        self.u_inside = torch.ones(self.X_inside.size()[0], 2)

        # 设置MSE
        self.MSE = torch.nn.MSELoss()

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

    def pde_loss(self, u_inside=None, dT_dt=None, dT_dxyt=None, dT_dxx=None, dT_dyy=None):
        # loss_eqution MSE中其实就是 已知的偏微分方程
        pde_func = dT_dt + torch.matmul(u_inside, dT_dxyt[:, :2].T)
        loss_eqution = self.MSE(pde_func, dT_dxx + dT_dyy)
        return loss_eqution

    # 损失函数
    # ！！！ 针对具体方程，损失函数需要重写
    def loss_fun(self, X_boundary, T_boundary):

        # 第一部分loss: 边界条件
        self.X_boundary = self.X_boundary
        self.T_boundary = self.T_boundary
        # 确定分块大小
        num_chunks1 = 100
        chunk_size = self.X_inside.size(0) // num_chunks1
        print(chunk_size)
        print(f"1 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        # 将张量分块
        X_boundary_chunks = torch.chunk(self.X_boundary, num_chunks1, dim=0)
        T_boundary_chunks = torch.chunk(self.T_boundary, num_chunks1, dim=0)
        loss_boundary_chunks = []
        for i in range(num_chunks1-1):
            X_boundary_chunk = X_boundary_chunks[i].to(self.device)
            T_pred_boundary_chunk = self.model(X_boundary_chunk)
            loss_boundary_chunks.append(self.MSE(T_boundary_chunks[i].to(self.device), T_pred_boundary_chunk))
        loss_boundary = sum(loss_boundary_chunks)
        print(f"2 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        # # 第二部分loss：内点吻合方程（物理性） 由于内部点数量大，容易显存不够用，我们分批进行计算损失
        # 第二部分loss：内点吻合方程（物理性）
        # 导数归零
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        # 确定分块大小
        num_chunks = 50
        chunk_size = self.X_inside.size(0) // num_chunks
        print(f"3 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        # 将张量分块
        X_inside_chunks = torch.chunk(self.X_inside, num_chunks, dim=0)
        u_inside_chunks = torch.chunk(self.u_inside, num_chunks, dim=0)

        # 初始化分块损失
        loss_equation_chunks = []
        print(f"4 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        for i in range(num_chunks-1):
            # 获取当前块
            X_inside_chunk = X_inside_chunks[i].to(self.device)
            u_inside_chunk = u_inside_chunks[i].to(self.device)
            X_inside_chunk.requires_grad = True  # 设置：需要计算对X的梯度

            # 计算当前块的T_inside
            T_inside_chunk = self.model(X_inside_chunk)

            # 使用自动微分方法得到T对X的导数
            dT_dxyt_chunk = torch.autograd.grad(
                inputs=X_inside_chunk,
                outputs=T_inside_chunk,
                grad_outputs=torch.ones_like(T_inside_chunk),
                retain_graph=True,
                create_graph=True
            )[0]
            dT_dt_chunk = dT_dxyt_chunk[:, 2]

            # 使用自动微分求U对X的二阶导数
            dT_dXX_chunk = torch.autograd.grad(
                inputs=X_inside_chunk,
                outputs=dT_dxyt_chunk,
                grad_outputs=torch.ones_like(dT_dxyt_chunk),
                retain_graph=True,
                create_graph=True
            )[0]
            dT_dxx_chunk = dT_dXX_chunk[:, 0]
            dT_dyy_chunk = dT_dXX_chunk[:, 1]
            print(f"5 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
            # 计算当前块的PDE损失
            loss_eq_chunk = self.pde_loss(u_inside_chunk, dT_dt_chunk, dT_dxyt_chunk, dT_dxx_chunk, dT_dyy_chunk)
            loss_equation_chunks.append(loss_eq_chunk)

        # 将所有块的损失相加
        loss_equation = sum(loss_equation_chunks)
        # 最终loss
        loss = loss_equation + loss_boundary
        loss.backward()

        return loss

    # 训练
    def train(self):

        self.model.train()
        self.start_time = time.time()
        loss_lst = []
        epoch_lst = []
        time_lst = []

        for i in range(self.epoch):
            loss = self.loss_fun()
            # loss反向传播， 用于给优化器提供梯度信息
            # 每计算100次loss在控制台上输出
            if i % 1 == 0:
                self.end_time = time.time()
                print('iter numbers:', i, ' loss:', loss.item(), 'total_time:', self.end_time-self.start_time)
                loss_cpu = loss.item() if loss.is_cuda else loss.item()
                loss_lst.append(loss_cpu)
                epoch_lst.append(i)
                time_lst.append(self.end_time-self.start_time)

            if i < 5000:
                # 然后运行adam优化器
                self.adam.step()
            else:
                # 然后运行lbfgs优化器
                self.lbfgs.step()

        # 假设每个列表中都有相同数量的数据行
        dp = pd.DataFrame({'loss': loss_lst, 'epoch': epoch_lst, 'time': time_lst})
        # dp = pd.DataFrame([loss_lst, epoch_lst, time_lst], columns=['loss', 'epoch', 'time'])


# 实例化PINN
epoch = 10

pinn = PINN(epoch)

parent_folder = 'results'
folder = '20grid'
path = os.path.join(parent_folder, folder)

try:
    os.makedirs(path)
    print(f"have made the foler '{folder}' in '{parent_folder}'")
except FileExistsError:
    print(f"The folder '{folder}' has been in '{parent_folder}'")

# 开始训练
pinn.train()

# 将模型保存到文件
torch.save(pinn.model, path+'\model_pro.pth')