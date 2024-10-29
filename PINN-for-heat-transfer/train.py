import math
import torch
import numpy as np
import os
import time
import pandas as pd
from network import Network
from dataloader import Inside_Grid,Boundary_Dataloader

# Heat tansfer and energy Conservation equation
# \partial T / \partial t + u\cdot \nabla T - \nabla^2 T = \rho H(x,y)
# rectangle
# Boundary condition: T(0,y,t) = 0, T(x,0,t) = 0,
# Initial condition:　T(x,y,0) = T_0(x,y,0)

class PINN:
    def __init__(self, epochs, batch_bdry, batch_inside):

        self.epochs = epochs
        self.batch_bdry = batch_bdry
        self.batch_inside = batch_inside

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
            act = torch.nn.ReLU
        ).to(self.device)


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



    def boundary_loss(self, X_boundary, T_boundary):
        # 第一部分loss: 边界条件

        self.X_boundary = X_boundary.to(self.device)
        self.T_boundary = T_boundary.to(self.device)
        T_pred_boundary = self.model(self.X_boundary)
        loss_boundary = self.MSE(T_pred_boundary, self.T_boundary)
        print(f"1 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")

        return loss_boundary

    # 损失函数
    def pde_loss(self, X_inside, u_inside):
        # loss_eqution MSE中其实就是 已知的偏微分方程
        # 第二部分loss：内点吻合方程（物理性） 由于内部点数量大，容易显存不够用，我们分批进行计算损失
        X_inside = X_inside.to(self.device)
        u_inside = u_inside.to(self.device)
        print(f"2 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")

        X_inside.requires_grad = True  # 设置：需要计算对X的梯度
        # 导数归零

        print(f"30 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        T_inside = self.model(X_inside)
        # 计算模型参数的总数量
        param_sum = sum(p.numel() for p in self.model.parameters())
        print('模型参数大小：', param_sum)
        print(f"31 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        # 使用自动微分方法得到T对X的导数
        dT_dxyt = torch.autograd.grad(
            inputs=X_inside,
            outputs=T_inside,
            grad_outputs=torch.ones_like(T_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        # dT_dx = dT_dxyt[:, 0]
        # dT_dy = dT_dxyt[:, 1]
        dT_dt = dT_dxyt[:, 2]
        print(f"4 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")
        # 使用自动微分求U对X的二阶导数
        dT_dXX = torch.autograd.grad(
            inputs=X_inside,
            outputs=dT_dxyt,
            grad_outputs=torch.ones_like(dT_dxyt),
            retain_graph=True,
            create_graph=True
        )[0]
        dT_dxx = dT_dXX[:, 0]
        dT_dyy = dT_dXX[:, 1]
        print(f"5 Memory before backward: {torch.cuda.memory_allocated(self.device)} bytes")

        pde_func = dT_dt + torch.matmul(u_inside, dT_dxyt[:, :2].T)
        loss_eqution = self.MSE(pde_func, dT_dxx + dT_dyy)

        return loss_eqution

    # 训练
    def train(self):

        self.model.train()
        self.start_time = time.time()
        loss_lst = []
        loss_pde_lst = []
        loss_bdry_lst = []
        epoch_lst = []
        time_lst = []

        for epoch in range(self.epochs):

            loss_eqution_total = 0
            loss_bdry_total = 0
            # loss反向传播， 用于给优化器提供梯度信息
            Inside_Grid_loader = Inside_Grid(batch_size=self.batch_inside, shuffle=False)
            Indide_batch_loader = Inside_Grid_loader.get_batches()
            for X_inside,u_inside in Indide_batch_loader:
                X_inside = X_inside.to(self.device)
                X_inside.requires_grad = True
                self.adam.zero_grad()
                loss_eqution = self.pde_loss(X_inside, u_inside)
                loss_eqution.backward()
                self.adam.step()
                loss_eqution_total += loss_eqution.item()
            loss_pde_lst.append(loss_eqution_total)


            Bdry_loader = Boundary_Dataloader(batch_size=self.batch_bdry, shuffle=False)
            Bdry_batch_loader = Bdry_loader.get_batches()
            for X_boundary,T_boundary in Bdry_batch_loader:
                self.adam.zero_grad()
                loss_bdry = self.boundary_loss(X_boundary, T_boundary)
                loss_bdry.backward()
                self.adam.step()
                loss_bdry_total += loss_bdry.item()
            loss_bdry_lst.append(loss_bdry_total)

            loss = loss_eqution_total + loss_bdry_total
            # 每计算100次loss在控制台上输出
            if epoch % 1 == 0:
                self.end_time = time.time()
                print(f'epoch numbers: {epoch} total_loss: {loss}, total_time: {self.end_time-self.start_time}, \
                pde_loss:{loss_eqution_total}, bdr_loss:{loss_bdry_total}')
                loss_lst.append(loss)
                epoch_lst.append(epoch)
                time_lst.append(self.end_time-self.start_time)


        # 假设每个列表中都有相同数量的数据行
        self.dp_log = pd.DataFrame({'epoch': epoch_lst, 'time': time_lst, 'loss': loss_lst, 'pde_loss': loss_pde_lst, 'bdry_loss':loss_bdry_lst})

    def save_log(self, path, file_name):

        self.dp_log.to_csv(os.path.join(path, file_name))


# 实例化PINN
epoch = 10
batch_bdry = 20000 # bdry每个batch的大小
batch_inside = 11000 # inside每个batch的大小
h = 0.01 # 设置空间步长
k = 0.01 # 设置时间步长
pinn = PINN(epoch, batch_bdry, batch_inside)

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

file_name = 'loss.csv'
pinn.save_log(path,file_name=file_name)
# 将模型保存到文件
torch.save(pinn.model, path+'\model_pro.pth')