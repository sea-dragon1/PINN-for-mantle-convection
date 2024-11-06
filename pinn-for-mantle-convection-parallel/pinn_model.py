import math
import torch
import numpy as np
import os
import time
import pandas as pd
from network import FC_Network
from dataloader0 import Inside_Grid,Boundary_Dataloader

# Heat tansfer and energy Conservation equation
# \partial T / \partial t + u\cdot \nabla T - \nabla^2 T = \rho H(x,y)
# rectangle
# Boundary condition: T(0,y,t) = 0, T(x,0,t) = 0,
# Initial condition:　T(x,y,0) = T_0(x,y,0)

class PINN:
    '''
    初始化网络实例
    确定方程与边界损失
    '''
    def __init__(self, epochs, batch_bdry, batch_inside, devive):
        self.device = devive
        self.epochs = epochs
        self.batch_bdry = batch_bdry
        self.batch_inside = batch_inside

        # 定义神经网络
        self.T_NN_model = FC_Network(
            input_size = 3,
            hidden_size = 16,
            output_size = 1,
            depth = 8,
            act = torch.nn.ReLU
        ).to(self.device)

        self.up_NN_model = FC_Network(
            input_size = 3,
            hidden_size = 16,
            output_size = 3,
            depth = 8,
            act = torch.nn.ReLU
        ).to(self.device)

        # 设置MSE
        self.MSE = torch.nn.MSELoss()


    def boundary_loss(self, X_boundary, T_boundary, u_boundary, v_boundary, p_boundary):
        # 第一部分loss: 边界条件

        self.X_boundary = X_boundary.to(self.device)
        self.T_boundary = T_boundary.to(self.device)
        self.u_boundary = u_boundary.to(self.device)
        self.v_boundary = v_boundary.to(self.device)
        self.p_boundary = p_boundary.to(self.device)
        T_pred_boundary = self.T_NN_model(self.X_boundary)
        u_pred_boundary,v_pred_boundary,p_pred_boundary = self.up_NN_model(self.X_boundary).chunk(3, dim=1) #
        loss_ht_boundary = self.MSE(T_pred_boundary, self.T_boundary)+ self.MSE(u_pred_boundary, self.u_boundary)
        loss_ns_boundary =  self.MSE(v_pred_boundary, self.v_boundary)+ self.MSE(p_pred_boundary, self.p_boundary)

        return loss_ht_boundary,loss_ns_boundary

    def heat_transfer_loss(self, X_inside, u_inside):

        # 第二部分loss：内点吻合方程（物理性）
        X_inside = X_inside.to(self.device)
        u_inside = u_inside.to(self.device)
        X_inside.requires_grad = True  # 设置：需要计算对X的梯度

        T_inside = self.T_NN_model(X_inside)

        # 使用自动微分方法得到T对X的导数
        T_xyt = torch.autograd.grad(
            outputs=T_inside,
            inputs=X_inside,
            grad_outputs=torch.ones_like(T_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        # dT_dx = T_xyt[:, 0]
        # dT_dy = T_xyt[:, 1]
        T_t = T_xyt[:, 2]

        # 使用自动微分求U对X的二阶导数
        T_XX = torch.autograd.grad(
            outputs=T_xyt,
            inputs=X_inside,
            grad_outputs=torch.ones_like(T_xyt),
            retain_graph=True,
            create_graph=True
        )[0]
        T_xx = T_XX[:, 0]
        T_yy = T_XX[:, 1]

        pde_func = T_t + torch.matmul(u_inside, T_xyt[:, :2].T)
        loss_eqution = self.MSE(pde_func, T_xx + T_yy)

        return loss_eqution

    def navier_stokes_loss(self, x, y, t):

        g = 9.8
        u, v, p = self.up_NN_model(torch.cat((x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0)), 0).T).chunk(3, dim=1)  # 分割输出为u和v
        rho = torch.ones_like(u)

        # 计算梯度
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        # v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # 计算拉普拉斯算子
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # Navier-Stokes方程的非线性项
        u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yx = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yx = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # 构建Navier-Stokes方程的残差
        res_u = - u_xx - u_yy - u_xx - v_yx + p_x - rho*g
        res_v = - v_xx - v_yy - v_yy - u_yx + p_y - rho*g
        # 质量守恒pde损失
        res_mass_u = u_x + u_y
        res_mass_v = v_x + v_y

        loss_ns = self.MSE(res_u, torch.zeros_like(res_u)) + self.MSE(res_v,torch.zeros_like(res_v)) \
                    + self.MSE(res_mass_u, torch.zeros_like(res_mass_u)) \
                    + self.MSE(res_mass_v, torch.zeros_like(res_mass_v))

        return loss_ns
