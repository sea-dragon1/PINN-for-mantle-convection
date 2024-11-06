import math
import torch
import numpy as np
import os
import time
import pandas as pd
from pinn_model import PINN_ns
from dataloader1 import Inside_Grid,NS_Boundary_Dataloader

def build_optimizer(network, optimizer_name, learning_rate):

    # 默认采用Adam优化器
    optimizer = torch.optim.Adam(network.parameters())
    if optimizer_name == 'lbfgs':
        # 设置lbfgs优化器
        optimizer = torch.optim.LBFGS(
            network.parameters(),
            lr=learning_rate,
            # max_iter=max_iter,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe',
        )
    return optimizer

def train_up(pinn_ns, epochs, batch_bdry, batch_inside, device):
    optimizer_name = 'adam'
    learning_rate = 0.1

    optimizer0 = build_optimizer(pinn_ns.up_NN_model, optimizer_name, learning_rate)

    start_time = time.time()
    epoch_lst = []
    time_lst = []
    loss_navier_stokes_lst = []
    loss_navier_stokes_pde_lst = []
    loss_navier_stokes_bdry_lst = []

    Inside_Grid_loader = Inside_Grid(batch_size=batch_inside, shuffle=False)
    Inside_batch_loader = Inside_Grid_loader.get_batches()

    Bdry_loader = NS_Boundary_Dataloader(batch_size=batch_bdry, shuffle=False)
    Bdry_batch_loader = Bdry_loader.get_batches()

    for epoch in range(epochs):
        loss_navier_stokes_pde_total = 0
        loss_navier_stokes_bdry_total = 0

        for X_inside in Inside_batch_loader:
            X_inside = X_inside.to(device)
            X_inside.requires_grad = True
            x, y= X_inside[:, 0], X_inside[:, 1]
            # navier_stocks 方程的pde损失
            optimizer0.zero_grad()
            loss_navier_stokes_pde = pinn_ns.navier_stokes_loss(x,y)
            loss_navier_stokes_pde.backward()
            optimizer0.step()
            loss_navier_stokes_pde_total += loss_navier_stokes_pde.item()
        loss_navier_stokes_pde_lst.append(loss_navier_stokes_pde_total)

        for X_bdry,u_bdry,v_bdry,p_bdry,rho_bdry,e_bdry in Bdry_batch_loader:
            optimizer0.zero_grad()
            loss_navier_stokes_bdry = pinn_ns.boundary_loss(X_bdry,u_bdry,v_bdry,p_bdry,rho_bdry,e_bdry)
            loss_navier_stokes_bdry.backward()
            optimizer0.step()
            loss_navier_stokes_bdry_total += loss_navier_stokes_bdry.item()
        loss_navier_stokes_bdry_lst.append(loss_navier_stokes_bdry_total)
        loss_naiver_stocks = loss_navier_stokes_pde_total + loss_navier_stokes_bdry_total
        loss_navier_stokes_lst.append(loss_naiver_stocks)
        epoch_lst.append(epoch)
        end_time = time.time()
        time_lst.append(end_time - start_time)
        # 每计算100次loss在控制台上输出
        if epoch % 1 == 0:
            print(f'epoch numbers: {epoch}, total_time: {end_time-start_time}, loss_naiver_stocks: {loss_naiver_stocks},',
            f'ns_pde_loss:{loss_navier_stokes_pde_total}, ns_bdr_loss:{loss_navier_stokes_bdry_total}, ')
    # 保存loss信息
    dp_log = pd.DataFrame({'epoch': epoch_lst, 'time': time_lst,
                           'loss_naiver_stocks':loss_navier_stokes_lst,
                           'ns_pde_loss': loss_navier_stokes_pde_lst,
                           'ns_bdr_loss':loss_navier_stokes_bdry_lst})
    return dp_log

def update_up(step, t_now):

    # 模型训练参数
    epochs = 1000
    device = 'cuda'
    batch_bdry = 10000  # bdry每个batch的大小
    batch_inside = 5000  # inside每个batch的大小

    #　设置权重文件保存位置
    parent_folder = r'results\up'
    folder = f'{t_now}'
    path = os.path.join(parent_folder, folder)
    try:
        os.makedirs(path)
        print(f"have made the foler '{folder}' in '{parent_folder}'")
    except FileExistsError:
        print(f"The folder '{folder}' has been in '{parent_folder}'")

    #　初始化模型实例,进行训练
    pinn_ns = PINN_ns(epochs, batch_bdry, batch_inside, device)
    pinn_ns.up_NN_model.train()
    dp_log = train_up(pinn_ns, epochs, batch_bdry, batch_inside, device)

    # 保存网络
    file_name = f'loss_ns_step{step}.csv'
    dp_log.to_csv(os.path.join(path, file_name))
    # 将模型保存到文件
    torch.save(pinn_ns.up_NN_model, f'{path}\\us_NN_model_step{step}.pth')
    pinn_ns.up_NN_model.eval()

    return pinn_ns.up_NN_model

if __name__ == '__main__':
    u_model = update_up(step=1, t_now=0)