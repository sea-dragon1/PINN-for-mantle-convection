import math
import torch
import numpy as np
import os
import time
import pandas as pd
from pinn_model import PINN
from dataloader0 import Inside_Grid,Boundary_Dataloader

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
def save_log( path, file_name, dp_log):

    dp_log.to_csv(os.path.join(path, file_name))

def train(epochs,device):
    optimizer_name = 'adam'
    learning_rate = 0.1

    pinn = PINN(epochs, batch_bdry, batch_inside, device)
    pinn.T_NN_model.train()
    pinn.up_NN_model.train()
    optimizer0 = build_optimizer(pinn.T_NN_model, optimizer_name, learning_rate)
    optimizer1 = build_optimizer(pinn.up_NN_model, optimizer_name, learning_rate)

    start_time = time.time()
    epoch_lst = []
    time_lst = []
    loss_lst = []
    loss_heat_transfer_lst = []
    loss_navier_stokes_lst = []
    loss_heat_transfer_pde_lst = []
    loss_navier_stokes_pde_lst = []
    loss_heat_transfer_bdry_lst = []
    loss_navier_stokes_bdry_lst = []

    Inside_Grid_loader = Inside_Grid(batch_size=batch_inside, shuffle=False)
    Indide_batch_loader = Inside_Grid_loader.get_batches()

    Bdry_loader = Boundary_Dataloader(batch_size=batch_bdry, shuffle=False)
    Bdry_batch_loader = Bdry_loader.get_batches()

    for epoch in range(epochs):
        loss_navier_stokes_pde_total = 0
        loss_heat_transfer_pde_total = 0
        loss_heat_transfer_bdry_total = 0
        loss_navier_stokes_bdry_total = 0

        for X_inside,u_inside in Indide_batch_loader:
            #　
            X_inside = X_inside.to(device)
            X_inside.requires_grad = True
            x, y, t = X_inside[:, 0], X_inside[:, 1], X_inside[:, 2]
            # heat_transfer 方程的pde损失
            optimizer0.zero_grad()
            loss_heat_transfer_pde = pinn.heat_transfer_loss(X_inside, u_inside)
            loss_heat_transfer_pde.backward()
            optimizer0.step()
            loss_heat_transfer_pde_total += loss_heat_transfer_pde.item()
            # navier_stocks 方程的pde损失
            optimizer1.zero_grad()
            loss_navier_stokes_pde = pinn.navier_stokes_loss(x,y,t)
            loss_navier_stokes_pde.backward()
            optimizer1.step()
            loss_navier_stokes_pde_total += loss_navier_stokes_pde.item()

        loss_heat_transfer_pde_lst.append(loss_heat_transfer_pde_total)
        loss_navier_stokes_pde_lst.append(loss_navier_stokes_pde_total)

        for X_boundary,T_boundary,u_boundary,v_boundary,p_boundary in Bdry_batch_loader:
            optimizer0.zero_grad()
            loss_heat_transfer_bdry,loss_navier_stokes_bdry = pinn.boundary_loss(X_boundary, T_boundary,u_boundary,v_boundary,p_boundary)
            loss_bdr = loss_heat_transfer_bdry+loss_navier_stokes_bdry
            loss_bdr.backward()
            optimizer0.step()
            loss_heat_transfer_bdry_total += loss_heat_transfer_bdry.item()
            loss_navier_stokes_bdry_total += loss_navier_stokes_bdry.item()
        loss_heat_transfer_bdry_lst.append(loss_heat_transfer_bdry_total)
        loss_navier_stokes_bdry_lst.append(loss_navier_stokes_bdry_total)

        loss_heat_transfer = loss_heat_transfer_pde_total + loss_heat_transfer_bdry_total
        loss_naiver_stocks = loss_navier_stokes_pde_total + loss_navier_stokes_bdry_total
        loss_total = loss_heat_transfer + loss_naiver_stocks

        # 每计算100次loss在控制台上输出
        if epoch % 1 == 0:
            end_time = time.time()
            print(f'epoch numbers: {epoch} total_loss: {loss_total}, total_time: {end_time-start_time},',
            f'ns_pde_loss:{loss_navier_stokes_pde_total}, ns_bdr_loss:{loss_navier_stokes_bdry_total}, ',
            f'ts_pde_loss:{loss_heat_transfer_pde_total}, ts_bdr_loss:{loss_heat_transfer_bdry_total}, '  )
            loss_lst.append(loss_total)
            epoch_lst.append(epoch)
            time_lst.append(end_time-start_time)


    # 保存loss信息
    dp_log = pd.DataFrame({'epoch': epoch_lst, 'time': time_lst, 'loss': loss_lst,
                           'ns_pde_loss': loss_navier_stokes_pde_lst, 'ns_bdr_loss':loss_navier_stokes_bdry_lst,
                           'ts_pde_loss': loss_heat_transfer_pde_lst, 'ts_bdr_loss':loss_heat_transfer_bdry_lst})
    return dp_log




if torch.cuda.is_available():
    device = torch.device('cuda')
    print('--------GPU运行中---------')
else:
    device = torch.device('cpu')
    print('--------CPU运行中---------')

# 实例化PINN
epochs = 10
batch_bdry = 20000 # bdry每个batch的大小
batch_inside = 10000 # inside每个batch的大小
h = 0.1 # 设置空间步长
k = 0.1 # 设置时间步长

dp_log = train(epochs,device)

parent_folder = 'results'
folder = '10grid'
path = os.path.join(parent_folder, folder)

try:
    os.makedirs(path)
    print(f"have made the foler '{folder}' in '{parent_folder}'")
except FileExistsError:
    print(f"The folder '{folder}' has been in '{parent_folder}'")


file_name = 'loss.csv'
save_log(path,file_name,dp_log)
# 将模型保存到文件
# torch.save(pinn.T_NN_model, path+'\T_NN_model_pro.pth')