import torch
import numpy as np
import os
import time
import pandas as pd
from pinn_model import PINN_T
from dataloader1 import T_Boundary_Dataloader, Init_Dataloader, Inside_Grid


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


def train_T(pinn, epochs, batch_bdry, batch_inside, up_model, T_now_nn, t_now, t_next, device, is_init):
    optimizer_name = 'adam'
    learning_rate = 0.1
    optimizer0 = build_optimizer(pinn.T_NN_model, optimizer_name, learning_rate)
    start_time = time.time()
    epoch_lst = []
    time_lst = []
    loss_heat_transfer_lst = []
    loss_heat_transfer_pde_lst = []
    loss_heat_transfer_init_lst = []
    loss_heat_transfer_bdry_lst = []

    Bdry_loader = T_Boundary_Dataloader(batch_size=batch_bdry, shuffle=False)
    Bdry_batch_loader = Bdry_loader.get_batches()
    Init_loader = Init_Dataloader(batch_size=batch_inside, shuffle=False)
    Init_batch_loader = Init_loader.get_batches()
    Inside_Grid_loader = Inside_Grid(batch_size=batch_inside, shuffle=False)
    Inside_Grid_batch_loader = Inside_Grid_loader.get_batches()

    if is_init:
        for epoch in range(epochs):

            loss_heat_transfer_pde_total = 0
            loss_heat_transfer_init_total = 0
            loss_heat_transfer_bdry_total = 0

            for X, u, v, p, T, d, e in Init_batch_loader:
                t_now_tensor = torch.full_like((X[:, 0]), t_now, device=X.device).unsqueeze(0)
                t_next_tensor = torch.full_like((X[:, 0]), t_next, device=X.device).unsqueeze(0)
                Xt_now = torch.cat((X, t_now_tensor.T), dim=1).to(device)
                Xt_next = torch.cat((X, t_next_tensor.T), dim=1).to(device)
                u_inside = torch.cat((u, v), dim=1).to(device)
                Xt_now.requires_grad = True
                Xt_next.requires_grad = True
                # heat_transfer 方程的pde损失 计算当前时间点与下一时间点
                optimizer0.zero_grad()
                loss_heat_transfer_pde = pinn.heat_transfer_loss(Xt_now, u_inside) + pinn.heat_transfer_loss(Xt_next,
                                                                                                             u_inside)
                loss_heat_transfer_pde.backward()
                optimizer0.step()
                loss_heat_transfer_pde_total += loss_heat_transfer_pde.item()
                # 初值损失
                optimizer0.zero_grad()
                loss_heat_transfer_init = pinn.boundary_loss(Xt_now, T, u, v, p)
                loss_heat_transfer_init.backward()
                optimizer0.step()
                loss_heat_transfer_init_total += loss_heat_transfer_init.item()

            loss_heat_transfer_pde_lst.append(loss_heat_transfer_pde_total)
            loss_heat_transfer_init_lst.append(loss_heat_transfer_init_total)

            for X_bdry, u_bdry, v_bdry, p_bdry, T_bdry, d_bdry, e_bdry in Bdry_batch_loader:
                t_now_tensor = torch.full_like((X_bdry[:, 0]), t_now, device=X_bdry.device).unsqueeze(0)
                t_next_tensor = torch.full_like((X_bdry[:, 0]), t_next, device=X_bdry.device).unsqueeze(0)
                Xt_bdry_now = torch.cat((X_bdry, t_now_tensor.T), dim=1).to(device)
                Xt_bdry_next = torch.cat((X_bdry, t_next_tensor.T), dim=1).to(device)
                Xt_bdry_now.requires_grad = True
                Xt_bdry_next.requires_grad = True
                optimizer0.zero_grad()
                loss_heat_transfer_bdry = pinn.boundary_loss(Xt_bdry_now, T_bdry, u_bdry, v_bdry,
                                                             p_bdry) + pinn.boundary_loss(Xt_bdry_next, T_bdry, u_bdry,
                                                                                          v_bdry, p_bdry)
                loss_heat_transfer_bdry.backward()
                optimizer0.step()
                loss_heat_transfer_bdry_total += loss_heat_transfer_bdry.item()
            loss_heat_transfer_bdry_lst.append(loss_heat_transfer_bdry_total)

            loss_heat_transfer = loss_heat_transfer_pde_total + loss_heat_transfer_init_total + loss_heat_transfer_bdry_total
            end_time = time.time()
            loss_heat_transfer_lst.append(loss_heat_transfer)
            epoch_lst.append(epoch)
            time_lst.append(end_time - start_time)

            # 每计算100次loss在控制台上输出
            if epoch % 1 == 0:
                print(
                    f'epoch numbers: {epoch}, total_time: {end_time - start_time}, loss_heat_transfer: {loss_heat_transfer}',
                    f'ts_pde_loss:{loss_heat_transfer_pde_total},ts_init_loss:{loss_heat_transfer_init_total},'
                    f' ts_bdr_loss:{loss_heat_transfer_bdry_total}, ')

    else:
        for epoch in range(epochs):
            loss_heat_transfer_pde_total = 0
            loss_heat_transfer_init_total = 0
            loss_heat_transfer_bdry_total = 0

            for X_inside in Inside_Grid_batch_loader:
                X_inside = X_inside.to(device)
                u_inside = up_model(X_inside)
                t_now_tensor = torch.full_like((X_inside[:, 0]), t_now, device=X_inside.device).unsqueeze(0)
                t_next_tensor = torch.full_like((X_inside[:, 0]), t_next, device=X_inside.device).unsqueeze(0)
                Xt_now = torch.cat((X_inside, t_now_tensor.T), dim=1).to(device)
                Xt_next = torch.cat((X_inside, t_next_tensor.T), dim=1).to(device)
                Xt_now.requires_grad = True
                Xt_next.requires_grad = True

                # heat_transfer 方程的pde损失
                optimizer0.zero_grad()
                loss_heat_transfer_pde = pinn.heat_transfer_loss(Xt_next, u_inside[:,0:2]) + pinn.heat_transfer_loss(Xt_now, u_inside[:,0:2])
                loss_heat_transfer_pde.backward()
                optimizer0.step()
                loss_heat_transfer_pde_total += loss_heat_transfer_pde.item()
                # heat_transfer 方程的初值损失
                T_init = T_now_nn(Xt_now)
                optimizer0.zero_grad()
                loss_heat_transfer_init = pinn.boundary_loss(Xt_now, T_init, u_inside[:,0], u_inside[:,1], u_inside[:,2])
                optimizer0.step()
                loss_heat_transfer_init_total += loss_heat_transfer_init.item()

            loss_heat_transfer_pde_lst.append(loss_heat_transfer_pde_total)
            loss_heat_transfer_init_lst.append(loss_heat_transfer_init_total)

            for X_bdry, u_bdry, v_bdry, p_bdry, T_bdry, d_bdry, e_bdry in Bdry_batch_loader:
                t_now_tensor = torch.full_like((X_bdry[:, 0]), t_now, device=X_bdry.device).unsqueeze(0)
                t_next_tensor = torch.full_like((X_bdry[:, 0]), t_next, device=X_bdry.device).unsqueeze(0)
                Xt_bdry_now = torch.cat((X_bdry, t_now_tensor.T), dim=1).to(device)
                Xt_bdry_next = torch.cat((X_bdry, t_next_tensor.T), dim=1).to(device)
                Xt_bdry_now.requires_grad = True
                Xt_bdry_next.requires_grad = True
                optimizer0.zero_grad()
                loss_heat_transfer_bdry = (pinn.boundary_loss(Xt_bdry_now, T_bdry, u_bdry, v_bdry,p_bdry)
                                           + pinn.boundary_loss(Xt_bdry_next, T_bdry, u_bdry, v_bdry, p_bdry))
                loss_heat_transfer_bdry.backward()
                optimizer0.step()
                loss_heat_transfer_bdry_total += loss_heat_transfer_bdry.item()
            loss_heat_transfer_bdry_lst.append(loss_heat_transfer_bdry_total)
            loss_heat_transfer = loss_heat_transfer_pde_total + loss_heat_transfer_init_total + loss_heat_transfer_bdry_total
            end_time = time.time()
            epoch_lst.append(epoch)
            time_lst.append(end_time - start_time)

            # 每计算100次loss在控制台上输出
            if epoch % 1 == 0:
                print(
                    f'epoch numbers: {epoch}, total_time: {end_time - start_time}, loss_heat_transfer: {loss_heat_transfer}',
                    f'ts_pde_loss:{loss_heat_transfer_pde_total}, ts_init_loss:{loss_heat_transfer_init_total},'
                    f'ts_bdr_loss:{loss_heat_transfer_bdry_total}, ')
                loss_heat_transfer_lst.append(loss_heat_transfer)

    # 保存loss信息
    dp_log = pd.DataFrame({'epoch': epoch_lst, 'time': time_lst, 'loss_heat_transfer': loss_heat_transfer,
                           'ts_pde_loss': loss_heat_transfer_pde_lst, 'ts_init_loss':loss_heat_transfer_init_total,
                           'ts_bdr_loss': loss_heat_transfer_bdry_lst})
    return dp_log, pinn.T_NN_model


def update_T(T_now_nn, up_model, step, t_now, t_next, is_init):
    up_model = up_model
    # 实例化PINN
    epochs = 1000
    device = 'cuda'
    batch_bdry = 10000  # bdry每个batch的大小
    batch_inside = 10000  # inside每个batch的大小

    parent_folder = 'results\T'
    folder = f'{t_next}'
    path = os.path.join(parent_folder, folder)

    try:
        os.makedirs(path)
        print(f"have made the foler '{folder}' in '{parent_folder}'")
    except FileExistsError:
        print(f"The folder '{folder}' has been in '{parent_folder}'")

    pinn_T = PINN_T(epochs, batch_bdry, batch_inside, device)
    pinn_T.T_NN_model.train()

    dp_log, T_nn = train_T(pinn_T,
                           epochs,
                           batch_bdry,
                           batch_inside,
                           up_model,
                           T_now_nn,
                           t_now,
                           t_next,
                           device,
                           is_init=is_init)
    pinn_T.T_NN_model.eval()

    file_name = 'loss_step{step}.csv'
    dp_log.to_csv(os.path.join(path, file_name))
    # 将模型保存到文件
    torch.save(pinn_T.T_NN_model, f'{path}\T_NN_model_step{step}.pth')

    return T_nn


if __name__ == '__main__':
    update_T(T_now_nn=None, up_model=None, step=1, t_now=0, t_next=1, is_init=True)
