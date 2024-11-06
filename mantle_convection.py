import pandas as pd
import csv
import os
import numpy as np
import updater_up,updater_T
import torch

device = 'cuda'
data_path = f'E:\data\mantle convection'

X_inside_pd = pd.read_csv('E:\data\mantle convection\solution_0.csv',usecols=['Points:0', 'Points:1'])
X_inside = torch.from_numpy(X_inside_pd.values).to(device).float()
steps = 3
t = [_ for _ in range(0,100)]
for step in range(steps):
    file_name = f'solution_{step+1}'
    csv_file_path = f'{data_path}\\{file_name}.csv'
    X_inside_pd.to_csv(csv_file_path)
    if step == 0:
        is_init = True
        T_nn = updater_T.update_T(None ,
                                     None,
                                     step,
                                     t[step],
                                     t[step+1],
                                     is_init)
    else:
        is_init = False
        T_nn = updater_T.update_T(T_nn ,
                                     up_nn,
                                     step,
                                     t[step],
                                     t[step+1],
                                     is_init= False)
    t_now_tensor = torch.full_like((X_inside[:, 0]), t[step+1], device=X_inside.device).unsqueeze(0)
    Xt_now = torch.cat((X_inside, t_now_tensor.T), dim=1).to(device)
    T_now = T_nn(Xt_now).detach().cpu().numpy()
    df = pd.read_csv(csv_file_path)
    df['T'] = T_now
    df.to_csv(csv_file_path,index=None)

    up_nn = updater_up.update_up(step, t_now=t[step])
    up_now = up_nn(X_inside).detach().cpu().numpy()
    df = pd.read_csv(csv_file_path)
    df_new = pd.DataFrame(up_now, columns=['velocity:0', 'velocity:1', 'p'])
    df_combined = pd.concat([df, df_new], axis=1)
    df_combined.to_csv(csv_file_path, index=False)











