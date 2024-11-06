import torch
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Inside_Grid(Dataset):
    def __init__(self, batch_size=1, shuffle=False):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        功能：
        加载网格
        """
        self.X_inside = torch.from_numpy(pd.read_csv('E:\data\mantle convection\solution_0.csv',
                                                     usecols=['Points:0','Points:1']).values).float()
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.X_inside)

    def __getitem__(self, idx):
        return self.X_inside[idx]

    def get_batches(self):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

class Init_Dataloader(Dataset):
    '''
    加载初始的数据
    '''
    def __init__(self, batch_size=1, shuffle=False):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        """
        self.df = pd.read_csv('E:\data\mantle convection\solution_0.csv',
                                     usecols=['Points:0','Points:1','velocity:0','velocity:1','T','density','viscosity','p'])
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 初始的T值
        self.X_init = torch.from_numpy(self.df[['Points:0', 'Points:1']].values).float()
        self.u_init = torch.from_numpy(self.df[['velocity:0']].values).float()
        self.v_init = torch.from_numpy(self.df[['velocity:1']].values).float()
        self.p_init = torch.from_numpy(self.df[['p']].values).float()
        self.T_init = torch.from_numpy(self.df[['T']].values).float()
        self.rho_init = torch.from_numpy(self.df[['density']].values).float()
        self.e_init = torch.from_numpy(self.df[['viscosity']].values).float()

    def __len__(self):

        return len(self.X_init)

    def __getitem__(self, idx):

        return self.X_init[idx], self.u_init[idx], self.v_init[idx], self.p_init[idx], self.T_init[idx], self.rho_init[idx], self.e_init[idx]

    def get_batches(self):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

class T_Boundary_Dataloader(Dataset):
    def __init__(self, batch_size=1, shuffle=False, h=2000, k=1):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        """
        self.df = pd.read_csv('E:\data\mantle convection\solution_0.csv',
                                     usecols=['Points:0','Points:1','velocity:0','velocity:1','T','density','viscosity','p'])
        self.bdrdf = self.df[(self.df['Points:1'] == 0) | (self.df['Points:1'] == self.df['Points:1'].max())]
        # 初始的T值
        self.X_bdry = torch.from_numpy(self.bdrdf[['Points:0', 'Points:1']].values).float()
        self.u_bdry = torch.from_numpy(self.bdrdf[['velocity:0']].values).float()
        self.v_bdry = torch.from_numpy(self.bdrdf[['velocity:1']].values).float()
        self.p_bdry = torch.from_numpy(self.bdrdf[['p']].values).float()
        self.T_bdry = torch.from_numpy(self.bdrdf[['T']].values).float()
        self.rho_bdry = torch.from_numpy(self.bdrdf[['density']].values).float()
        self.e_bdry = torch.from_numpy(self.bdrdf[['viscosity']].values).float()
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):

        return len(self.bdrdf)

    def __getitem__(self, idx):

        return self.X_bdry[idx], self.u_bdry[idx], self.v_bdry[idx], self.p_bdry[idx], self.T_bdry[idx], self.rho_bdry[idx], self.rho_bdry[idx]

    def get_batches(self):

        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)


class NS_Boundary_Dataloader(Dataset):
    def __init__(self, batch_size=1, shuffle=False, h=0.1, k=0.1):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.df = pd.read_csv('E:\data\mantle convection\solution_0.csv',
                                     usecols=['Points:0','Points:1','velocity:0','velocity:1','T','density','viscosity','p'])
        self.bdrdf = self.df[(self.df['Points:1'] == 0) | (self.df['Points:1'] == self.df['Points:1'].max())]
        # 初始的T值
        self.X_bdry = torch.from_numpy(self.bdrdf[['Points:0', 'Points:1']].values).float()
        self.u_bdry = torch.from_numpy(self.bdrdf[['velocity:0']].values).float() /365/24/3600
        self.v_bdry = torch.from_numpy(self.bdrdf[['velocity:1']].values).float() /365/24/3600
        self.p_bdry = torch.from_numpy(self.bdrdf[['p']].values).float()
        self.T_bdry = torch.from_numpy(self.bdrdf[['T']].values).float()
        self.rho_bdry = torch.from_numpy(self.bdrdf[['density']].values).float()
        self.e_bdry = torch.from_numpy(self.bdrdf[['viscosity']].values).float()

    def __len__(self):

        return len(self.p_bdry)

    def __getitem__(self, idx):

        return self.X_bdry[idx], self.u_bdry[idx], self.v_bdry[idx], self.p_bdry[idx], self.rho_bdry[idx], self.e_bdry[idx]

    def get_batches(self):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
# Inside_Grid_loader = Inside_Grid(batch_size=2, shuffle=False, h=0.01, k=0.01)
# Indide_batch_loader = Inside_Grid_loader.get_batches()
# print(len(Inside_Grid_loader))
# Bdry_loader = Boundary_Dataloader(batch_size=2, shuffle=False, h=0.01, k=0.01)
# Bdry_batch_loader = Bdry_loader.get_batches()
# print(len(Bdry_loader))