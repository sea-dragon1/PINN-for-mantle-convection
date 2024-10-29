import torch
import math
from torch.utils.data import Dataset, DataLoader


class Inside_Grid(Dataset):
    def __init__(self, batch_size=1, shuffle=False, h=0.1, k=0.1):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        """
        self.h = h  # 设置空间步长
        self.k = k  # 设置时间步长
        x = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为x
        y = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为y
        t = torch.arange(0, 1 + self.k, self.k)  # 在[0,1]区间均匀取值，记为t

        # 将t和x组合， 形成时间空间网格，记录在张量X_inside中
        self.X_inside = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T

        # 速度场u
        self.u_inside = torch.ones(self.X_inside.size()[0], 2)

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):

        return len(self.X_inside)

    def __getitem__(self, idx):

        return self.X_inside[idx],self.u_inside[idx]

    def get_batches(self):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

class Boundary_Dataloader(Dataset):
    def __init__(self, batch_size=1, shuffle=False, h=0.1, k=0.1):
        """
        参数:
        batch_size (int): 每个批次的大小。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h = h  # 设置空间步长
        self.k = k  # 设置时间步长
        x = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为x
        y = torch.arange(-1, 1 + self.h, self.h)  # 在[-1,1]区间上均匀取值，记为y
        t = torch.arange(0, 1 + self.k, self.k)  # 在[0,1]区间均匀取值，记为t

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
        T_bcx1 = torch.zeros(len(bcx1))  # x=-1处的第一类边界条件 T=0
        T_bcx2 = torch.zeros(len(bcx2))  # x=1 处的第二类边界条件 T=0
        T_bcy1 = torch.zeros(len(bcy1))  # y=-1处的第一类边界条件 T=0
        T_bcy2 = torch.zeros(len(bcy2))  # y=1 处的第二类边界条件 T=0
        T_ic = -torch.sin(math.pi * ic[:, 0]) - torch.sin(math.pi * ic[:, 1])  # t=0的初值条件 T=-sin(pi*x)-sin(pi*y)
        self.T_boundary = torch.cat([T_bcx1, T_bcx2, T_bcy1, T_bcy2, T_ic])  # 将所有边界处的T值整合为一个张量
        self.T_boundary = self.T_boundary.unsqueeze(1)

    def __len__(self):

        return len(self.T_boundary)

    def __getitem__(self, idx):

        return self.X_boundary[idx], self.T_boundary[idx]

    def get_batches(self):
        """
        创建并返回一个DataLoader对象，用于分批加载数据。
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)


Inside_Grid_loader = Inside_Grid(batch_size=2, shuffle=False, h=0.01, k=0.01)
Indide_batch_loader = Inside_Grid_loader.get_batches()
print(len(Inside_Grid_loader))
Bdry_loader = Boundary_Dataloader(batch_size=2, shuffle=False, h=0.01, k=0.01)
Bdry_batch_loader = Bdry_loader.get_batches()
print(len(Bdry_loader))