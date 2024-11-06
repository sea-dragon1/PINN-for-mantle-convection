# https://blog.csdn.net/weixin_43325228/article/details/132317132
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

class FC_Network(nn.Module):

    def __init__(self,
                input_size, # 输入层神经元个数
                hidden_size, # 隐藏层神经元个数
                output_size, # 输出层神经元个数
                depth, # 隐藏层深度
                act = torch.nn.Tanh):

        super(FC_Network, self).__init__() # 调用父类的构造函数

        # 输入层
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        # 隐藏层
        for i in range(depth):
            layers.append(
                ('hidden_%d' %i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('input_activation_%d' %i, act()))

        # 输出层
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))
        # 将这些层组装在一起
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    #   前向计算方法
    def forward(self, x):
        return self.layers(x)

