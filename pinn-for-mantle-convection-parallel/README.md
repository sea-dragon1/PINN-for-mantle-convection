# 📘 PINN for Mantle Convection（PINN求解地幔对流）


> 利用深度学习尤其是PINN(基于物理信息约束的神经网络)求解地幔对流问题。
## 进程说明
    1. 2024-10-23 实现按batch取点输入网络进行训练，主要是因为直接所有点全局输入会超出显存，无法运行。相当于梯度下降
    2. 2024-10-29 将代码改为更方便使用的对分量求导形式，将train从model类中分离出来，实现NS方程耦合求解
    3. 2024-10-30 耦合求解存在问题，改为分别求解循环迭代
## 🎉 Features

- 特性1
- 特性2
- 特性3

## 📸 Preview


## 🚀 Installation

在你的终端中运行以下命令来安装此项目：

```bash
git clonehttps://github.com/sea-dragon1/PINN-for-mantle-convection/tree/main
cd yourproject
npm install
```
## 🔧 Setup
在项目目录中，运行以下命令来设置你的开发环境：

```bash
npm run setup
```
## 🛠 Usage

这里是如何使用你的项目：

python
## 示例代码
💻 Tech Stack
技术1
技术2
技术3
## 🤝 Contributing
请阅读贡献指南来了解如何为项目做出贡献。

## 📜 License
此项目使用MIT许可证。

## 👤 Authors
刘海龙 - hiloong
## 💌 Support
如果你对这个项目有任何问题或建议，可以通过Issues与我联系。
    
邮箱：810695865@qq.com
    
微信：mathdragon_THU
