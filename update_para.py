import numpy as np

def update_rho(X,T,T_0,rho_0,alpha):
    rho = rho_0 *(1-alpha*(T-T_0))
    return