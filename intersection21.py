import numpy as np
from scipy import integrate

class Intersection21:

    def __init__(self, rho_in_1_0, rho_in_2_0, rho_out_1_0, f_func, f_func_argmax, q) -> None:
        self.q = q
        self.demand = lambda rho: f_func(rho) if rho<= f_func_argmax else f_func(f_func_argmax)
        self.supply = lambda rho: f_func(rho) if rho> f_func_argmax else f_func(f_func_argmax)

        self.gamma_max = min(self.demand(rho_in_1_0) + self.demand(rho_in_2_0), self.supply(rho_out_1_0))
        self.f_rho_in_1 = min(self.demand(rho_in_1_0), self.q/(1-self.q)*self.demand(rho_in_2_0), self.q*self.gamma_max)
        self.f_rho_in_2 = (1-self.q)/self.q*self.f_rho_in_1
        self.f_rho_out_1 = self.f_rho_in_1+self.f_rho_in_2
    
    def update(self, rho_in_1, rho_in_2, rho_out_1):
        self.gamma_max = min(self.demand(rho_in_1) + self.demand(rho_in_2), self.supply(rho_out_1))
        self.f_rho_in_1 = min(self.demand(rho_in_1), self.q/(1-self.q)*self.demand(rho_in_2), self.q*self.gamma_max)
        self.f_rho_in_2 = (1-self.q)/self.q*self.f_rho_in_1
        self.f_rho_out_1 = self.f_rho_in_1+self.f_rho_in_2
