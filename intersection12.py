import numpy as np
from scipy import integrate

class Intersection12:

    def __init__(self, rho_in_1_0, rho_out_1_0, rho_out_2_0, alpha_1, alpha_2, f_func, f_func_argmax) -> None:
        assert alpha_1+alpha_2 == 1, 'alpha not correct'
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.demand = lambda rho: f_func(rho) if rho<= f_func_argmax else f_func(f_func_argmax)
        self.supply = lambda rho: f_func(rho) if rho> f_func_argmax else f_func(f_func_argmax)

        self.f_rho_in_1 = min(self.demand(rho_in_1_0), self.supply(rho_out_1_0)/alpha_1, self.supply(rho_out_2_0)/alpha_2)
        self.f_rho_out_1 = alpha_1*self.f_rho_in_1
        self.f_rho_out_2 =  alpha_2*self.f_rho_in_1
    
    def update(self, rho_in_1, rho_out_1, rho_out_2):
        self.f_rho_in_1 = min(self.demand(rho_in_1), self.supply(rho_out_1)/self.alpha_1, self.supply(rho_out_2)/self.alpha_2)
        self.f_rho_out_1 = self.alpha_1*self.f_rho_in_1
        self.f_rho_out_2 =  self.alpha_2*self.f_rho_in_1
