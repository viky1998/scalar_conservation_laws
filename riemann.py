import numpy as np

class Riemann:

    def __init__(self, rho_L, rho_R) -> None:
        # initialize values
        self.rho_L = rho_L
        self.rho_R = rho_R

        self.f = lambda rho: rho*(1-rho)
        self.f_prime = lambda rho: 1-2*rho
        self.f_prime_inverse = lambda y: (1-y)/2
        self.s = (self.f(rho_L)-self.f(rho_R))/(rho_L-rho_R)


    def rh_func(self, x, t):
        # rho function if rho_L < rho_R
        # then for this example the Rankine-Hugonoit Conditions are satified
        if x < self.s*t:
            return self.rho_L
        elif x > self.s*t:
            return self.rho_R
        
    def else_func(self, x, t):
        # rho function if rho_L > rho_R
        if x < self.f_prime(self.rho_L)*t:
            return self.rho_L
        
        elif (self.f_prime(self.rho_L)*t < x) and (x < self.f_prime(self.rho_R)*t):
            return self.f_prime_inverse(x/t)
        
        else:
            return self.rho_R

    def pde_solution(self, x_span, time):
        # for given x-values and a time t, this function returns the corresponding rho values
        if self.rho_L < self.rho_R:
            solutions = [self.rh_func(x,time) for x in x_span]
            return np.array(solutions)
        elif self.rho_L > self.rho_R:
            solutions = [self.else_func(x,time) for x in x_span]
            return np.array(solutions)
