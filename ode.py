import scipy
import numpy as np

class ODE:

    def __init__(self, v_func, L, rho_R) -> None:
        # initialize values
        self.v_func = v_func
        self.L = L
        self.rho_R = rho_R


    def ode_system(self, t, y):
        # creates the ODE System using the function v
        dydt = np.zeros_like(y)
        for i in range(int(1/self.L-1)):
            dydt[i] = self.v_func(self.L / (y[i+1] - y[i]))
        dydt[-1] = self.v_func(self.rho_R)

        return dydt


    def ode_solver(self, x_start, time_span):
        # uses the ODE Solver with Runge-Kutta method to solve the ODE System
        return scipy.integrate.solve_ivp(
            fun=self.ode_system, 
            t_span=time_span, 
            y0=x_start, method='RK23',
            max_step=np.exp(-8))