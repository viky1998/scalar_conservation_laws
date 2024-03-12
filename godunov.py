import numpy as np
from scipy import integrate

class Godunov:

    def __init__(self, f, f_prime, f_prime_inverse, rho_0, f_1, f_2, a, b, fineness) -> None:
        # initialize values
        self.f = f
        self.f_prime = f_prime
        self.f_prime_inverse = f_prime_inverse
        self.rho_0 = rho_0
        self.f_1 = f_1
        self.f_2 = f_2
        self.a = a
        self.b = b
        self.fineness = fineness

        self.grid_size = (b-a)/fineness
        self.grid = np.arange(fineness+1)*self.grid_size+a

        self.cell_averages = self.get_initial_cell_averages()
        self.current_time = 0

    def get_initial_cell_averages(self):
        cell_averages = np.zeros((1, self.fineness))
        for i in range(self.fineness):
            lower_x = self.grid[i]-self.grid_size/2
            upper_x = self.grid[i]+self.grid_size/2
            delta_x = upper_x - lower_x
            cell_averages[0,i] = 1/delta_x * integrate.quad(self.rho_0, lower_x, upper_x)[0]
        return cell_averages
    
    def calc_step(self, time_step):
        next_averages = np.zeros(self.fineness)
        last_averages = self.cell_averages[-1,:]
        next_averages[0] = last_averages[0] + (1/self.grid_size)*(integrate.quad(self.f_1, self.current_time, self.current_time+time_step)[0])-(time_step/self.grid_size)*self.f(self.riemann_solution(last_averages[0], last_averages[1]))
        next_averages[-1] = last_averages[-1] + (time_step/self.grid_size)*(self.f(self.riemann_solution(last_averages[-2], last_averages[-1])))-(1/self.grid_size)*integrate.quad(self.f_2, self.current_time, self.current_time+time_step)[0]
        for ii in range(1,len(last_averages)-1):
            if last_averages[ii] + (time_step/self.grid_size)*(self.f(self.riemann_solution(last_averages[ii-1], last_averages[ii]))-self.f(self.riemann_solution(last_averages[ii], last_averages[ii+1]))) > 0:
                pass
            next_averages[ii] = last_averages[ii] + (time_step/self.grid_size)*(self.f(self.riemann_solution(last_averages[ii-1], last_averages[ii]))-self.f(self.riemann_solution(last_averages[ii], last_averages[ii+1])))
        if (last_averages - next_averages).any():
            pass
        return next_averages
        
    def numeric_iteration(self):
        clf_temp = self.get_cfl_condition()
        self.cell_averages = np.vstack([self.cell_averages, self.calc_step(clf_temp)])
        self.current_time += clf_temp
        

    def riemann_solution(self, rho_L, rho_R):
        if rho_L==rho_R:
            return rho_L
        else:
            s = (self.f(rho_L)-self.f(rho_R))/(rho_L-rho_R)
        if self.f_prime(rho_L)>=0 and self.f_prime(rho_R)>=0:
            return rho_L
        elif self.f_prime(rho_L)>=0 and self.f_prime(rho_R)<=0 and s>=0:
            return rho_L
        elif self.f_prime(rho_L)<=0 and self.f_prime(rho_R)<=0:
            return rho_R
        elif self.f_prime(rho_L)>=0 and self.f_prime(rho_R)<=0 and s<=0:
            return rho_R
        elif self.f_prime(rho_L)<=0 and self.f_prime(rho_R)>=0:
            return self.f_prime_inverse(0)
        

    def get_cfl_condition(self):
        current_cell_averages = self.cell_averages[-1,:]
        return self.grid_size/max(abs(self.f_prime(current_cell_averages)))
    
    def iterate(self, times: list[int]):
        time_indices = sorted(range(len(times)), key=lambda k: times[k])
        eval_values = np.zeros((len(times), self.fineness))
        for i in time_indices:
            eval_time = times[i]
            while eval_time > self.current_time + self.get_cfl_condition():
                self.numeric_iteration()
            eval_values[i] = self.calc_step(eval_time-self.current_time)

        return eval_values

            