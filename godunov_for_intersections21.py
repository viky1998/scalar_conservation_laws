from intersection21 import Intersection21
import numpy as np
from scipy import integrate

class GodunovForIntersections:
    def __init__(self, f, f_prime, f_prime_inverse, rho_0_in_1, rho_0_in_2, rho_0_out_1, f_1_in_1, f_1_in_2, f_2_out_1, a, b, fineness, intersection: Intersection21) -> None:
        self.f = f
        self.f_prime = f_prime
        self.f_prime_inverse = f_prime_inverse
        self.rho_0_in_1 = rho_0_in_1
        self.rho_0_in_2 = rho_0_in_2
        self.rho_0_out_1 = rho_0_out_1
        self.f_1_in_1 = f_1_in_1
        self.f_1_in_2 = f_1_in_2
        self.f_2_out_1 = f_2_out_1
        self.a = a
        self.b = b
        self.fineness = fineness

        self.intersection = intersection

        self.grid_size = (b-a)/fineness
        self.grid = np.arange(fineness+1)*self.grid_size+a

        self.cell_averages_in_1 = self.get_initial_cell_averages('in_1')
        self.cell_averages_in_2 = self.get_initial_cell_averages('in_2')
        self.cell_averages_out_1 = self.get_initial_cell_averages('out_1')
        self.current_time = 0

        self.f_rho_in_1 = self.intersection.f_rho_in_1
        self.f_rho_in_2 = self.intersection.f_rho_in_2
        self.f_rho_out_1 = self.intersection.f_rho_out_1

        
    def get_initial_cell_averages(self, edge):
        if edge == 'in_1':
            rho_0 = self.rho_0_in_1
        elif edge == 'in_2':
            rho_0 = self.rho_0_in_2
        elif edge == 'out_1':
            rho_0 = self.rho_0_out_1
        
        cell_averages = np.zeros((1, self.fineness))
        for i in range(self.fineness):
            lower_x = self.grid[i]-self.grid_size/2
            upper_x = self.grid[i]+self.grid_size/2
            delta_x = upper_x - lower_x
            cell_averages[0,i] = 1/delta_x * integrate.quad(rho_0, lower_x, upper_x)[0]
        return cell_averages
        
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
        cfl_in_1 = self.grid_size/max(abs(self.f_prime(self.cell_averages_in_1[-1,:])))
        cfl_in_2 = self.grid_size/max(abs(self.f_prime(self.cell_averages_in_2[-1,:])))
        cfl_out_1 = self.grid_size/max(abs(self.f_prime(self.cell_averages_out_1[-1,:])))
        min_cfl = min(cfl_in_1, cfl_in_2, cfl_out_1)
        return min_cfl
    
    def numeric_iteration(self):
        cfl_temp = self.get_cfl_condition()
        self.cell_averages_in_1 = np.vstack([self.cell_averages_in_1, self.calc_step_in_1(cfl_temp)])
        self.cell_averages_in_2 = np.vstack([self.cell_averages_in_2, self.calc_step_in_2(cfl_temp)])
        self.cell_averages_out_1 = np.vstack([self.cell_averages_out_1, self.calc_step_out_1(cfl_temp)])

        rho_in_1 = self.cell_averages_in_1[-1,-1]
        rho_in_2 = self.cell_averages_in_2[-1,-1]
        rho_out_1 = self.cell_averages_out_1[-1,-1]
        self.intersection.update(rho_in_1, rho_in_2, rho_out_1)

        self.current_time += cfl_temp

    def calc_step_in_1(self, time_step):
        next_averages = np.zeros(self.fineness)
        last_averages = self.cell_averages_in_1[-1,:]
        next_averages[0] = last_averages[0] + \
            (1/self.grid_size)*(integrate.quad(self.f_1_in_1, self.current_time, self.current_time+time_step)[0])-\
            (time_step/self.grid_size)*self.f(self.riemann_solution(last_averages[0], last_averages[1]))
        next_averages[-1] = last_averages[-1] + \
            (time_step/self.grid_size)*\
            (self.f(self.riemann_solution(last_averages[-2], last_averages[-1]))-self.f_rho_in_1)
        for ii in range(1,len(last_averages)-1):
            next_averages[ii] = last_averages[ii] + \
                (time_step/self.grid_size)*\
                (self.f(self.riemann_solution(last_averages[ii-1], last_averages[ii]))-\
                 self.f(self.riemann_solution(last_averages[ii], last_averages[ii+1])))
        return next_averages
    
    def calc_step_in_2(self, time_step):
        next_averages = np.zeros(self.fineness)
        last_averages = self.cell_averages_in_2[-1,:]
        next_averages[0] = last_averages[0] + \
            (1/self.grid_size)*(integrate.quad(self.f_1_in_2, self.current_time, self.current_time+time_step)[0])-\
            (time_step/self.grid_size)*self.f(self.riemann_solution(last_averages[0], last_averages[1]))
        next_averages[-1] = last_averages[-1] + \
            (time_step/self.grid_size)*\
            (self.f(self.riemann_solution(last_averages[-2], last_averages[-1]))-self.f_rho_in_2)
        for ii in range(1,len(last_averages)-1):
            next_averages[ii] = last_averages[ii] + \
                (time_step/self.grid_size)*\
                    (self.f(self.riemann_solution(last_averages[ii-1], last_averages[ii]))-\
                     self.f(self.riemann_solution(last_averages[ii], last_averages[ii+1])))
        return next_averages
    
    def calc_step_out_1(self, time_step):
        next_averages = np.zeros(self.fineness)
        last_averages = self.cell_averages_out_1[-1,:]
        next_averages[0] = last_averages[0] + \
            (time_step/self.grid_size)*(self.f_rho_out_1-\
                self.f(self.riemann_solution(last_averages[0], last_averages[1])))
        next_averages[-1] = last_averages[-1] + \
            (time_step/self.grid_size)*(self.f(self.riemann_solution(last_averages[-2], last_averages[-1])))-\
                (1/self.grid_size)*integrate.quad(self.f_2_out_1, self.current_time, self.current_time+time_step)[0]
        for ii in range(1,len(last_averages)-1):
            next_averages[ii] = last_averages[ii] + \
                (time_step/self.grid_size)*\
                    (self.f(self.riemann_solution(last_averages[ii-1], last_averages[ii]))-\
                     self.f(self.riemann_solution(last_averages[ii], last_averages[ii+1])))
        return next_averages
    
    def iterate(self, times: list[int]):
        time_indices = sorted(range(len(times)), key=lambda k: times[k])
        eval_rho_in_1 = np.zeros((len(times), self.fineness))
        eval_rho_in_2 = np.zeros((len(times), self.fineness))
        eval_rho_out_1 = np.zeros((len(times), self.fineness))

        for i in time_indices:
            eval_time = times[i]
            while eval_time > self.current_time + self.get_cfl_condition():
                self.numeric_iteration()
            eval_rho_in_1[i] = self.calc_step_in_1(eval_time-self.current_time)
            eval_rho_in_2[i] = self.calc_step_in_2(eval_time-self.current_time)
            eval_rho_out_1[i] = self.calc_step_out_1(eval_time-self.current_time)

        return eval_rho_in_1, eval_rho_in_2, eval_rho_out_1