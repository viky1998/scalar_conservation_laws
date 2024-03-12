from godunov import Godunov
from riemann import Riemann
from matplotlib import pyplot as plt
import numpy as np

f = lambda rho: rho*(1-rho)
f_prime = lambda rho: 1-2*rho
f_prime_inverse = lambda y: (1-y)/2
rho_L = 0
rho_R = .5
rho_0 = lambda x: rho_L if x<0 else rho_R
f_1 = lambda rho: f(rho_L)
f_2 = lambda rho: f(rho_R)
a = -5
b = 5
fineness = 200

gg = Godunov(f, f_prime, f_prime_inverse, rho_0, f_1, f_2, a, b, fineness)

eval_times = [0,1,2,3]
eval_values = gg.iterate(eval_times)
for value in eval_values:
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), value)



riemann = Riemann(rho_L=rho_L, rho_R=rho_R)
x_span = np.arange(start=a, stop=b, step=(b-a)/5000)

for time in eval_times:
    plt.plot(x_span, riemann.pde_solution(x_span, time), '--')
 
plt.show()