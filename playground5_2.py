import numpy as np
from matplotlib import pyplot as plt
from intersection12 import Intersection12
from godunov_for_intersections import GodunovForIntersections

f_func_argmax = .5
f = lambda rho: rho*(1-rho)
f_prime = lambda rho: 1-2*rho
f_prime_inverse = lambda y: (1-y)/2
rho_L = 0.4
rho_R = .4
rho_0 = lambda x: rho_L if x<0 else rho_R
f_1 = lambda rho: f(rho_L)
f_2 = lambda rho: f(rho_R)
a = 0
b = 5
fineness = 200

rho_in_1_0 = .1
rho_out_1_0 = .4
rho_out_2_0 = .6
alpha_1 = .5
alpha_2 = .5

intersection12 = Intersection12(rho_in_1_0, rho_out_1_0, rho_out_2_0, alpha_1, alpha_2, f, f_func_argmax)
godunov = GodunovForIntersections(f, f_prime, f_prime_inverse, rho_0, f_1, f_2, a, b, fineness, intersection12)

godunov_in, godunov_out_1, godunov_out_2 = godunov.iterate([0,3,5])


for i in range(len(godunov_in)):
    plt.subplot(2,2,1)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_in[i])
    plt.title("Ingoing Edge")

    plt.subplot(2,2,2)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_out_1[i])
    plt.title("Outgoing Edge 1")

    plt.subplot(2,2,4)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_out_2[i])
    plt.title("Outgoing Edge 2")

plt.show()

