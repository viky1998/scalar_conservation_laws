import numpy as np
from matplotlib import pyplot as plt
from intersection12 import Intersection12
from godunov_for_intersections12 import GodunovForIntersections

f_func_argmax = .5
f = lambda rho: rho*(1-rho)
f_prime = lambda rho: 1-2*rho
f_prime_inverse = lambda y: (1-y)/2

rho_in_1_0 = .4
rho_out_1_0 = .9
rho_out_2_0 = .7

rho_0_in = lambda x: rho_in_1_0
rho_0_out_1 = lambda x: rho_out_1_0
rho_0_out_2 = lambda x: rho_out_2_0

f_1 = lambda rho: f(rho_in_1_0)
f_2_out_1 = lambda rho: f(rho_out_1_0)
f_2_out_2 = lambda rho: f(rho_out_2_0)

a = 0
b = 5
fineness = 200


alpha_1 = .5
alpha_2 = .5

intersection12 = Intersection12(rho_in_1_0, rho_out_1_0, rho_out_2_0, alpha_1, alpha_2, f, f_func_argmax)
godunov = GodunovForIntersections(f, f_prime, f_prime_inverse, rho_0_in, rho_0_out_1, rho_0_out_2, f_1, f_2_out_1, f_2_out_2, a, b, fineness, intersection12)

godunov_in, godunov_out_1, godunov_out_2 = godunov.iterate([0,5])

for i in range(len(godunov_in)):
    plt.subplot(2,2,1)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_in[i])
    plt.ylim(-0.1, 1)
    plt.title("Ingoing Edge")

    plt.subplot(2,2,2)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_out_1[i])
    plt.ylim(-0.1, 1)
    plt.title("Outgoing Edge 1")

    plt.subplot(2,2,4)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_out_2[i])
    plt.ylim(-0.1, 1)
    plt.title("Outgoing Edge 2")

plt.show()

