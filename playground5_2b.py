import numpy as np
from matplotlib import pyplot as plt
from intersection21 import Intersection21
from godunov_for_intersections21 import GodunovForIntersections

f_func_argmax = .5
f = lambda rho: rho*(1-rho)
f_prime = lambda rho: 1-2*rho
f_prime_inverse = lambda y: (1-y)/2

rho_in_1_0 = 0.8
rho_in_2_0 = 0.5
rho_out_1_0 = 0.3

rho_0_in_1 = lambda x: rho_in_1_0
rho_0_in_2 = lambda x: rho_in_2_0
rho_0_out_1 = lambda x: rho_out_1_0

f_1_in_1 = lambda rho: f(rho_in_1_0)
f_1_in_2 = lambda rho: f(rho_in_2_0)
f_2_out_1 = lambda rho: f(rho_out_1_0)

a = 0
b = 5
fineness = 200

q=0.5

intersection21 = Intersection21(rho_in_1_0, rho_in_2_0, rho_out_1_0, f, f_func_argmax, q)

godunov = GodunovForIntersections(f, f_prime, f_prime_inverse, rho_0_in_1, rho_0_in_2, rho_0_out_1, f_1_in_1, f_1_in_2, f_2_out_1, a, b, fineness, intersection21)

godunov_in_1, godunov_in_2, godunov_out_1 = godunov.iterate([0,5])

for i in range(len(godunov_in_1)):
    plt.subplot(2,2,1)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_in_1[i])
    plt.ylim(-0.1, 1)
    plt.title("Ingoing Edge 1")

    plt.subplot(2,2,3)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_in_2[i])
    plt.ylim(-0.1, 1)
    plt.title("Ingoing Edge 2")

    plt.subplot(2,2,4)
    plt.plot(list(np.arange(a,b,(b-a)/fineness)), godunov_out_1[i])
    plt.ylim(-0.1, 1)
    plt.title("Outgoing Edge 1")

plt.show()

