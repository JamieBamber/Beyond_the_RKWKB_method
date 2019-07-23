# Script to implement a fourth order Runge-Kutta method
#
# © https://www.codeproject.com/Tips/792927/%2FTips%2F792927%2FFourth-Order-Runge-Kutta-Method-in-Python

import numpy as np
#import scipy as sp

# define 4th order RK integrator function
def rKN(x, fx, n, hs):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    xk = []
    for i in range(n):
        k1.append(fx[i](x)*hs)
    for i in range(n):
        xk.append(x[i] + k1[i]*0.5)
    for i in range(n):
        k2.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k2[i]*0.5
    for i in range(n):
        k3.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k3[i]
    for i in range(n):
        k4.append(fx[i](xk)*hs)
    for i in range(n):
        x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
    return x

# adapt this to implement the more efficient Runge-Kutta 4(5) method ?

"""
from the RKWKB paper:

A general RK method can be written as 

y_n+1 = y_n + h Σ[b_i*k_i]						(A.1)

k_s = f(t_n + c_s*h , y_n + h Σ[a_si*k_i] ) 	(A.2)

a_si = Runge–Kutta matrix
b_i  = weights
c_i  = nodes

The RK4(5) method is slightly more complicated. It is also called the  Runge–Kutta–Fehlberg method and uses two
methods order 4 and 5.

"""





