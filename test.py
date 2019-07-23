
import sympy as sym
import numpy as np
from autograd import grad as gd
from scipy import linalg
from scipy.integrate import quadrature as quad
from numpy.lib.scimath import sqrt as csqrt

print("done loading libraries")

def f(x):
	return x + 2

def g(x):
	return x**2

ts = sym.Symbol("ts")
ts0 = sym.Symbol("ts0")

a = np.array([2, 4, 6])
print("a shape", a.shape)
a = a.reshape((1, 3))
print("a shape", a.shape)
a = a.reshape((3,))
print("a shape", a.shape)
a = a.reshape((3,1))
print("a shape", a.shape)
a = a[:,0]
print("a shape", a.shape)
