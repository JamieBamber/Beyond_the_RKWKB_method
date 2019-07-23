# Magnus Solver comparison with WKB approximation

"""
To solve 

x''(t) = -ω^2(t) x(t)
"""

import numpy as np

import time
import sympy as sym
from scipy import special, linalg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_start = time.time()

############# Set up Equations / A matrices ########################

########## Symbolics #################
#
#	Symbolic manipulation using sympy

t_add = 5
t_stop = 20

def w2(t):
	return (t+t_add)

def A_from_w2_sym(w2):
	def f(t):
		M = sym.Matrix([[0, 1], [-w2(t), 0]])
		return M
	return f
	
A_sym = A_from_w2_sym(w2)
	
"""
define the first and second terms of the Magnus expansion (symbolic form)

Ω_1(t) = \int_t_0^t ( A(t') ) dt'

Ω_2(t) = 0.5 \int_t_0^t( \int_t_0^t'( [A(t'),A(t'')] )dt'' )dt'

"""

ts0 = sym.Symbol('ts0')
ts = sym.Symbol('ts')
ts1 = sym.Symbol('ts1')

def Com(A, B):
	return (A*B - B*A)

def Omega_1_sym(A):
	ts1 = sym.Symbol('ts1')
	integral = sym.integrate(A(ts1), (ts1, ts0, ts))
	return integral
	
def Omega_2_sym(A):
	ts1 = sym.Symbol('ts1')
	ts2 = sym.Symbol('ts2')
	integral_1 = sym.integrate(Com(A(ts1),A(ts2)), (ts2, ts0, ts1))
	integral_2 = sym.integrate(integral_1, (ts1, ts0, ts))
	return 0.5*integral_2
	
def w1(t):
	return sym.sqrt(w2(t))

def WKB_matrix_sym():
	xA = sym.cos(sym.integrate(w1(ts1), (ts1, ts0, ts)))/sym.sqrt(w1(ts))
	xB = sym.sin(sym.integrate(w1(ts1), (ts1, ts0, ts)))/sym.sqrt(w1(ts))
	dxA = sym.diff(xA, ts)
	dxB = sym.diff(xB, ts)
	x_mat = sym.Matrix([[xA, xB], [dxA, dxB]])
	x_mat_0 = x_mat.subs(ts, ts0)
	M = x_mat*x_mat_0.inv()
	return M

'''
def WKB_matrix_sym():
	# define the equivalent matrix for the 2nd order S0 + S1 WKB method
	M11 = sym.cos(sym.integrate(w1(ts1), (ts1, ts0, ts)))*sym.sqrt(w1(ts0)/w1(ts))
	dw = sym.diff(w1(ts), ts)
	M12 = (dw.subs(ts, ts0)/(2*w2(ts0)))*sym.sin(sym.integrate(w1(ts1), (ts1, ts0, ts)))*sym.sqrt(w1(0)/w1(ts))
	M21 = -(dw/(2*w1(ts)**(3/2)*sym.sqrt(w1(ts0))))*( sym.cos(sym.integrate(w1(ts1), (ts1, ts0, ts))) 
		+ (dw.subs(ts, ts0)/(2*w1(ts0)))*sym.sin(sym.integrate(w1(ts1), (ts1, ts0, ts))) )
	M22 = M11 - (dw/(2*w1(ts)**(3/2)*sym.sqrt(w1(ts0))))*sym.sin(sym.integrate(w1(ts1), (ts1, ts0, ts)))
	M = sym.Matrix([[M11, M12], [M21, M22]])
	return M
'''

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']

print("Omega 1 = ", Omega_1_sym(A_sym))
Omega_1_exact = sym.lambdify((ts0, ts), Omega_1_sym(A_sym), modules=array2mat)
print("Omega 2 = ", Omega_2_sym(A_sym))
Omega_2_exact = sym.lambdify((ts0, ts), Omega_2_sym(A_sym), modules=array2mat)
print("WKB matrix = ", WKB_matrix_sym()) 
WKB_matrix_exact = sym.lambdify((ts0, ts), WKB_matrix_sym(), modules=array2mat)

########## Numerics #################

def A_from_w2_num(w2):
	def f(t):
		M = np.matrix([[0, 1], [-w2(t), 0]])
		return M
	return f
	
A = A_from_w2_num(w2)

Ai0, Aip0, Bi0, Bip0 = special.airy(-t_add)

x0 = np.array([Ai0, -Aip0]).reshape(2, 1)

def true_soln(t):
	# true solution with w^2 = (t+1)
	Ai, Aip, Bi, Bip = special.airy(-(t+t_add))
	x_true = Ai
	dxdt_true = Aip
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x
	
########## Integration #################

n_steps = 500
t_vec = np.linspace(0, t_stop, n_steps)
t_vec0 = np.linspace(0, t_stop, 1000)
	
def Integrator_1(t_vec, x0, M):
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(2)
	for i in range(1,len(t_vec)):
		x[i,:] = (M(0, t_vec[i]) @ x0).reshape(2)
		if (i*100) % (len(t_vec)-1) == 0:
			print("\r" + "integrated {:.0%}".format(i/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return x
	
######### plot graph ##################

def M1(t0, t):
	return linalg.expm(Omega_1_exact(t0, t))
	
def M2(t0, t):
	return linalg.expm(Omega_1_exact(t0, t) + Omega_2_exact(t0, t))

x_M1_exact = Integrator_1(t_vec, x0, M1)[:,0]
x_M2_exact = Integrator_1(t_vec, x0, M2)[:,0]
x_WKB_exact = Integrator_1(t_vec, x0, WKB_matrix_exact)[:,0]

filename = "WKB_vs_Magnus_Airy_t_add_" + str(t_add)
title = "Modified Airy equation $\\ddot{x} = -(t+$" + str(t_add) + "$)x$ : WKB vs Magnus expansion"

line_marker = '-'
colours = ['r', 'b', 'g']
x_true = true_soln(t_vec0)[:,0]
plt.plot(t_vec0, x_true, color="0", linewidth=2, linestyle="--", label="true soln.")
plt.plot(t_vec, x_M1_exact, 'g-', markersize=2, linewidth=1, label="Magnus sol. with only $\\Omega_1$")
plt.plot(t_vec, x_M2_exact, 'b-', markersize=2, linewidth=1, label="Magnus sol. with $\\Omega_1 + \\Omega_2$")
plt.plot(t_vec, x_WKB_exact, 'r-', markersize=2, linewidth=1, label="WKB sol.")
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-1, 1)
plt.title(title)
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()
print("made plot")
print("saved plot as " + "Plots/" + filename + ".pdf")
