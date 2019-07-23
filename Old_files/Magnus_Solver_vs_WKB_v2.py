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

"""
Define a function for the 2x2 matrix from a 1D equation of the form

x'' + ω^2(t)x = 0

[ x' ]     [ 0      1 ] [ x ]
[ x'']  =  [ -ω^2   0 ] [ x']

"""
def A_from_w2(w2, num_vs_sym):
	def f(t):
		if num_vs_sym:
			# numpy matrix
			M = np.matrix([[0, 1], [-w2(t), 0]])
		elif num_vs_sym == False:
			# sympy matrix
			M = sym.Matrix([[0, 1], [-w2(t), 0]])
		return M
	return f

# --- Airy equation stuff --- #
Airy = {}
Airy["name"] = "Airy"
Ai0, Aip0, Bi0, Bip0 = special.airy(0)
Airy["x0"] = np.array([Ai0, -Aip0])
Airy["ylim"] = (-0.75, 0.75)

def w2_Airy(t):
	return t
Airy["w2"] = w2_Airy
Airy["A"] = A_from_w2(w2_Airy, True)

def Airy_sol(t):
	Ai0, Aip0, Bi0, Bip0 = special.airy(0)
	M = (1/(-Ai0*Bip0 + Aip0*Bi0))*np.matrix([[-Bip0, -Bi0], [+Aip0, Ai0]])
	ab = M @ Airy["x0"].reshape(2, 1)	
	Ai, Aip, Bi, Bip = special.airy(-t)
	a = ab[0, 0]
	b = ab[1, 0]
	x_true = a*Ai + b*Bi
	dxdt_true = a*Aip + b*Bip
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x
	
Airy["true_sol"] = Airy_sol # function
Airy["t_start"] = 0
Airy["t_stop"] = 35
# ---------------------------- #
# --- Burst equation stuff --- #
burst = {}
burst["n"] = 40
burst["name"] = "Burst_n=" + str(burst["n"])
burst["ylim"] = (-0.5, 0.5)

def w2_burst(t):
	n = burst["n"]
	w = (n**2 - 1)/(1 + t**2)**2
	return w
burst["w2"] = w2_burst
burst["A"] = A_from_w2(w2_burst, True)

def burst_soln(t, n):
	if n % 2 == 0:
		x = (np.sqrt(1 + t**2)/n)*((-1)**(n/2))*np.sin(n*np.arctan(t))
	elif (n+1) % 2 == 0:
		x = (np.sqrt(1 + t**2)/n)*((-1)**((n-1)/2))*np.cos(n*np.arctan(t))
	return x

def dburst_soln(t, n):
	if n % 2 == 0:
		x = (1/(np.sqrt(1 + t**2)*n) )*((-1)**(n/2))*(t*np.sin(n*np.arctan(t)) + n*np.cos(n*np.arctan(t)))
	elif (n+1) % 2 == 0:
		x = (1/(np.sqrt(1 + t**2)*n))*((-1)**((n-1)/2))*(t*np.cos(n*np.arctan(t)) - n*np.sin(n*np.arctan(t)))
	return x

def Burst_sol(t):
	x_true = burst_soln(t, burst["n"])
	dxdt_true = dburst_soln(t, burst["n"])
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x

burst["t_start"] = -0.5*burst["n"]
burst["t_stop"] = +0.5*burst["n"]
burst["x0"] = x0 = np.array([burst_soln(burst["t_start"],burst["n"]), dburst_soln(burst["t_start"],burst["n"])])
burst["true_sol"] = Burst_sol

# ---------------------------- #

#### Choose equation

Eq = burst

t_start = -10
t_stop = 10

########## Symbolics #################
#
#	Symbolic manipulation using sympy

A_sym = A_from_w2(Eq["w2"], False)
	
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
	return sym.sqrt(Eq["w2"](t))

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
	
A = Eq["A"]

t0 = np.array([t_start])
x0 = Eq["true_sol"](t0).reshape(2, 1)
print("x0 = ", x0) 
	
########## Integration #################

t_start0 = -20
t_stop0 = 20

n_steps = 500
t_vec = np.linspace(t_start, t_stop, n_steps)
t_vec0 = np.linspace(t_start0, t_stop0, 1000)
	
def Integrator_1(t_vec, x0, M):
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(2)
	for i in range(1,len(t_vec)):
		x[i,:] = (M(t_vec[0], t_vec[i]) @ x0).reshape(2)
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

filename = "WKB_vs_Magnus_" + Eq["name"] + "_t_start=" + str(t_start) + "_t_stop=" + str(t_stop) 
if Eq == burst:
	title = "Burst equation (n = " + str(burst["n"]) + ") : WKB vs Magnus expansion"
elif Eq == Airy:
	title = "Airy equation : WKB vs Magnus expansion"

line_marker = '-'
colours = ['r', 'b', 'g']
x_true = Eq["true_sol"](t_vec0)[:,0]
plt.plot(t_vec0, x_true, color="0", linewidth=1, linestyle="--", label="true soln.")
plt.plot(t_vec, x_M1_exact, 'g-', markersize=2, linewidth=1, label="Magnus sol. with only $\\Omega_1$")
plt.plot(t_vec, x_M2_exact, 'b-', markersize=2, linewidth=1, label="Magnus sol. with $\\Omega_1 + \\Omega_2$")
plt.plot(t_vec, x_WKB_exact, 'r-', markersize=2, linewidth=1, label="WKB sol.")
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-0.5, 0.5)
plt.title(title)
plt.minorticks_on()
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()
print("made plot")
print("saved plot as " + "Plots/" + filename + ".pdf")
