# Magnus Solver

"""
To solve 

x'(t) = A(t) x(t)

"""
"""
Using Magnus expansion derived numerical methods from
"the Magnus expansion and some of its applications"
pg. 91 - 95

and new related methods.
"""

import numpy as np
from numpy.lib.scimath import sqrt as csqrt

import time
import sympy as sym
from scipy import special, linalg

# choose numerical integrator
from scipy.integrate import quadrature as quad

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import gridspec

from sys import exit as sysexit

T_start = time.time()

############# Set up Equations / A matrices ########################

"""
Define a function for the A matrix and the true solution
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

def Simplify(Expr):
	#E1 = sym.powsimp(Expr, deep=True, force=True)
	E1 = sym.simplify(Expr)
	E2 = sym.nsimplify(E1)
	return E2

ts0 = sym.Symbol('ts0', real=True)
ts = sym.Symbol('ts', real=True)
ts1 = sym.Symbol('ts1', real=True)

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']
array2mat_c = [{'ImmutableDenseMatrix': np.matrix}, {'sqrt': csqrt}, 'numpy']

# --- Airy equation stuff --- #
Airy = {}
Airy["name"] = "Airy"
Airy["t_start"] = 1
Airy["t_stop"] = 35
Ai0, Aip0, Bi0, Bip0 = special.airy(-Airy["t_start"])
Airy["x0"] = np.array([Ai0, -Aip0])
Airy["ylim"] = (-0.75, 0.75)

def w2_Airy(t):
	return t
Airy["w2"] = w2_Airy
Airy["A_num"] = A_from_w2(w2_Airy, True)
Airy["A_sym"] = A_from_w2(w2_Airy, False)(ts)

def Airy_sol(t):
	Ai0, Aip0, Bi0, Bip0 = special.airy(-Airy["t_start"])
	M = np.linalg.inv(np.matrix([[Ai0, Bi0], [-Aip0, -Bip0]]))
	ab = M @ Airy["x0"].reshape(2, 1)	
	Ai, Aip, Bi, Bip = special.airy(-t)
	a = ab[0, 0]
	b = ab[1, 0]
	x_true = a*Ai + b*Bi
	dxdt_true = -(a*Aip + b*Bip)
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x
	
Airy["true_sol"] = Airy_sol # function

Airy["title"] = "Airy equation"
# ---------------------------- #
# --- Burst equation stuff --- #
burst = {}
burst["n"] = 40
burst["name"] = "Burst_n=" + str(burst["n"]) #+ "_centre"
burst["ylim"] = (-0.4, 0.4)

def w2_burst(t):
	n = burst["n"]
	w = (n**2 - 1)/(1 + t**2)**2
	return w
burst["w2"] = w2_burst
burst["A_num"] = A_from_w2(w2_burst, True)
burst["A_sym"] = A_from_w2(w2_burst, False)(ts)

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

burst["t_start"] = -10
burst["t_stop"] = +10
#burst["t_start"] = -2
#burst["t_stop"] = +2
burst["x0"] = Burst_sol(np.array([burst["t_start"]]))
burst["true_sol"] = Burst_sol
burst["title"] = "Burst equation (n = " + str(burst["n"]) + ")"
# ---------------------------- #
# -- Triplet equation stuff -- #

def f(t):
	return (t*t+t)
	
def f_num(t):
	return (t*t+t)
	
def g(t):
	return (sym.exp(-t**2/64))
	
def g_num(t):
	return (np.exp(-t**2/64))

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

F = f(ts)
G = g(ts)
dF = sym.diff(F, ts)
ddF = sym.diff(dF, ts)
dG = sym.diff(G, ts)
ddG = sym.diff(dG, ts)

#print("F = ", F)
#print("G = ", G)
#print("dF = ", dF)
#print("dG = ", dG)
#print("ddG = ", ddG)
#print("ddF = ", ddF)

A_sym_triplet = sym.Matrix([[dG/G, -dF, 0], [+dF, 0, 1], [+(dF*dG)/G, ddG/G, 0]])
#A = sym.Matrix([[0, 1, 0], [(ddG/G - dF**2), 0, ddF - 2*dG/G], [dF, 0, dG/G]])
	
A_num_triplet = sym.lambdify((ts), A_sym_triplet, modules=array2mat)
	
triplet = {}
triplet["name"] = "triplet_v2"
triplet["title"] = "3D equation, with $x = e^{-t^2/4}\\cos(t^2)$"
triplet["n"] = 10
triplet["sigma"] = 4
triplet["f"] = f
triplet["g"] = g
triplet["A_sym"] = Simplify(A_sym_triplet)
triplet["A_num"] = A_num_triplet

def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

triplet["w2"] = 0	# dummy value
triplet["true_sol"] = triplet_sol
triplet["ylim"] = (-1.0, 1.0)
triplet["t_start"] = 1
triplet["t_stop"] = 20
triplet["x0"] = triplet_sol(np.array([triplet["t_start"]]))

# ---------------------------- #
# -- Second Triplet equation stuff -- #
"""
def f(t):
	return (4*t**(5/2))
	
def f_num(t):
	return (4*t**(5/2))
	
def g(t):
	return (sym.exp(-t**2/64))
	
def g_num(t):
	return (np.exp(-t**2/64))

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

F = f(ts)
G = g(ts)
dF = sym.diff(F, ts)
ddF = sym.diff(dF, ts)
dG = sym.diff(G, ts)
ddG = sym.diff(dG, ts)

print("F = ", F)
print("G = ", G)
print("dF = ", dF)
print("dG = ", dG)
print("ddG = ", ddG)
print("ddF = ", ddF)


A_sym_triplet2 = sym.Matrix([[dG/G, -dF, 0], [+dF, 0, 1], [+(dF*dG)/G, ddG/G, 0]])
#A_sym_triplet2 = sym.Matrix([[0, 1, 0], [(ddG/G - dF**2), 0, ddF - 2*dG/G], [dF, 0, dG/G]])
A_sym_triplet2 = sym.simplify(A_sym_triplet2)

#A_sym_triplet2 = sym.Matrix([[0, -dF, 1], [dF, dG/G, 0], [ddG/G, -dF*dG/G, 0]])
A_num_triplet2 = sym.lambdify((ts), A_sym_triplet2, modules=array2mat)
	
triplet2 = {}
triplet2["name"] = "triplet2"
triplet2["title"] = "3D equation, with $x = e^{-t/4}\\cos((2 + \\cos(t))t)$"
triplet2["n"] = 10
triplet2["sigma"] = 4
triplet2["f"] = f
triplet2["g"] = g
triplet2["A_sym"] = A_sym_triplet2
triplet2["A_num"] = A_num_triplet2

def triplet2_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.cos(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_


def triplet2_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(dg(t)*np.cos(f_num(t)) - df(t)*g_num(t)*np.sin(f_num(t)))
	z = np.array(g_num(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

	
triplet2["w2"] = 0	# dummy value
triplet2["true_sol"] = triplet2_sol
triplet2["t_start"] = 1
triplet2["t_stop"] = 10
triplet2["x0"] = triplet2_sol(np.array([triplet2["t_start"]]))
"""
# ---------------------------- #
# -- doublet equation stuff -- #
def f(t):
	return (t**2)
	
def f_num(t):
	return (t**2)
	
def g(t):
	return (1/t)
	
def g_num(t):
	return (1/t)

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

F = f(ts)
G = g(ts)
dF = sym.diff(F, ts)
ddF = sym.diff(dF, ts)
dG = sym.diff(G, ts)
ddG = sym.diff(dG, ts)
"""
print("F = ", F)
print("G = ", G)
print("dF = ", dF)
print("dG = ", dG)
print("ddF = ", ddF)
print("ddG = ", ddG)
"""
A_sym_doublet = sym.Matrix([[0, 1], [ddG/G - dF**2 - 2*(dG/G)**2 - (dG*ddF)/(G*dF), ddF/dF + 2*dG/G]])
	
A_num_doublet = sym.lambdify((ts), A_sym_doublet, modules=array2mat)
	
doublet = {}
doublet["name"] = "doublet"
doublet["title"] = "2D eq. $x = t^{-1}\\cos(t^2)$"
doublet["n"] = 10
doublet["sigma"] = 4
doublet["f"] = f
doublet["g"] = g
doublet["A_sym"] = Simplify(A_sym_doublet)
doublet["A_num"] = A_num_doublet

def doublet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(dg(t)*np.cos(f_num(t)) - df(t)*g_num(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1))) 
	return x_

doublet["w2"] = 0	# dummy value
doublet["true_sol"] = doublet_sol
doublet["ylim"] = (-1.0, 1.0)
doublet["t_start"] = 2
doublet["t_stop"] = 15
doublet["x0"] = doublet_sol(np.array([doublet["t_start"]]))
# ---------------------------- #
# -- PhotonBaryon equation stuff -- #

alpha = 0.05 #6371

eta_0 = 0.1
k = 110 #10

Og0 = 0.5 
Ob0 = 0.5

def Apb(k, Og0, Ob0):
	A = sym.Matrix(
		[[-2*Og0/ts, -k, -Ob0/2, 0, -ts*(k**2)/3 - 1/ts], 
		 [k/3, alpha*Ob0/(ts**2), 0, -sym.I*alpha*Ob0/(3*ts**2), k/3],
		 [-6*Og0/ts, 0, -3*Ob0/2, -sym.I*k, -ts*k**2 - 3/ts],
		 [0, 4*sym.I*alpha*Og0/(ts**3), 0, 4*alpha*Og0/(3*ts**3) - 1/(ts**2), -sym.I*k],
		[-2*Og0/ts, 0, -Ob0/2, 0, -ts*(k**2)/3 - 1/ts]])
	return A
	
A_sym_pb = Apb(k, Og0, Ob0)
	
A_num_pb = sym.lambdify((ts), A_sym_pb, modules=array2mat_c)
	
PhotonBaryon = {}
PhotonBaryon["name"] = "PhotonBaryon"
PhotonBaryon["title"] = "Tightly coupled photon baryon system"
PhotonBaryon["A_sym"] = A_sym_pb
PhotonBaryon["A_num"] = A_num_pb
PhotonBaryon["x0"] = np.array([1, 0, 1, 0, 1])

def rs(t):
	Rs = (8*Og0/(3*np.sqrt(3)*Ob0))*(np.sqrt(1 + (3*Ob0)*t/(4*Og0)) - 1)
	return Rs
	
def drs(t):
	# derivative of r_s
	dRs = (1/np.sqrt(3))*1/np.sqrt(1 + (3*Ob0)*t/(4*Og0))
	return dRs
	
def C_sol(t):
	C = np.cos(k*rs(t))
	return C
	
def S_sol(t):
	S = np.sin(k*rs(t))
	return S
	
def Om0_sol(t, t0):
	# define the first row of the solution matrix M_true(t, t0) corresponding to
	# Θ_0 
	line1 = np.array([S_sol(t0)*S_sol(t) - C_sol(t)*C_sol(t0) 
			- ((C_sol(t)*S_sol(t0) - S_sol(t)*C_sol(t0))/(drs(t0)*k))*(2*Og0/t0),
			- ((C_sol(t)*S_sol(t0) - S_sol(t)*C_sol(t0))/drs(t0)), 
			-((C_sol(t)*S_sol(t0) - S_sol(t)*C_sol(t0))/(drs(t0)*k))*(Ob0/2),
	0, ((C_sol(t)*S_sol(t0) - S_sol(t)*C_sol(t0))/(drs(t0)*k))*(t0*(k**2)/3 + 1/t0)])
	return line1
	
def PhotonBaryon_true_sol_single(t):
	# define the "true" solution (only true for Θ_0 so far)
	line1 = Om0_sol(t, eta_0).reshape(1, 5)
	Id = np.identity(5)
	M = Id
	M[0,:] = line1
	Result = M @ PhotonBaryon["x0"].reshape((5, 1))
	return Result.reshape(5)
	
#def PhotonBaryon_true_sol(t):
	output = np.zeros((t.size, 5))
	for i in range(t.size):
		output[i, :] = PhotonBaryon_true_sol_single(t[i])
	return output

def PhotonBaryon_true_sol(t):
	output = np.zeros((t.size, 5))
	for i in range(t.size):
		output[i, 0] = C_sol(t[i])
	return output

PhotonBaryon["true_sol"] = PhotonBaryon_true_sol
PhotonBaryon["t_start"] = eta_0
PhotonBaryon["t_stop"] = 0.8
PhotonBaryon["ylim"] = (-2, 2)
# ---------------------------- #


################### Choose equation #########################

Eq = PhotonBaryon

############# define some functions ##########

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA

def Com(A, B):
	return (A*B - B*A)
	
#------> set up alpha functions

def alpha_D(t0, t, A, order=4):
	# compute the alpha coefficients using the autograd
	# derivative 
	h = t - t0
	dt = 0.000001*h
	a_1 = h*A(t0 + 0.5*h)
	dA = eg(A, dt)
	a_2 = (h**2)*dA(t0 + 0.5*h)
	if order == 4:
		return (a_1, a_2)
	elif order == 6:
		ddA = eg(dA, dt)
		a_3 = (1/2)*(h**3)*ddA(t0 + 0.5*h)
		return (a_1, a_2, a_3)
	
def alpha_GL(t0, t, A, order=4):
	# compute the alpha coefficients using the Gauss-Legendre quadrature
	# rule
	h = t - t0
	if order == 4:
		A1 = A(t0 + (0.5 - np.sqrt(3)/6)*h)
		A2 = A(t0 + (0.5 + np.sqrt(3)/6)*h)
		a_1 = 0.5*h*(A1 + A2)
		a_2 = (np.sqrt(3)/12)*h*(A2 - A1)
		return (a_1, a_2)
	elif order == 6:
		A1 = A(t0 + (0.5 - 0.1*np.sqrt(15))*h)
		A2 = A(t0 + 0.5*h)
		A3 = A(t0 + (0.5 + 0.1*np.sqrt(15))*h)
		a_1 = h*A2
		a_2 = (np.sqrt(15)/3)*h*(A3 - A1)
		a_3 = (10/3)*h*(A3 - 2*A2 + A1)
		return (a_1, a_2, a_3)
		
def alpha_SNC(t0, t, A, order=4):
	# compute the alpha coefficients using the Simpson and Newton–
	# Cotes quadrature rules using equidistant A(t) points
	h = t - t0
	if order == 4:
		A1 = A(t0)
		A2 = A(t0 + 0.5*h)
		A3 = A(t0 + h)
		a_1 = (h/6)*(A1 + 4*A2 + A3)
		a_2 = h*(A3 - A1)
		return (a_1, a_2)
	elif order == 6:
		A1 = A(t0)
		A2 = A(t0 + 0.25*h)
		A3 = A(t0 + 0.5*h)
		A4 = A(t0 + 0.75*h)
		A5 = A(t0 + h)
		a_1 = (1/60)*(-7*(A1 + A5) + 28*(A2 + A4) + 18*A3)
		a_2 = (1/15)*(7*(A5 - A1) + 16*(A4 - A2))
		a_3 = (1/3)*(7*(A1 + A5) - 4*(A2 + A4) - 6*A3)
		return (a_1, a_2, a_3)

#------> set up quadrature integragrators  

scipy_quad_maxiter=200
	
def scipy_c_quad(f, t0, t, ARGS=()):
	# integrate complex valued function f(t) from t0 to t using scipy.integrate.quadrature
	MAXITER=scipy_quad_maxiter
	def f_real(x, *args):
		f_ = f(x, *args)
		return np.real(f_)
	def f_imag(x, *args):
		f_ = f(x, *args)
		return np.imag(f_)
	Int_real = quad(f_real, t0, t, args=ARGS, maxiter=MAXITER, vec_func=False)[0]
	Int_imag = 1j*quad(f_imag, t0, t, args=ARGS, maxiter=MAXITER, vec_func=False)[0]
	Int_ = Int_real + Int_imag
	return Int_
	
def scipy_M_quad(A, t0, t, ARGS=()):
	# integrate complex matrix valued function f(t) from t0 to t using scipy.integrate.quadrature
	MAXITER=scipy_quad_maxiter
	ni, nj = A(1).shape
	def f_real(x, I, J, *args):
		f_ = A(x, *args)[I, J]
		return np.real(f_)
	def f_imag(x, I, J, *args):
		f_ = A(x, *args)[I, J]
		return np.imag(f_)
	Int_M = np.zeros((ni, nj))*(1.0+0.j)
	for I in range(ni):
		for J in range(nj):
			IJ_ARGS = (I, J) + ARGS
			Int_M[I, J] = quad(f_real, t0, t, args=IJ_ARGS, maxiter=MAXITER, vec_func=False)[0] + 1j*quad(f_imag, t0, t, args=IJ_ARGS, maxiter=MAXITER, vec_func=False)[0]
	return Int_M
	
# set quadrature integrator (for the moment just set one)
c_quad = scipy_c_quad
	
#----> other functions

def Omega_num(A, alpha, order):
	# function to return an Omega(t0, t) function
	def Omega(t0, t):
		# the Magnus expansion Omega truncated to the appropriate order in h
		if order == 4:
			a_1, a_2 = alpha(t0, t, A, 4)
			Om = a_1 - (1/12)*Com(a_1, a_2)
			return Om
		elif order == 6:
			a_1, a_2, a_3 = alpha(t0, t, A, 6)
			C1 = Com(a_1, a_2)
			C2 = -(1/60)*Com(a_1, 2*a_3 + C1)
			Om = a_1 + (1/12)*a_3 + (1/240)*Com(-20*a_1-a_3+C1, a_2+C2)
			return Om
	return Omega

def ferr(x_0, x_l):
	# a function to evaluate an error between step estimates
	# returns a vector
	err = np.abs(x_0 - x_l)
	return err
	
def log_minor_ticks(ax):
	locmin = ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10)) 
	ax.yaxis.set_minor_locator(locmin)
	ax.yaxis.set_minor_formatter(ticker.NullFormatter())

###################### Symbolics #########################
#
#	Symbolic manipulation using sympy

A_sym = Eq["A_sym"]

print("A = ", A_sym)
print()

A_num = Eq["A_num"]

print(A_num(0.25))
print()
	
"""
define the first and second terms of the Magnus expansion (symbolic form)

Ω_1(t) = \int_t_0^t ( A(t') ) dt'

Ω_2(t) = 0.5 \int_t_0^t( \int_t_0^t'( [A(t'),A(t'')] )dt'' )dt'

"""

made_Om_1 = False

def Omega_1_sym(A):
	integral = sym.integrate(A.subs(ts, ts1), (ts1, ts0, ts))
	return integral
	
def Omega_2_sym(A):
	ts2 = sym.Symbol('ts2')
	integral_1 = sym.integrate(Com(A.subs(ts, ts1),A.subs(ts, ts2)), (ts2, ts0, ts1))
	print("integral_1 = ", integral_1)
	print()
	integral_2 = sym.integrate(integral_1, (ts1, ts0, ts))
	return 0.5*integral_2
	
def Magnus1(alpha):
	def Make_func():
		if alpha == "analytic":
			global made_Om_1
			if not made_Om_1:
				Om_1 = Omega_1_sym(A_sym)
				print("Omega 1 = ", sym.nsimplify(Om_1))
				print()
				global Omega_1_exact
				Omega_1_exact = sym.lambdify((ts0, ts), Om_1, modules=array2mat)
				made_Om_1 = True
			Omega = Omega_1_exact
		elif alpha != "analytic":
			Omega = Omega_num(A_num, alpha, 4)
		def Mf(t0, t):
			Om = Omega(t0, t)
			return linalg.expm(Om)
		return Mf
	return Make_func
	
def Magnus2(alpha):
	def Make_func():
		if alpha == "analytic":
			global made_Om_1
			if not made_Om_1:
				Om_1 = Omega_1_sym(A_sym)
				print("Omega 1 = ", sym.nsimplify(Om_1))
				print()
				global Omega_1_exact
				Omega_1_exact = sym.lambdify((ts0, ts), Om_1, modules=array2mat)
				made_Om_1 = True
			Om_2 = Omega_2_sym(A_sym)
			print("Omega 2 = ", sym.nsimplify(Om_2))
			print()
			Omega_2_exact = sym.lambdify((ts0, ts), Om_2, modules=array2mat)
			def Omega(t0, t):
				Om = Omega_1_exact(t0, t) + Omega_2_exact(t0, t)
				return Om
		elif alpha != "analytic":
			Omega = Omega_num(A_num, alpha, 6)
		def Mf(t0, t):
			Om = Omega(t0, t)
			return linalg.expm(Om)
		return Mf
	return Make_func
	
def Cayley(alpha, order):
	# Caley method
	def Make_func():
		if alpha == "analytic":
			# only a order 4 method available
			A = A_sym.subs(ts, ts0)
			Ndim = A.shape[0]
			Om = Omega_1_sym(A_sym) + Omega_2_sym(A_sym)
			Id = sym.eye(Ndim)
			C_ = Om*(Id - (1/12)*(Om**2)*(Id - (1/10)*(Om**2)))
			M_sym = (Id - (1/2)*C_).inv()*(Id + (1/2)*C_)
			print("4th order Cayley matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
			return Mf
		elif alpha != "analytic":
			# order 4 or order 6 methods available
			Omega = Omega_num(A_num, alpha, order)
			Ndim = Eq["x0"].size
			Id = np.identity(Ndim)
			def Mf(t0, t):
				Om = Omega(t0, t)
				if order == 4:
					C_ = Om*(Id - (1/12)*(Om**2))
				elif order ==6:
					C_ = Om*(Id - (1/12)*(Om**2)*(1 - (1/10)*(Om**2)))
				M_ = np.linalg.inv(Id - 0.5*C_)*(Id + 0.5*C_)
			return Mf
	return Make_func
	
def w1_func(t):
	return sym.sqrt(Eq["w2"](t))

def WKB_analytic():
	xA = sym.cos(sym.integrate(w1_func(ts1), (ts1, ts0, ts)))/sym.sqrt(w1_func(ts))
	xB = sym.sin(sym.integrate(w1_func(ts1), (ts1, ts0, ts)))/sym.sqrt(w1_func(ts))
	dxA = sym.diff(xA, ts)
	dxB = sym.diff(xB, ts)
	x_mat = sym.Matrix([[xA, xB], [dxA, dxB]])
	x_mat_0 = x_mat.subs(ts, ts0)
	M_sym = x_mat*x_mat_0.inv()
	print("WKB matrix = ", M_sym)
	print()
	Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
	return Mf

### new methods

def Jordan_WKB(Use_numerics):
	Use_Aprime2_or_J = False
	def Make_func():
		# symbolics
		A = A_sym.subs(ts, ts0)
		Aprime = sym.diff(A, ts0) + A*A
		Ndim = A.shape[0]
		P_0, J_0 = Aprime.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
		if Use_numerics == 0 or Use_numerics == 1:
			J = sym.simplify(J_0)
			P = sym.simplify(P_0)
			print("JWKB:")
			print("J = ", J)
			print()
			print("P = ", P)
			print()
			Pinv = P.inv()
			print("Pinv = ", Pinv)
			print()
			dPinv = sym.diff(Pinv, ts0)
			print("dPinv = ", dPinv)
			print()
			if Use_Aprime2_or_J:
				ddPinv = sym.diff(dPinv, ts0)
				print("ddPinv = ", ddPinv)
				print()
				Aprime2 = ddPinv*P + 2*dPinv*A*P + J 
				print("A'' = ", Aprime2)
				print()
				W2 = -Aprime2
			elif not Use_Aprime2_or_J:
				W2 = -J
			w1_sym = []
			for i in range(0, Ndim):
				w2 = W2[i,i]
				print("w2 = ", w2)
				w1 = sym.sqrt(w2)
				w1_sym.append(w1)
			if Use_numerics == 0:
				# symbolic version
				M11 = sym.eye(Ndim)
				M12 = sym.eye(Ndim)
				for i in range(0, Ndim):
					w1 = w1_sym[i]
					C = sym.cos(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
					S = sym.sin(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
					dw1 = sym.diff(w1, ts0)
					M11[i,i] = C + S*dw1/(2*w1**2)
					M12[i,i] = S/w1
				M_sym = (P.subs(ts0, ts))*(M11*Pinv + M12*(dPinv + Pinv*A))
				print()
				print("Jordan_WKB matrix = ", M_sym)
				print()
				Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
			elif Use_numerics == 1:
				# semi-numerical version
				A_num = Eq["A_num"]
				P_num = sym.lambdify((ts0), P, modules=array2mat_c)
				Pinv_num = sym.lambdify((ts0), Pinv, modules=array2mat_c)
				dPinv_num = sym.lambdify((ts0), dPinv, modules=array2mat_c)
			
				if Use_Aprime2_or_J:
					Aprime2_num = sym.lambdify((ts0), Aprime2, modules=array2mat_c)
				elif not Use_Aprime2_or_J:
					J_num = sym.lambdify((ts0), J, modules=array2mat_c)
				Id = np.identity(Ndim)
				M11 = Id.astype(np.complex64)
				M12 = Id.astype(np.complex64)
				w1_num = []
				dw1_num = []
				# convert symbolic form into numerical functions
				for i in range(0, Ndim):
					w1_num.append(sym.lambdify((ts0), w1_sym[i], modules=array2mat_c))
					dw1_num.append(eg(w1_num[i], 0.00001))
				def Mf(t0, t):
					# define a function to compute the M matrix
					for i in range(Ndim):
						w1 = w1_num[i](t)
						w10 = w1_num[i](t0)
						dw10 = dw1_num[i](t0)
						Int_w1 = c_quad(w1_num[i], t0, t, ARGS=())
						C = np.cos(Int_w1)*csqrt(w10/w1)
						S = np.sin(Int_w1)*csqrt(w10/w1)
						M11[i,i] = C + S*dw10/(2*(w10)**2)
						M12[i,i] = S/w10
					M_ = P_num(t) @ (M11 @ Pinv_num(t0) + M12 @ (dPinv_num(t0) + Pinv_num(t0) @ A_num(t0)))
					return M_
		elif Use_numerics == 2:
			# version minimising the amount of symbolic manipulation required
			J = J_0
			P = P_0
			print("JWKB:")
			print("J = ", J)
			print()
			print("P = ", P)
			print()
			#Pinv = P.inv()
			#print("Pinv = ", Pinv)
			#print()
			P_num = sym.lambdify((ts0), P, modules=array2mat_c)
			def Pinv_num(t):
				Pt = P_num(t)
				Pinvt = np.linalg.inv(Pt)
				return Pinvt
			J_num = sym.lambdify((ts0), J, modules=array2mat_c)
			dPinv_num = eg(Pinv_num, 0.00001)
			if Use_Aprime2_or_J:
				ddPinv_num = eg(dP_num, 0.00001)
				A_num = Eq["A_num"]
				def Aprime2_num(t):
					ddPinvt = ddPinv_num(t)
					Pt = P_num(t)
					Pinvt = np.linalg.inv(Pt)
					At = A_num(t)
					Jt = J_num(t)
					Aprime2t = ddPinvt @ Pt + 2*dPinvt @ At @ Pt + Jt
					return Aprim2t
				negW2 = Aprime2_num
			elif not Use_Aprime2_or_J:
				negW2 = J_num
			def w1_num(t, n):
				return csqrt(-negW2(t)[n,n])
			def w1_vec(t):
				w1 = np.ones(Ndim)
				W2 = - -negW2(t)
				for i in range(0, Ndim):
					w1[i] = csqrt(W2[i, i])
				return w1
			dw1 = eg(w1_vec, 0.00001)
			def Mf(t0, t):
				# define a function to compute the M matrix
				w1 = w1_vec(t)
				w10 = w1_vec(t0)
				dw10 = dw1(t0)
				for i in range(Ndim):
					Int_w1 = c_quad(w1_sing, t0, t, ARGS=(i))
					C = np.cos(Int_w1)*csqrt(w10[i]/w1[i])
					S = np.sin(Int_w1)*csqrt(w10[i]/w1[i])
					M11[i,i] = C + S*dw10[i]/(2*(w10[i])**2)
					M12[i,i] = S/w10[i] 
					Pinvt0 = dPinv_num(t0)
				M_ = P_num(t) @ (M11 @ Pinvt0 + M12 @ () + Pinvt0 @ A_num(t0))
				return M_
		return Mf
	return Make_func

def Pseudo_WKB(Use_numerics):
	# Pseudo-WKB method
	def Make_func():
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		Aprime = sym.diff(A, ts0) + A*A
		print("A' = ", Aprime)
		print()
		w1_sym = []
		for i in range(0, Ndim):
			w2 = -Aprime[i,i]
			print("w2 = ", w2)
			w1 = sym.sqrt(w2)
			w1_sym.append(w1)
		if Use_numerics == 0:
			# symbolic version
			M11 = sym.eye(Ndim)
			M12 = sym.eye(Ndim)
			for i in range(0, Ndim):
				w1 = w1_sym[i]
				C = sym.cos(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
				S = sym.sin(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
				dw1 = sym.diff(w1, ts0)
				M11[i,i] = C + S*dw1/(2*w1**2)
				M12[i,i] = S/w1
			M_sym = M11 + M12*A
			print()
			print("Pseudo-WKB matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
		elif Use_numerics == 1:
			# numerical version 
			Ap = sym.lambdify((ts0), Aprime, modules=array2mat)
			Id = np.identity(Ndim)
			M11 = Id.astype(np.complex64)
			M12 = Id.astype(np.complex64)
			w1_num = []
			dw1_num = []
			# convert symbolic form into numerical functions
			for i in range(0, Ndim):
				w1_num.append(sym.lambdify((ts0), w1_sym[i], modules=array2mat_c))
				dw1_num.append(eg(w1_num[i], 0.00001))
			def Mf(t0, t):
				# define a function to compute the M matrix
				for i in range(Ndim):	
					w1 = w1_num[i](t)
					w10 = w1_num[i](t0)
					dw10 = dw1_num[i](t0)
					Int_w1 = c_quad(w1_num[i], t0, t, ARGS=())
					C = np.cos(Int_w1)*csqrt(w10/w1)
					S = np.sin(Int_w1)*csqrt(w10/w1)
					M11[i,i] = C + S*dw10/(2*(w10)**2)
					M12[i,i] = S/w10
				M_ = (M11 + M12 @ A_num(t0))
				return M_
		return Mf
	return Make_func

def Jordan_Magnus(Lambda_only, Use_numerics):
	def Make_func():
		# symbolics
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		P_, J_ = A.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
		P = sym.simplify(P_)
		J = sym.simplify(J_)
		print("J = ", J)
		print()
		print("P = ", P)
		print()
		Pinv = P.inv()
		LK_ = J + sym.diff(Pinv, ts0)*P
		LK = sym.simplify(LK_)
		print("LK = ", LK)
		print()
		if Lambda_only:
			# only use the diagonal elements
			LK = sym.eye(Ndim).multiply_elementwise(LK)
			print("L = ", LK)
			print()
		if Use_numerics == 0:
			Om1 = sym.integrate(LK.subs(ts0, ts1), (ts1, ts0, ts)) 
			print("Ω1 = ", Om1)
			print()
			Om1_num = sym.lambdify((ts0, ts), Om1, modules=array2mat_c)
			#JM1["name"] = JM1["name"] + " (analytic)"
		elif Use_numerics == 1:
			LK_num = sym.lambdify((ts0), LK, modules=array2mat_c)
			#
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""
			Om1_num = Omega_num(LK_num, alpha_GL, 4)
		P_num = sym.lambdify((ts0), P, modules=array2mat_c)
		#
		def Mf(t0, t):
			M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ np.linalg.inv(P_num(t0))
			return M_.astype(np.float64)
		return Mf
	return Make_func
	
def Ext_Pseudo_WKB(Use_numerics):
	# Extended Pseudo-WKB method
	def Make_func():
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		Id_sym = sym.eye(Ndim)
		Aprime = sym.diff(A, ts0) + A*A
		print("A' = ", Aprime)
		print()
		Ainv = A.inv()
		w1d_sym = []
		gamma_sym = []
		for i in range(0, Ndim):
			# extract diagonal elements of various matrices
			Ap_ = Aprime[i, i]
			A_ = A[i, i]
			Ainv_  = Ainv[i, i]
			ApAinv_ = (Aprime @ Ainv)[i, i]
			# 
			w2 = (ApAinv_*A_ - Ap_)/(1 - A_*Ainv_)
			gamma = (Ainv_*Ap_ - ApAinv_)/(1 - A_*Ainv_)
			#w1 = sym.sqrt(w2)
			print("w2 = ", w2)
			print("gamma = ", gamma)
			w1d = sym.sqrt(w2 - (gamma**2)/4)
			w1d_sym.append(w1d)
			gamma_sym.append(gamma)	
		if Use_numerics == 0:
			# symbolic version
			M11 = sym.eye(Ndim)
			M12 = sym.eye(Ndim)
			for i in range(0, Ndim):
				w1d = w1d_sym[i]
				gamma = gamma_sym[i]
				Int_gamma = sym.integrate(gamma.subs(ts0, ts1), (ts1, ts0, ts))
				C = sym.exp(-(1/2)*Int_gamma)*sym.cos(sym.integrate(w1d.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1d/w1d.subs(ts0, ts))
				S = sym.exp(-(1/2)*Int_gamma)*sym.sin(sym.integrate(w1d.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1d/w1d.subs(ts0, ts))
				dw1d = sym.diff(w1d, ts0)
				M11[i,i] = C + S*(gamma/2*w1d + dw1d/(2*w1d**2))
				M12[i,i] = S/w1d
			M_sym = M11 + M12*A
			print()
			print("Ext-Pseudo-WKB matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
		elif Use_numerics == 1:
			# numerical version 
			Id = np.identity(Ndim)
			M11 = Id.astype(np.complex64)
			M12 = Id.astype(np.complex64)
			w1d_num = []
			gamma_num = []
			dw1d_num = []
			# convert symbolic forms into numerical functions
			for i in range(0, Ndim):
				w1d_num.append(sym.lambdify((ts0), w1d_sym[i], modules=array2mat_c))
				gamma_num.append(sym.lambdify((ts0), gamma_sym[i], modules=array2mat_c))
				dw1d_num.append(eg(w1d_num[i], 0.00001))
			def Mf(t0, t):
				# define a function to compute the M matrix
				Int_w1d = Id
				for i in range(Ndim):
					w1d = w1d_num[i](t)
					w1d0 = w1d_num[i](t0)
					dw1d0 = dw1d_num[i](t0)
					g0 = gamma_num[i](t0)
					Int_gamma = c_quad(gamma_num[i], t0, t)
					Int_w1d = c_quad(w1d_num[i], t0, t)
					C = np.exp(-0.5*Int_gamma)*np.cos(Int_w1d)*csqrt(w1d0/w1d)
					S = np.exp(-0.5*Int_gamma)*np.sin(Int_w1d)*csqrt(w1d0/w1d)
					M11[i,i] = C + S*(g0/(2*w1d0) + dw1d0/(2*(w1d0)**2))
					M12[i,i] = S/w1d0
				M_ = (M11 + M12 @ A_num(t0))
				return M_
		return Mf
	return Make_func
	
def Modified_M1(Use_numerics, alpha):
	# modified Magnus expansion from Iserles 2002a 
	# "ON THE GLOBAL ERROR OF DISCRETIZATION METHODS ... "
	def Make_func():
		A = A_sym.subs(ts, ts0)
		h = sym.Symbol("h", nonzero=True)
		t_half = sym.Symbol("t_half")
		A_half = A.subs(ts0, t_half)
		"""
		def B(t):
			A_t = A.subs(ts0, t)
			B = sym.exp((ts0 - t)*A_half)*(A_t - A_half)*sym.exp((t - ts0)*A_half)
			return B
		"""
		B_ = sym.exp(-h*A_half)*(A.subs(ts0, ts1) - A_half)*sym.exp(h*A_half)
		B_ = sym.nsimplify(B_)
		B_ = B_.rewrite(sym.cos)
		B_ = sym.simplify(B_)
		print("B = ", B_)
		print()
		if Use_numerics == 0:
			Om = sym.integrate(B_, (ts1, ts0, ts))
			Om_ = Om.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
			print("Om = ", Om_)
			print()
			M_sym = sym.exp(h*A_half)*sym.exp(Om)
			M_sym_ = M_sym.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
			print("Modified Magnus 1 matrix = ", M_sym_)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym_, modules=array2mat_c)
		elif Use_numerics == 1:
			A_half_ = A_half.subs(t_half, (1/2)*(ts0 + ts)) 
			A_half_num = sym.lambdify((ts0, ts), A_half_, modules=array2mat)
			def B_num(t1, t0, t):
				A_t = Eq["A_num"](t1)
				A_h = A_half_num(t0, t)
				B = linalg.expm((t0 - t)*A_h) @ (A_t - A_h) @ linalg.expm((t - t0)*A_h)
				return B
			"""
			B_ = B(ts1)
			B_num = sym.lambdify((ts1, ts0, ts), B_, modules=array2mat_c)
			"""
			def Omega_B_num(t0, t):
				def A(t1):
					A_ = B_num(t1, t0, t)
					return A_
				a_1, a_2 = alpha(t0, t, A, 4)
				Om = a_1 - (1/12)*Com(a_1, a_2)
				return Om
			#	
			def Mf(t0, t):
				M_ = linalg.expm((t-t0)*A_half_num(t0, t)) @ linalg.expm(Omega_B_num(t0, t))
				return M_
		return Mf
	return Make_func

###### set up integrator dictionaries #########################

"""
maybe put some other settings in here to make using different integrators easier?
"""
RKF45 = {
	"name" : "RKF 4(5)",
	"fname" : "RKF45" 
}

M1 = {
	"name" : "Magnus with $\\Omega_1$, analytic func.",
	"fname" : "M1",
	"alpha" : "analytic",
	"order" : 2, 
	"Mfunc" : Magnus1("analytic")
}

M2 = {
	"name" : "Magnus with $\\Omega_1+\\Omega_2$, analytic func.",
	"fname" : "M2",
	"alpha" : "analytic",
	"order" : 4, 
	"Mfunc" : Magnus2("analytic")
}

M4_GL = {
	"name" : "Magnus 4$^\\circ$, GL quad",
	"fname" : "M4GL",
	"alpha" : alpha_GL,
	"order" : 4, 
	"Mfunc" : Magnus1(alpha_GL)
}

M4_D = {
	"name" : "Magnus 4$^\\circ$, num. diff",
	"fname" : "M4D",
	"alpha" : alpha_D,
	"order" : 4, 
	"Mfunc" : Magnus1(alpha_D)
}

M4_SNC = {
	"name" : "Magnus 4$^\\circ$, Simpson quad",
	"fname" : "M4SNC",
	"alpha" : alpha_SNC,
	"order" : 4,  
	"Mfunc" : Magnus1(alpha_SNC)
}

M6_D = {
	"name" : "Magnus 6$^\\circ$, num. diff",
	"fname" : "M6D",
	"alpha" : alpha_D,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_D)
}

M6_GL = {
	"name" : "Magnus 6$^\\circ$, GL quad",
	"fname" : "M6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_GL)
}

M6_SNC = {
	"name" : "Magnus 6$^\\circ$, NC quad",
	"fname" : "M6SNC",
	"alpha" : alpha_SNC,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_SNC)
}

WKB = {
	"name" : "WKB, analytic",
	"fname" : "WKB",
	"order" : 4,
	"Mfunc" : WKB_analytic
}

C4_GL = {
	"name" : "Cayley 4$^\\circ$, GL quad",
	"fname" : "C4GL",
	"alpha" : alpha_GL,
	"order" : 4, 
	"Mfunc" : Cayley(alpha_GL, 4)
}

C6_GL = {
	"name" : "Cayley 6$^\\circ$, GL quad",
	"fname" : "C6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"Mfunc" : Cayley(alpha_GL, 6)
}

JWKB = {
	"name" : "JWKB",
	"fname" : "JWKB",
	"order" : 4, 
	"analytic" : 2,
	"Use_numerics" : 0,
	"Mfunc" : Jordan_WKB(0) 
}

JWKBnum = {
	"name" : "JWKB",
	"fname" : "JWKBnum",
	"order" : 4, 
	"Use_numerics" : 1,
	"Mfunc" : Jordan_WKB(1)
}

PWKB = {
	"name" : "PWKB",
	"fname" : "PWKB",
	"order" : 4, 
	"Use_numerics" : 0, 
	"Mfunc" : Pseudo_WKB(0)
}

PWKBnum = {
	"name" : "PWKB",
	"fname" : "PWKB",
	"order" : 4, 
	"Use_numerics" : 1, 
	"Mfunc" : Pseudo_WKB(1)
}

JMl = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Jordan_Magnus(True, 0)
}

JMlk = {
	"name" : "JM ($\\Lambda$ and $K$)",
	"fname" : "JM1lk",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Jordan_Magnus(False, 0)
}

JMlnum = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus(True, 1)
}

JMlknum = {
	"name" : "JM ($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus(False, 1)
}

EPWKB = {
	"name" : "EPWKB",
	"fname" : "EPWKB",
	"order" : 4,
	"Use_numerics" : 0,
	"Mfunc" : Ext_Pseudo_WKB(0)
}

EPWKBnum = {
	"name" : "EPWKB",
	"fname" : "EPWKBnum",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Ext_Pseudo_WKB(1)
}

MM1 = {
	"name" : "MM1",
	"fname" : "MM1",
	"alpha" : "analytic",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Modified_M1(0, "analytic")
}

MM1num = {
	"name" : "MM1",
	"fname" : "MM1",
	"alpha" : alpha_GL,
	"order" : 2,
	"Use_numerics" : 1, 
	"Mfunc" : Modified_M1(1, alpha_GL)
}

######################################

# choose the lines to plot (i.e. the integrators to use)
lines = [M4_GL]

############### set up Numerics #################

Use_RK = False

for line in lines:
	if line["fname"] != "RKF45":
		line["M"] = line["Mfunc"]()
	elif line["fname"] == "RKF45":
		Use_RK = True
		
# correct line labels
for M in [JWKBnum, PWKBnum, JMlnum, JMlknum, MM1num, EPWKBnum]:
	M["name"] = M["name"] + " (scipy quad, maxiter=" + str(scipy_quad_maxiter) + ")"

########## Integrator #################

# set error tolerance
epsilon	= 0.005
epsilon_RK = 0.005
rtol = 1		# rel. error tolerance for Magnus in units of ε
atol = 0.005	# abs. error tolerance for Magnus in units of ε
rtol_RK = 4		# rel. error tolerance for RKF4(5) in units of ε_RK
atol_RK = 2		# abs. error tolerance for RKF4(5) in units of ε_RK

def RKF45_Integrator(t_start, t_stop, h0, x0, A):
	# An integrator using a 4(5) RKF method
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	n_step = number of steps
	A = A(t) matrix function
	"""
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.1*h0 #(10**(-2))*2*h0
	h_max = 5*2*h0
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using 4th and 5th order RK methods
			k1 = np.dot(h*A(t),x_n)
			k2 = h*A(t + 0.25*h) @ (x_n + 0.25*k1)
			k3 = h*A(t + (3/8)*h) @ (x_n + (3/32)*k1 + (9/32)*k2)
			k4 = h*A(t + (12/13)*h) @ (x_n + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
			k5 = h*A(t + h) @ (x_n + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
			k6 = h*A(t + 0.5*h) @ (x_n - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
			y_np1 = x_n + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (11/40)*k5
			z_np1 = x_n + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
			#print("  A(t) = ", A(t))
			print("  t = ", t)
			print("z_np1[0] = ", z_np1[0])
			if np.isnan(z_np1[0]):
				print("NaN error")
				sysexit()
			if np.any(np.isinf(z_np1)) or np.any(np.isinf(y_np1)):
				print("y_np1 = ", y_np1)
				print("z_np1 = ", z_np1)
				print("Inf error")
				sysexit()
			#
			Err = np.abs(y_np1[0] - z_np1[0])
			print("Err = ", Err)
			#Err = ferr(y_np1, z_np1)
			"""
			Err_max = ε(rtol*|z_np1| + atol)
			"""
			Err_max = epsilon_RK*(rtol_RK*np.abs(z_np1) + atol_RK)
			Err_ratio = np.asscalar(np.mean(Err / Err_max))
			#
			if Err_ratio <= 1:
				h_new = h*S*np.power(Err_ratio, -1.0/5)
				#Delta = max(np.asscalar(max(Err)), epsilon_RK*0.1)
				#h_new = h*(epsilon_RK*h/Delta)**(1/4)
				if h_new > 10*h:	# limit how fast the step size can increase
					h_new = 10*h
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/4)
				#h_new = h*(epsilon_RK*h/np.asscalar(max(Err)))**(1/4)
				if h_new < 0.2*h:	# limit how fast the step size decreases
					h_new = 0.2*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
		t = t + h
		x_ = np.vstack((x_,z_np1.reshape(1, Ndim))) # add x_n+1 to the array of x values
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		if True: #np.round(((t-t_start)/(t_stop-t_start))*100000) % 1000 == 0:
			print("\r" + "integrated {:.1%}".format((t-t_start)/(t_stop-t_start)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)
	
def Magnus_Integrator(t_start, t_stop, h0, x0, Method):
	# An integrator for all non-RKF4(5) methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = (10**(-2))*2*h0
	h_max = 7.5*2*h0
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	#
	M = Method["M"]
	order = Method["order"]
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using one step of h & two steps of h/2
			#print("\r" + "trying step " + str(n) + " h=" + str(h) + " ...", end='')
			x_np1_0 = M(t, t+h) @ x_n
			x_np1_l = M(t+0.5*h, t+h) @ (M(t, t+0.5*h) @ x_n)
			# compute error
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
			#
			if Err_ratio <= 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/(order + 1)) # h*1.5
				if h_new > 10*h:	# limit how fast the step size can increase
					h_new = 10*h
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/(order))
				if h_new < 0.2*h:	# limit how fast the step size decreases
					h_new = 0.2*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
		t = t + h
		x_ = np.vstack((x_,x_np1_l.reshape(1, Ndim))) # add x_n+1 to the array of x values
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		if True: #np.round(((t-t_start)/(t_stop-t_start))*100000) % 1000 == 0:
			print("\r" + Method["fname"] + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)

###### plot graph #####################

def plot_graph():
	# function for plotting a graph of the results.
	h_unit = 0.025
	h0 = 0.5*h_unit
	
	MKR_size = 3	# marker size
	log_h = True
		
	######## Integration ##################
	
	t_start = Eq["t_start"]
	t_stop = Eq["t_stop"]
	
	for M in lines:
		if M["fname"] != "RKF45":
			M["data"] = Magnus_Integrator(t_start, t_stop, h0, Eq["x0"], M)
		elif M["fname"] == "RKF45":
			M["data"] = RKF45_Integrator(t_start, t_stop, h0, Eq["x0"], Eq["A_num"])

	t_vec0 = np.linspace(t_start, t_stop, 1000)
	x_true = Eq["true_sol"](t_vec0)
	
	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_adapt_" + linenames  
	
	"""
	filename = filename + "rtol=" + str(rtol) + "_atol=" + str(atol) + "_epsil=" + str(epsilon) 
	if Use_RK:
		filename = filename + "_rtolRK=" + str(rtol_RK) + "_atolRK=" + str(atol_RK) + "_epsilRK=" + str(epsilon_RK)
	filename = filename + "scipy_quad_MI=" + str(scipy_quad_maxiter)
	"""
	
	colours = ['c', 'r', 'm', 'g', 'r']
	markertypes = ['x', '+', 'o', '^', 'x']
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
	
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	ax0.plot(t_vec0, x_true[:,0], color="0.7", linewidth=1, linestyle="--", label="true soln.")
	#
	ax2 = plt.subplot(gs[2], sharex = ax0)
	ax2.plot(np.linspace(Eq["t_start"], t_stop, 20), np.log10(epsilon*np.ones(20)), color="k", linewidth=1, linestyle=":", label="$\epsilon$")
	#ax2.annotate("$\epsilon$", xy=(1.05*Eq["t_stop"], epsilon))
	#
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = (line["data"][1][:,0]).reshape(t.size, 1)
		x_true = (Eq["true_sol"](t)[:,0]).reshape(t.size, 1)
		error = np.log10(np.abs((x - x_true)/x_true))
		T = line["data"][2]
		ax0.plot(t, x, colours[i] + markertypes[i], markersize=MKR_size, linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		ax2.plot(t, error, colours[i] + '--', linewidth=1, alpha=1)
	ax0.set_ylabel("x")
	ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	ax0.set_xlim(Eq["t_start"], t_stop)
	ax2.set_xlim(Eq["t_start"], t_stop)
	ax2.set_ylabel("log$_{10}$(rel. error)")
	ax2.legend()
	ax2.set_xlabel("t")
	ax2.minorticks_on()
	ymin, ymax = ax2.get_ylim()
	if ymax>1:
		ax2.set_ylim(top=1)
	ax0.minorticks_on()
	ax0.set_title(Eq["title"] + " : comparing different methods", y=1.08)
	lgnd = ax0.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.25, 0.85, 0.50, 0.25), ncol = 2, shadow=False)
	for i in range(0, len(lines)+1):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)
	
	################ Stepsize plot
	# shared axis X
	ax1 = plt.subplot(gs[1], sharex = ax0)
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		h = t[1:] - t[:-1]
		t_av = 0.5*(t[1:] + t[:-1])
		if log_h:
			ax1.plot(t_av, np.log10(h), colours[i] + '-', linewidth=1, label="{:s}".format(line["name"]))
		elif log_h == False:
			ax1.plot(t_av, h, colours[i] + '-', linewidth=1, label="{:s}".format(line["name"]))
	if log_h:
		ax1.set_ylabel("log$_{10}$(h)")
		#savename = "Plots/" + filename + "_log_h.pdf"
	elif log_h == False:
		ax1.set_ylabel("h")
		#savename = "Plots/" + filename + ".pdf"
	savename = "Plots/" + filename + ".pdf"
	ax1.minorticks_on()
	ax1.set_xlim(Eq["t_start"], t_stop)
	# remove last tick label for the second subplot
	#plt.setp(ax1.get_yticklabels()[-2], visible=False) 
	#plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.subplots_adjust(hspace=.0)
	plt.savefig(savename, transparent=True)
	plt.clf()
	print("made plot")
	print("saved as " + savename)
	
#################################################################

plot_graph()

def get_times():
	# function for finding the times taken to integrate
	h_unit = 0.05
	h0 = 0.5*h_unit
		
	######## Integration ##################
	
	t_start = Eq["t_start"]
	t_stop = Eq["t_stop"]
	
	N_lines = len(lines)
	N_times = 20
	
	t_arr = np.zeros((N_times, N_lines))
	
	for I in range(0, N_times):
		for J in range(0, N_lines):
			M = lines[J]
			if M["fname"] != "RKF45":
				M["data"] = Magnus_Integrator(t_start, t_stop, h0, Eq["x0"], M)
			elif M["fname"] == "RKF45":
				M["data"] = RKF45_Integrator(t_start, t_stop, h0, Eq["x0"], Eq["A_num"])	
			t_arr[I, J] = M["data"][2]
		t_arr[I,:] = t_arr[I,:]/t_arr[I,0]
		print("### I = ", I)
	
	print(t_arr)
	
