# Magnus Solver comparison with WKB approximation

"""
To solve 

x''(t) = -ω^2(t) x(t)
"""

import numpy as np
import math as m
from numpy.lib.scimath import sqrt as csqrt

import time
import sympy as sym
from scipy import special, linalg

# choose numerical integrator
from scipy.integrate import quadrature as quad

from sys import exit as sysexit

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
	Ai0, Aip0, Bi0, Bip0 = special.airy(0)
	M = (1/(-Ai0*Bip0 + Aip0*Bi0))*np.matrix([[-Bip0, -Bi0], [+Aip0, Ai0]])
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
burst["ylim"] = (-0.5, 0.5)

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

burst["t_start"] = -0.5*burst["n"]
burst["t_stop"] = +0.5*burst["n"]
#burst["t_start"] = -2
#burst["t_stop"] = +2
burst["x0"] = Burst_sol(np.array([burst["t_start"]]))
burst["true_sol"] = Burst_sol
burst["title"] = "Burst equation (n = " + str(burst["n"]) + ")"
# ---------------------------- #
# -- Triplet equation stuff -- #
"""
def f(t):
	return (t*t)
	
def f_num(t):
	return (t*t)
	
def g(t):
	return (sym.exp(-t/4))
	
def g_num(t):
	return (np.exp(-t/4))

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
triplet["name"] = "triplet"
triplet["title"] = "3D equation, with $x = e^{-t/4}\\cos(t^2)$"
triplet["n"] = 10
triplet["sigma"] = 4
triplet["f"] = f
triplet["g"] = g
triplet["A_sym"] = A_sym_triplet
triplet["A_num"] = A_num_triplet

def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

triplet["w2"] = 0	# dummy value
triplet["true_sol"] = triplet_sol
triplet["t_start"] = 1
triplet["t_stop"] = 10
triplet["x0"] = triplet_sol(np.array([triplet["t_start"]]))
"""
# ---------------------------- #
# -- Second Triplet equation stuff -- #
def f(t):
	return (t**3)
	
def f_num(t):
	return (t**3)
	
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
print("ddG = ", ddG)
print("ddF = ", ddF)
"""

A_sym_triplet = sym.Matrix([[dG/G, -dF, 0], [+dF, 0, 1], [+(dF*dG)/G, ddG/G, 0]])
#A_sym_triplet = sym.Matrix([[0, -dF, 1], [dF, dG/G, 0], [ddG/G, -dF*dG/G, 0]])
	
A_num_triplet = sym.lambdify((ts), A_sym_triplet, modules=array2mat)
	
triplet2 = {}
triplet2["name"] = "triplet2"
triplet2["title"] = "3D equation, with $x = e^{-t/4}\\cos((2 + \\cos(t))t)$"
triplet2["n"] = 10
triplet2["sigma"] = 4
triplet2["f"] = f
triplet2["g"] = g
triplet2["A_sym"] = A_sym_triplet
triplet2["A_num"] = A_num_triplet

def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.cos(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

triplet2["w2"] = 0	# dummy value
triplet2["true_sol"] = triplet_sol
triplet2["t_start"] = 1
triplet2["t_stop"] = 10
triplet2["x0"] = triplet_sol(np.array([triplet2["t_start"]]))
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
doublet["A_sym"] = A_sym_doublet
doublet["A_num"] = A_num_doublet

def doublet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(dg(t)*np.cos(f_num(t)) - df(t)*g_num(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1))) 
	return x_

doublet["w2"] = 0	# dummy value
doublet["true_sol"] = doublet_sol
doublet["t_start"] = 2
doublet["t_stop"] = 15
doublet["x0"] = doublet_sol(np.array([doublet["t_start"]]))
# ---------------------------- #

############## define numerical functions #############################

def Com(A, B):
	return (A*B - B*A)

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA

def c_quad(f, t0, t, ARGS=(), MAXITER=50):
	# integrate complex valued function f(t) from t0 to t using quadrature
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
	
def M_quad(A, t0, t, ARGS=(), MAXITER=50):
	# integrate complex matrix valued function f(t) from t0 to t using quadrature
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
		
def Omega(A, alpha, order):
	# the Magnus expansion Omega truncated to the appropriate order in h
	if order == 4:
		def Omega_f(t0, t):
			a_1, a_2 = alpha(t0, t, A, 4)
			Om = a_1 - (1/12)*Com(a_1, a_2)
			return Om
		return Omega
	elif order == 6:
		def Omega_f(t0, t):
			a_1, a_2, a_3 = alpha(t0, t, A, 6)
			C1 = Com(a_1, a_2)
			C2 = -(1/60)*Com(a_1, 2*a_3 + C1)
			Om = a_1 + (1/12)*a_3 + (1/240)*Com(-20*a_1-a_3+C1, a_2+C2)
			return Om
		return Omega_f

################### Choose equation #########################

Eq = triplet2

################ set up integrators ##############################

M1 = {
	"name" : "Magnus sol. with only $\\Omega_1$",
	"fname" : "M1",
	"order" : 2
}

M2 = {
	"name" : "Magnus sol. with $\\Omega_1 + \\Omega_2$",
	"fname" : "M2",
	"order" : 4
}

WKB = {
	"name" : "WKB sol.",
	"fname" : "WKB",
	"order" : 4
}

JWKB = {
	"name" : "Jordan-WKB sol.",
	"fname" : "JWKB",
	"order" : 4, 
	"analytic" : False
}

PWKB = {
	"name" : "Pseudo-WKB sol.",
	"fname" : "PWKB",
	"order" : 4, 
	"analytic" : False
}

C4 = {
	"name" : "4th order Cayley-transform sol.",
	"fname" : "C4",
	"order" : 4
}

JM1 = {
	"name" : "Jordan-Magnus 1 sol.",
	"fname" : "JM1",
	"order" : 2,
	"analytic" : False,
	"lambda_only" : True
}

MM1 = {
	"name" : "Modified Magnus 1 sol.",
	"fname" : "MM1",
	"order" : 2,
	"analytic" : True
}

EPWKB = {
	"name" : "Extended Pseudo-WKB sol.",
	"fname" : "EPWKB",
	"order" : 4,
	"analytic" : False
}

############################################

# choose the lines to plot (i.e. the integrators to use)
lines = [JM1]

###################### Symbolics #########################
#
#	Symbolic manipulation using sympy

A_sym = Eq["A_sym"]

print("A = ", A_sym)
print()

A_num = Eq["A_num"]
	
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
	
def Magnus_1():
	global made_Om_1
	if not made_Om_1:
		Om_1 = Omega_1_sym(A_sym)
		print("Omega 1 = ", sym.nsimplify(Om_1))
		print()
		global Omega_1_exact
		Omega_1_exact = sym.lambdify((ts0, ts), Om_1, modules=array2mat)
		made_Om_1 = True
	def Mf(t0, t):
		return linalg.expm(Omega_1_exact(t0, t))
	return Mf
	
def Magnus_2():
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
	def Mf(t0, t):
		return linalg.expm(Omega_1_exact(t0, t) + Omega_2_exact(t0, t))
	return Mf
	
def Cayley():
	# fourth order Cayley method
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
	
def w1_func(t):
	return sym.sqrt(Eq["w2"](t))

def WKB_matrix_sym():
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

# set up quadrature integragrator
quad_maxiter=200

def Jordan_WKB():
	Use_symbolics = JWKB["analytic"]
	Use_Aprime2_or_J = False
	# symbolics
	A = A_sym.subs(ts, ts0)
	Aprime = sym.diff(A, ts0) + A*A
	Ndim = A.shape[0]
	P_0, J_0 = Aprime.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
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
	if Use_symbolics:
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
	elif not Use_symbolics:
		# numerical version
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
				Int_w1 = c_quad(w1_num[i], t0, t, ARGS=(), MAXITER=quad_maxiter)
				C = np.cos(Int_w1)*csqrt(w10/w1)
				S = np.sin(Int_w1)*csqrt(w10/w1)
				M11[i,i] = C + S*dw10/(2*(w10)**2)
				M12[i,i] = S/w10
			M_ = P_num(t) @ (M11 @ Pinv_num(t0) + M12 @ (dPinv_num(t0) + Pinv_num(t0) @ A_num(t0)))
			return M_
	return Mf

def Pseudo_WKB():
	Use_symbolics = PWKB["analytic"]
	#
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
	if Use_symbolics:
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
	elif not Use_symbolics:
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
				Int_w1 = c_quad(w1_num[i], t0, t, ARGS=(), MAXITER=quad_maxiter)
				C = np.cos(Int_w1)*csqrt(w10/w1)
				S = np.sin(Int_w1)*csqrt(w10/w1)
				M11[i,i] = C + S*dw10/(2*(w10)**2)
				M12[i,i] = S/w10
			M_ = (M11 + M12 @ A_num(t0))
			return M_
	return Mf

def Jordan_Magnus1():
	Use_symbolic = JM1["analytic"]
	Lambda_only = JM1["lambda_only"]
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
	if Use_symbolic:
		Om1 = sym.integrate(LK.subs(ts0, ts1), (ts1, ts0, ts)) 
		print("Ω1 = ", Om1)
		print()
		Om1_num = sym.lambdify((ts0, ts), Om1, modules=array2mat_c)
		#JM1["name"] = JM1["name"] + " (analytic)"
	elif not Use_symbolic:
		LK_num = sym.lambdify((ts0), LK, modules=array2mat_c)
		#
		def Omega_num(t0, t):
			Om = M_quad(LK_num, t0, t, MAXITER=quad_maxiter)
			return Om
		Om1_num = Omega_num
	P_num = sym.lambdify((ts0), P, modules=array2mat_c)
	#
	def Mf(t0, t):
		M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ np.linalg.inv(P_num(t0))
		return M_.astype(np.float64)
		
	#print(Mf(2, 1))
	#sysexit()
		
	return Mf
	
def Modified_M1():
	# modified Magnus expansion from Iserles 2002a 
	# "ON THE GLOBAL ERROR OF DISCRETIZATION METHODS ... "
	
	Use_symbolic = MM1["analytic"]
	
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
	if 	Use_symbolic:
		Om = sym.integrate(B_, (ts1, ts0, ts))
		Om_ = Om.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
		print("Om = ", Om_)
		print()
		M_sym = sym.exp(h*A_half)*sym.exp(Om)
		M_sym_ = M_sym.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
		print("Modified Magnus 1 matrix = ", M_sym_)
		print()
		Mf = sym.lambdify((ts0, ts), M_sym_, modules=array2mat_c)
	elif not Use_symbolic:
		A_half_num = sym.lambdify((ts0, ts), A_half, modules=array2mat)
		def B_num(t1, t0, t):
			A_t = Eq["A_num"](t1)
			A_h = A_half_num(t0, t)
			B = linalg.expm((t0 - t)*A_h) @ (A_t - A_h) @ linalg.expm((t - t0)*A_h)
			return B
		"""
		B_ = B(ts1)
		B_num = sym.lambdify((ts1, ts0, ts), B_, modules=array2mat_c)
		"""
		def Omega_num(t0, t):
			Om = M_quad(B_num, t0, t, ARGS=(t0, t), MAXITER=quad_maxiter)
			return Om
		def Mf(t0, t):
			M_ = linalg.expm((t-t0)*A_half_num(t0, t)) @ linalg.expm(Omega_num(t0, t))
			return M_
	return Mf

def Ext_Pseudo_WKB():
	Use_symbolics = EPWKB["analytic"]
	#
	A = A_sym.subs(ts, ts0)
	Ndim = A.shape[0]
	Id_sym = sym.eye(Ndim)
	Aprime = sym.diff(A, ts0) + A*A
	print("A' = ", Aprime)
	print()
	Ainv = A.inv()
	#Gamma = -Id_sym.multiply_elementwise(Aprime*Ainv)
	#Lambda = Id_sym.multiply_elementwise(Aprime + Gamma*A)
	w1_sym = []
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
		w1 = sym.sqrt(w2)
		print("w2 = ", w2)
		print("gamma = ", gamma)
		w1_sym.append(w1)
		gamma_sym.append(gamma)	
	if Use_symbolics:
		# symbolic version
		M11 = sym.eye(Ndim)
		M12 = sym.eye(Ndim)
		for i in range(0, Ndim):
			w1 = w1_sym[i]
			gamma = gamma_sym[i]
			Int_gamma = sym.integrate(gamma.subs(ts0, ts1), (ts1, ts0, ts))
			C = sym.exp(-(1/2)*Int_gamma)*sym.cos(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
			S = sym.exp(-(1/2)*Int_gamma)*sym.sin(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
			dw1 = sym.diff(w1, ts0)
			M11[i,i] = C + S*(gamma/2*w1 + dw1/(2*w1**2))
			M12[i,i] = S/w1
		M_sym = M11 + M12*A
		print()
		print("Ext-Pseudo-WKB matrix = ", M_sym)
		print()
		Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
	elif not Use_symbolics:
		# numerical version 
		Id = np.identity(Ndim)
		M11 = Id.astype(np.complex64)
		M12 = Id.astype(np.complex64)
		w1_num = []
		gamma_num = []
		dw1_num = []
		# convert symbolic forms into numerical functions
		for i in range(0, Ndim):
			w1_num.append(sym.lambdify((ts0), w1_sym[i], modules=array2mat_c))
			gamma_num.append(sym.lambdify((ts0), gamma_sym[i], modules=array2mat_c))
			dw1_num.append(eg(w1_num[i], 0.00001))
		def Mf(t0, t):
			# define a function to compute the M matrix
			Int_w1 = Id
			for i in range(Ndim):
				w1 = w1_num[i](t)
				w10 = w1_num[i](t0)
				dw10 = dw1_num[i](t0)
				g0 = gamma_num[i](t0)
				Int_gamma = quad(gamma_num[i], t0, t, args=(), maxiter=quad_maxiter, vec_func=False)[0]
				Int_w1 = c_quad(w1_num[i], t0, t, ARGS=(), MAXITER=quad_maxiter)
				C = np.exp(-0.5*Int_gamma)*np.cos(Int_w1)*csqrt(w10/w1)
				S = np.exp(-0.5*Int_gamma)*np.sin(Int_w1)*csqrt(w10/w1)
				M11[i,i] = C + S*(g0/(2*w10) + dw10/(2*(w10)**2))
				M12[i,i] = S/w10
			M_ = (M11 + M12 @ A_num(t0))
			return M_
	return Mf

########### Asign Integrator M functions ############

M1["Mfunc"] 	= Magnus_1
M2["Mfunc"] 	= Magnus_2
WKB["Mfunc"]	= WKB_matrix_sym
C4["Mfunc"] 	= Cayley

JWKB["Mfunc"]	= Jordan_WKB
PWKB["Mfunc"] 	= Pseudo_WKB
JM1["Mfunc"] 	= Jordan_Magnus1
MM1["Mfunc"] 	= Modified_M1
EPWKB["Mfunc"] = Ext_Pseudo_WKB

# correct file names & line labels

for M in [JWKB, PWKB, JM1, MM1]:
	if not M["analytic"]:
		M["fname"] = M["fname"] + "num"
		M["name"] = M["name"] + " (numeric GL quad, maxiter=" + str(quad_maxiter) + ")"
	elif M["analytic"]:
		pass
	
if JM1["lambda_only"]:
	JM1["name"] = JM1["name"] + "(with $\\Lambda$ only)"
	JM1["fname"] = JM1["fname"] + "l"

############### set up Numerics #################

for line in lines:
	line["M"] = line["Mfunc"]()
		
########## Integration #################

t_start = Eq["t_start"]
t_stop = Eq["t_stop"]

"""
if Eq == Airy:
	t_start0 = t_start
	t_stop0 = t_stop
else:
	t_start0 = -10
	t_stop0 = 10
"""

t0 = np.array([t_start])
x0 = Eq["true_sol"](t0)
x0 = x0.reshape(x0.size, 1)
print("x0 = ", x0) 

n_steps = 250
t_vec = np.linspace(t_start, t_stop, n_steps)
t_vec0 = np.linspace(t_start, t_stop, 1000)
	
def Integrator_1(t_vec, x0, M):
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	Ndim = x0.size
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(Ndim)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		M_ = M(t0, t)
		x[n,:] = (M_ @ x0).reshape(Ndim)
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return x
	
######### plot graph ##################

for data in lines:
	print(data["fname"] + ": ")
	data["x"] = Integrator_1(t_vec, x0, data["M"])[:,0]

linenames = ""
for data in lines:
	linenames = linenames + data["fname"] + "_"

filename = "Analytic_test_" + Eq["name"] + "_" + linenames #"_t_start=" + str(t_start) + "_t_stop=" + str(t_stop) 
title = Eq["title"] + " : Testing integrators over one step"

colours = ['c', 'm', 'g', 'b', 'r']
lines_or_points = True
if lines_or_points:
	linetypes = ['-', '-', '-', '-', '-']
elif not lines_or_points:
	linetypes = ['x', '+', '^', 'x', '+']
x_true = Eq["true_sol"](t_vec0)[:,0]

for i in range(len(lines)):
	M = lines[i]
	plt.plot(t_vec, M["x"], colours[i]+linetypes[i], markersize=4, linewidth=1, label=M["name"])

plt.plot(t_vec0, x_true, color="0", linewidth=1, linestyle="--", label="true soln.")

plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-0.75, 0.75)
plt.title(title)
plt.minorticks_on()
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()
print("made plot")
print("saved plot as " + "Plots/" + filename + ".pdf")
