# Magnus Solver comparison with WKB approximation

"""
To solve 

x''(t) = -ω^2(t) x(t)
"""

import numpy as np
from numpy.lib.scimath import sqrt as csqrt

import time
import sympy as sym
from scipy import special, linalg
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

ts0 = sym.Symbol('ts0')
ts = sym.Symbol('ts')
ts1 = sym.Symbol('ts1')

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']
array2mat_c = [{'ImmutableDenseMatrix': np.matrix}, {'sqrt': csqrt}, 'numpy']

# --- Airy equation stuff --- #
Airy = {}
Airy["name"] = "Airy"
Airy["t_start"] = 2
Airy["t_stop"] = 35
Ai0, Aip0, Bi0, Bip0 = special.airy(-Airy["t_start"])
Airy["x0"] = np.array([Ai0, -Aip0])
Airy["ylim"] = (-0.75, 0.75)

def w2_Airy(t):
	return t
Airy["w2"] = w2_Airy
Airy["A_num"] = A_from_w2(w2_Airy, True)
Airy["A_sym"] = A_from_w2(w2_Airy, False)

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
burst["name"] = "Burst_n=" + str(burst["n"])
burst["ylim"] = (-0.5, 0.5)

def w2_burst(t):
	n = burst["n"]
	w = (n**2 - 1)/(1 + t**2)**2
	return w
burst["w2"] = w2_burst
burst["A_num"] = A_from_w2(w2_burst, True)
burst["A_sym"] = A_from_w2(w2_burst, False)

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
burst["x0"] = Burst_sol(np.array([burst["t_start"]]))
burst["true_sol"] = Burst_sol
burst["title"] = "Burst equation (n = " + str(burst["n"]) + ")"
# ---------------------------- #
# -- Triplet equation stuff -- #
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

def A_sym_triplet(t):
	F = f(t)
	G = g(t)
	dF = sym.diff(F, t)
	ddF = sym.diff(dF, t)
	dG = sym.diff(G, t)
	ddG = sym.diff(dG, t)
	#A = sym.Matrix([[dG/G, -dF, 0], [+dF, 0, 1], [+dF*dG/G, ddG/G, 0]])
	A = sym.Matrix([[0, 1, 0], [(ddG/G - dF**2), 0, ddF - 2*dG/G], [dF, 0, dG/G]])
	return A
	
A_num_triplet = sym.lambdify((ts), A_sym_triplet(ts), modules=array2mat)
	
triplet = {}
triplet["name"] = "3Deq"
triplet["title"] = "Three variable equation"
triplet["n"] = 10
triplet["sigma"] = 4
triplet["f"] = f
triplet["g"] = g
triplet["A_sym"] = A_sym_triplet
triplet["A_num"] = A_num_triplet

"""
def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.cos(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_
"""

def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(dg(t)*np.cos(f_num(t)) - df(t)*dg(t)*np.sin(f_num(t)))
	z = np.array(g_num(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

triplet["w2"] = 0	# dummy value
triplet["true_sol"] = triplet_sol
triplet["t_start"] = 4
triplet["t_stop"] = 10
triplet["x0"] = triplet_sol(np.array([triplet["t_start"]]))
# ---------------------------- #
# -- doublet equation stuff -- #
def f(t):
	return (t**2)
	
def f_num(t):
	return (t**2)
	
def g(t):
	return (0.75)
	
def g_num(t):
	return (0.75)

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

def A_sym_doublet(t):
	F = f(t)
	G = g(t)
	dF = sym.diff(F, t)
	ddF = sym.diff(dF, t)
	dG = sym.diff(G, t)
	ddG = sym.diff(dG, t)
	A = sym.Matrix([[0, 1], [ddG/G - dF**2 - 2*(dG/G)**2 - (dG*ddF)/(G*dF), ddF/F - 2*dG/G]])
	return A
	
A_num_doublet = sym.lambdify((ts), A_sym_doublet(ts), modules=array2mat)
	
doublet = {}
doublet["name"] = "new_2Deq"
doublet["title"] = "Three variable equation"
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
doublet["t_start"] = 4
doublet["t_stop"] = 10
doublet["x0"] = doublet_sol(np.array([doublet["t_start"]]))
# ---------------------------- #

# define numerical functions

def Com(A, B):
	return (A*B - B*A)

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA

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
		
def Omega_num(A, alpha, order):
	# the Magnus expansion Omega truncated to the appropriate order in h
	if order == 4:
		def Omega(t0, t):
			a_1, a_2 = alpha(t0, t, A, 4)
			Om = a_1 - (1/12)*Com(a_1, a_2)
			return Om
		return Omega
	elif order == 6:
		def Omega(t0, t):
			a_1, a_2, a_3 = alpha(t0, t, A, 6)
			C1 = Com(a_1, a_2)
			C2 = -(1/60)*Com(a_1, 2*a_3 + C1)
			Om = a_1 + (1/12)*a_3 + (1/240)*Com(-20*a_1-a_3+C1, a_2+C2)
			return Om
		return Omega

# set up integrators

M1 = {
	"name" : "Magnus sol. with only $\\Omega_1$",
	"fname" : "M1",
	"order" : 2, 
}

M2 = {
	"name" : "Magnus sol. with $\\Omega_1 + \\Omega_2$",
	"fname" : "M2",
	"order" : 4, 
}

WKB = {
	"name" : "WKB sol.",
	"fname" : "WKB",
	"order" : 4,  
}

JWKB = {
	"name" : "Jordan-WKB sol.",
	"fname" : "JWKB",
	"order" : 4, 
}

C4 = {
	"name" : "4th order Cayley-transform sol.",
	"fname" : "C4",
	"order" : 4, 
}

JM1 = {
	"name" : "Jordan-Magnus 1 sol.",
	"fname" : "JM1",
	"order" : 2, 
}

#### Choose equation

Eq = doublet

# choose the lines to plot (i.e. the integrators to use)
lines = [M1, JM1]

########## Symbolics #################
#
#	Symbolic manipulation using sympy

A_sym = Eq["A_sym"]

print("A = ", A_sym(ts))
print()

A_num = Eq["A_num"]
	
"""
define the first and second terms of the Magnus expansion (symbolic form)

Ω_1(t) = \int_t_0^t ( A(t') ) dt'

Ω_2(t) = 0.5 \int_t_0^t( \int_t_0^t'( [A(t'),A(t'')] )dt'' )dt'

"""

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
	
def Cayley_sym():
	A = A_sym(ts0)
	Ndim = A.shape[0]
	Om = Omega_1_sym(A_sym) + Omega_2_sym(A_sym)
	Id = sym.eye(Ndim)
	C_ = Om*(Id - (1/12)*(Om**2)*(Id - (1/10)*(Om**2)))
	M_ = (Id - (1/2)*C_).inv()*(Id + (1/2)*C_)
	return M_
	
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

def Jordan_WKB():
	A = A_sym(ts0)
	Aprime = sym.diff(A, ts0) + A*A
	Ndim = A.shape[0]
	P, J = Aprime.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
	M11 = sym.eye(Ndim)
	M12 = sym.eye(Ndim)
	for i in range(0, Ndim):
		w2 = -J[i,i]
		print("w2 = ", w2)
		w1 = sym.sqrt(w2)
		C = sym.cos(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
		S = sym.sin(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
		dw1 = sym.diff(w1, ts0)
		M11[i,i] = C + S*dw1/(2*w1**2)
		M12[i,i] = S/w1
	dP = sym.diff(P, ts0)
	Pinv = (P.inv()).subs(ts0, ts)
	M_ = Pinv*(M11*P + M12*(dP + P*A))
	return M_
	
def 
	
def Jordan_WKB_num():
	# numerical version
	# symbolics #############
	A = A_sym(ts0)
	Aprime = sym.diff(A, ts0) + A*A
	Ndim = A.shape[0]
	P_sym, J_sym = Aprime.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
	#P_sym = (1 + 0.j)*P_sym
	#J_sym = (1 + 0.j)*J_sym
	print("J = ", J_sym)
	print()
	print("P = ", P_sym)
	print()
	dP = sym.lambdify((ts0), sym.diff(P_sym, ts0), modules=array2mat_c)
	# numerics ##############
	P = sym.lambdify((ts0), P_sym, modules=array2mat_c)
	J = sym.lambdify((ts0), J_sym, modules=array2mat_c)
	Id = np.identity(Ndim)
	M11 = Id.astype(np.complex64)
	M12 = Id.astype(np.complex64)
	def w1_vec(t):
		J_ = J(t)
		w1_ = np.ones(Ndim).astype(np.complex64)
		for n in range(Ndim):
			w2 = -J_[n,n]
			if w2.size > 1:
				w2 = w2[0]
			w1_[n] = np.asscalar(csqrt(w2))
		return w1_
	def w1_sing(t, n, real):
		J_ = J(t)
		w2 = -J_[n,n]
		if w2.size > 1:
			w2 = w2[0]
		w1_ = np.asscalar(csqrt(w2))
		if real:
			w1_ = np.real(w1_)
		elif not real:
			w1_ = np.imag(w1_)
		return w1_
	dw1 = eg(w1_vec, 0.00001)
	def Mf(t0, t):
		# define a function to compute the M matrix
		w1 = w1_vec(t)
		w10 = w1_vec(t0)
		dw10 = dw1(t0)
		Int_w1 = Id
		for i in range(Ndim):
			Int_w1 = quad(w1_sing, t0, t, args=(i, True), maxiter=10)[0] + 1j*quad(w1_sing, t0, t, args=(i, False), maxiter=10)[0]
			C = np.cos(Int_w1)*csqrt(w10[i]/w1[i])
			S = np.sin(Int_w1)*csqrt(w10[i]/w1[i])
			M11[i,i] = C + S*dw10[i]/(2*(w10[i])**2)
			M12[i,i] = S/w1[i]
		Pinv = np.linalg.inv(P(t))
		M_ = Pinv @ (M11 @ P(t0) + M12 @ (dP(t0) + P(t0) @ A_num(t0)))
		return M_
	return Mf
	
def Jordan_Magnus1():
	# symbolics
	A = A_sym(ts0)
	Ndim = A.shape[0]
	P, J = A.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
	print("J = ", J)
	print()
	print("P = ", P)
	print()
	Pinv = P.inv()
	LK = J + sym.diff(Pinv, ts0)*P
	LK = sym.eye(Ndim).multiply_elementwise(LK)
	print("LK = ", LK)
	print()
	
	Use_symbolic = True
	if Use_symbolic:
		Om1 = sym.integrate(LK.subs(ts0, ts1), (ts1, ts0, ts)) 
		print("Ω1 = ", Om1)
		print()
		Om1_num = sym.lambdify((ts0, ts), Om1, modules=array2mat_c)
	elif not Use_symbolic:
		LK_num = sym.lambdify((ts0), LK, modules=array2mat_c)
		Om1_num = Omega_num(LK_num, alpha_GL, 4)
		JM1["name"] = JM1["name"] + " (numerical, GL quad)"
	P_num = sym.lambdify((ts0), P, modules=array2mat_c)
	Pinv_num = sym.lambdify((ts0), P.inv(), modules=array2mat_c)
	#
	def Mf(t0, t):
		M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ Pinv_num(t0)
		return M_.astype(np.float64)
	return Mf

################# Numerics #################

# do I need to calculate the M(t) matrices for 
# lines M1, M2, WKB, JWKB, C4 ?
Integrators = ["M1", "M2", "WKB", "JWKB", "C4", "JM1"]
Calculate_Matrix = [False, False, False, False, False, False]

for i in range(len(lines)):
	M = lines[i]
	for j in range(len(Integrators)):
		if (M["fname"] == Integrators[j]):
			Calculate_Matrix[j] = True
		
if Calculate_Matrix[0] or Calculate_Matrix[1]:
	print("Omega 1 = ", Omega_1_sym(A_sym))
	print()
	Omega_1_exact = sym.lambdify((ts0, ts), Omega_1_sym(A_sym), modules=array2mat)
	def Magnus1(t0, t):
		return linalg.expm(Omega_1_exact(t0, t))
	M1["M"] = Magnus1
if Calculate_Matrix[1]:
	print("Omega 2 = ", Omega_2_sym(A_sym))
	print()
	Omega_2_exact = sym.lambdify((ts0, ts), Omega_2_sym(A_sym), modules=array2mat)
	def Magnus2(t0, t):
		return linalg.expm(Omega_1_exact(t0, t) + Omega_2_exact(t0, t))
	M2["M"] = Magnus2
if Calculate_Matrix[2]:
	print("WKB matrix = ", WKB_matrix_sym())
	print()
	WKB["M"] = sym.lambdify((ts0, ts), WKB_matrix_sym(), modules=array2mat)
"""
if Calculate_Matrix[3]:
	print("Jordan-WKB matrix = ", Jordan_WKB())
	JWKB["M"] = sym.lambdify((ts0, ts), Jordan_WKB(), modules=array2mat)
"""
if Calculate_Matrix[3]:
	JWKB["M"] =  Jordan_WKB_num()

if Calculate_Matrix[4]:
	print("4th order Cayley matrix = ", Cayley_sym())
	C4["M"] = sym.lambdify((ts0, ts), Cayley_sym(), modules=array2mat)
	
if Calculate_Matrix[5]:
	JM1["M"] = Jordan_Magnus1()
		
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
#print("x0 = ", x0) 

n_steps = 500
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
		if (n*100) % (len(t_vec)-1) == 0:
			print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return x
	
######### plot graph ##################

for data in lines:
	data["x"] = Integrator_1(t_vec, x0, data["M"])[:,0]

linenames = ""
for data in lines:
	linenames = linenames + data["fname"] + "_"

filename = "Analytic_test_" + Eq["name"] + "_" + linenames #"_t_start=" + str(t_start) + "_t_stop=" + str(t_stop) 
title = Eq["title"] + " : WKB vs alternative integrators"

colours = ['r-', 'b-', 'g-', 'c-', 'm-']
x_true = Eq["true_sol"](t_vec0)[:,0]
plt.plot(t_vec0, x_true, color="0", linewidth=1, linestyle="--", label="true soln.")
for i in range(len(lines)):
	M = lines[i]
	plt.plot(t_vec, M["x"], colours[i], markersize=2, linewidth=1, label=M["name"])
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-1, 1)
plt.title(title)
plt.minorticks_on()
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()
print("made plot")
print("saved plot as " + "Plots/" + filename + ".pdf")
