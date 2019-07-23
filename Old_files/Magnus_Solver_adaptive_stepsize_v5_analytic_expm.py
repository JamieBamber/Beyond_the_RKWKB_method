# Magnus Solver

"""
To solve 

x'(t) = A(t) x(t)

"""
"""
Using Magnus expansion derived numerical methods from
"the Magnus expansion and some of its applications"
pg. 91 - 95

and the exact analytic solutions
"""

import autograd.numpy as np
#from autograd import elementwise_grad as eg
# package to do numerical differentiation
# https://github.com/HIPS/autograd/blob/master/docs/tutorial.md

import time
import sympy as sym
from scipy import special, linalg

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import gridspec

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
burst["n"] = 20
burst["name"] = "Burst"
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

############# define some functions ##########

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA

def Com(A, B):
	return (A*B - B*A)

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

###### set up integrator dictionaries #########################

"""
maybe put some other settings in here to make using different integrators easier?
"""
M1 = {
	"name" : "Magnus with $\\Omega_1$, analytic func.",
	"fname" : "M1",
	"alpha" : "analytic",
	"order" : 2,
	"type" : "Magnus" 
}

M2 = {
	"name" : "Magnus with $\\Omega_1+\\Omega_2$, analytic func.",
	"fname" : "M2",
	"alpha" : "analytic",
	"order" : 4,
	"type" : "Magnus" 
}

M4_GL = {
	"name" : "Magnus 4$^\\circ$, GL quad",
	"fname" : "M4GL",
	"alpha" : alpha_GL,
	"order" : 4,
	"type" : "Magnus"  
}

M4_D = {
	"name" : "Magnus 4$^\\circ$, num. diff",
	"fname" : "M4D",
	"alpha" : alpha_D,
	"order" : 4,
	"type" : "Magnus"  
}

M6_D = {
	"name" : "Magnus 6$^\\circ$, num. diff",
	"fname" : "M6D",
	"alpha" : alpha_D,
	"order" : 6,
	"type" : "Magnus"  
}

M6_GL = {
	"name" : "Magnus 6$^\\circ$, GL quad",
	"fname" : "M6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"type" : "Magnus"  
}

M4_SNC = {
	"name" : "Magnus 4$^\\circ$, Simpson quad",
	"fname" : "M4SNC",
	"alpha" : alpha_SNC,
	"order" : 4,
	"type" : "Magnus"  
}

M6_SNC = {
	"name" : "Magnus 6$^\\circ$, NC quad",
	"fname" : "M6SNC",
	"alpha" : alpha_SNC,
	"order" : 6,
	"type" : "Magnus"  
}

RKF45 = {
	"name" : "RKF 4(5)",
	"fname" : "RKF45",
	"type" : "RKF45"  
}

C4_GL = {
	"name" : "Caley 4$^\\circ$, GL quad",
	"fname" : "C4GL",
	"alpha" : alpha_GL,
	"order" : 4,
	"type" : "Caley"  
}

C6_GL = {
	"name" : "Caley 6$^\\circ$, GL quad",
	"fname" : "C6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"type" : "Caley"  
}

######################################

# Choose equation
Eq = burst

# choose the lines to plot (i.e. the integrators to use)
lines = [M1, M4_GL, C4_GL]

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

def w(t):
	return sym.sqrt(Eq["w2"](t))
	
def B(ts, ts0):
	w = sym.sqrt(Eq["w2"](ts))
	w0 = sym.sqrt(Eq["w2"](ts0))
	dw = sym.diff(w(ts), ts)
	dw0 = d2.subs(ts, ts0)
	B_ = sym.Matrix([[1, 0], [dw/(w**2), 1/w]])
	return B_

def Omega_WKB(ts, ts0):
	w = sym.sqrt(Eq["w2"](ts))
	w0 = sym.sqrt(Eq["w2"](ts0))
	dw = sym.diff(w(ts), ts)
	dw0 = d2.subs(ts, ts0)
	A_WKB = sym.Matrix([[-dw/(2*w), w], [-w,-dw/(2*w)]])
	Om = sym.integrate(A(ts1), (ts1, ts0, ts))
	return Om

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']

make_Om1 = False
make_Om2 = False
for M in lines:
	if (M["fname"] == "M1"):
		make_Om1 = True
	if (M["fname"] == "M2"):
		make_Om2 = True
		
if make_Om1 or make_Om2:
	print("Omega 1 = ", Omega_1_sym(A_sym))
	Omega_1_exact = sym.lambdify((ts0, ts), Omega_1_sym(A_sym), modules=array2mat)
if make_Om2:
	print("Omega 2 = ", Omega_2_sym(A_sym))
	Omega_2_exact = sym.lambdify((ts0, ts), Omega_2_sym(A_sym), modules=array2mat)

"""
print("Omega 1 = ", Omega_1_sym(A_sym))
Omega_1_exact = sym.lambdify((ts0, ts), Omega_1_sym(A_sym), modules=array2mat)
print("Omega 2 = ", Omega_2_sym(A_sym))
Omega_2_exact = sym.lambdify((ts0, ts), Omega_2_sym(A_sym), modules=array2mat)
"""

############# define some more functions ##########
		
def Omega(t0, t, A, alpha, order):
	if alpha == "analytic":
		# exact analytic functions for Omega
		if order == 2:
			Om = Omega_1_exact(t0, t)
			return Om 
		elif order == 4:
			Om = Omega_1_exact(t0, t) + Omega_2_exact(t0, t)
			return Om
	elif alpha != "analytic":
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

def ferr(x_0, x_l):
	# a function to evaluate an error between step estimates
	# returns a vector
	err = np.abs(x_0 - x_l)
	return err
	
def log_minor_ticks(ax):
	locmin = ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10)) 
	ax.yaxis.set_minor_locator(locmin)
	ax.yaxis.set_minor_formatter(ticker.NullFormatter())
	
########## Integrator #################

# set error tolerance
epsilon	= 0.005
epsilon_RK = 0.005
rtol = 1		# rel. error tolerance for Magnus in units of ε
atol = 0.005	# abs. error tolerance for Magnus in units of ε
rtol_RK = 2		# rel. error tolerance for RKF4(5) in units of ε_RK
atol_RK = 1		# abs. error tolerance for RKF4(5) in units of ε_RK

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
	h_min = h0*(10**(-2))
	h_max = 5*h0
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
			test_M = np.matrix([[1, 2], [1, 2]])
			k1 = np.dot(test_M,x_n)
			k1 = np.dot(h*A(t),x_n)
			k2 = h*A(t + 0.25*h) @ (x_n + 0.25*k1)
			k3 = h*A(t + (3/8)*h) @ (x_n + (3/32)*k1 + (9/32)*k2)
			k4 = h*A(t + (12/13)*h) @ (x_n + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
			k5 = h*A(t + h) @ (x_n + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
			k6 = h*A(t + 0.5*h) @ (x_n - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
			y_np1 = x_n + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (11/40)*k5
			z_np1 = x_n + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
			#
			Err =  ferr(y_np1, z_np1)
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

def Magnus_Integrator(t_start, t_stop, h0, x0, A, alpha, type_, order):
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	A = A(t) matrix function
	alpha = alpha generating function
	order = 4 or 6
	
	also support Caley methods
	
	"""
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = h0*(10**(-2))
	h_max = 5*h0
	n = 0
	t = t_start
	#
	S = 0.99				# safety factor
	#
	if type_ == "Magnus":
		def M(time, hstep):
			M_ = linalg.expm(Omega(time, time+hstep, A, alpha, order))	
			return M_
	elif type_ == "Caley":
		def M(time, hstep):
			Id = np.identity(Ndim)
			Om = Omega(time, time+hstep, A, alpha, order)
			if order == 4:
				C_ = Om*(Id - (1/12)*(Om**2))
			elif order ==6:
				C_ = Om*(Id - (1/12)*(Om**2)*(1 - (1/10)*(Om**2)))
			M_ = np.linalg.inv(Id - 0.5*C_)*(Id + 0.5*C_)
			return M_
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using one step of h & two steps of h/2
			#print("\r" + "trying step " + str(n) + " h=" + str(h) + " ...", end='')
			x_np1_0 = M(t, h) @ x_n
			x_np1_l = M(t+0.5*h, 0.5*h) @ (M(t, 0.5*h) @ x_n)
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
			print("\r" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)

###### plot graph #####################

def plot_graph():
	# function for plotting a graph of the results.
	h0 = 1.0
	
	MKR_size = 1	# marker size
		
	######## Integration ##################
	
	t_stop = Eq["t_stop"]
	
	for M in lines:
		if M["type"] != "RKF45":
			M["data"] = Magnus_Integrator(Eq["t_start"], t_stop, h0, Eq["x0"], Eq["A"], M['alpha'], M["type"], M['order'])
		elif M["type"] == "RKF45":
			M["data"] = RKF45_Integrator(Eq["t_start"], t_stop, h0, Eq["x0"], Eq["A"])

	t_vec0 = np.linspace(Eq["t_start"], t_stop, 1000)
	x_true = Eq["true_sol"](t_vec0)
	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_adaptive_" + linenames + "rtol=" + str(rtol) + "_atol=" + str(atol) + "_epsil=" + str(epsilon) + "_rtolRK=" + str(rtol_RK)
	#
	colours = ['r', 'b', 'g', 'c', 'm']
	markertypes = ['x', '+', 'x', '^', 'x']
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
	
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	ax0.plot(t_vec0, x_true[:,0], color="0.7", linewidth=1, linestyle="--", label="true soln.")
	#
	ax2 = plt.subplot(gs[2], sharex = ax0)
	ax2.plot(np.linspace(Eq["t_start"], t_stop, 20), np.log10(epsilon*np.ones(20)), color="0.5", linewidth=1, linestyle=":", label="$\epsilon$")
	#ax2.annotate("$\epsilon$", xy=(1.05*Eq["t_stop"], epsilon))
	#
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = line["data"][1][:,0]
		x_true = Eq["true_sol"](t)[:,0]
		T = line["data"][2]
		ax0.plot(t, x, colours[i] + markertypes[i], markersize=MKR_size, linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		ax2.plot(t, np.log10(np.abs((x - x_true)/x_true)), colours[i] + '--', linewidth=1, alpha=1)
	ax0.set_ylabel("x")
	ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	ax2.set_ylabel("log$_{10}$(rel. error)")
	ax2.legend()
	ax2.set_xlabel("t")
	ax2.minorticks_on()
	ax0.minorticks_on()
	ax0.set_title(Eq["name"] + " equation, adaptive step size", y=1.08)
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
		ax1.plot(t_av, h, colours[i] + '-', linewidth=1, label="{:s}".format(line["name"]))
	ax1.set_ylabel("h")
	ax1.minorticks_on()
	# remove last tick label for the second subplot
	#plt.setp(ax2.get_yticklabels()[-2], visible=False) 
	#plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.subplots_adjust(hspace=.0)
	plt.savefig("Plots/" + filename + ".pdf", transparent=True)
	plt.clf()
	print("made plot")
	print("saved as " + "Plots/" + filename + ".pdf")
	
#################################################################

plot_graph()

'''
######## Save Results #################

filename = Equation + "_" + str(n_steps) + "_steps"
file_path = "Data/" + filename + ".txt"

f = open(file_path, "w") # empty file
f.close()
f = open(file_path, "w")

f.write(time.strftime("%X") + "\n")
f.write("time to do numerics	= {}s \n".format(T_num - T_start))
f.write("t	x	x' (first order) x	x' (2nd order) etc. \n")
for i in range(len(t_vec)):
	f.write("{}	".format(t_vec[i]))
	for N in range(1, N_max+1):
		f.write("{}	{}	".format(x[i, 0, N-1], x[i, 1, N-1]))
	f.write("\n")

f.close()

print("saved data")
'''

	
