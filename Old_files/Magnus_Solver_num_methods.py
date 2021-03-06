# Magnus Solver

"""
To solve 

x'(t) = A(t) x(t)

"""
"""
Using Magnus expansion derived numerical methods from
"the Magnus expansion and some of its applications"
pg. 91 - 95
"""

import numpy as np

#from autograd import elementwise_grad as eg
# package to do numerical differentiation
# https://github.com/HIPS/autograd/blob/master/docs/tutorial.md

import time
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
def A_from_w2(w2):
	def f(t):
		M = np.matrix([[0, 1], [-w2(t), 0]])
		return M
	return f

# --- Airy equation stuff --- #
Airy = {}
Airy["name"] = "Airy"
Airy["x0"] = np.array([0.75, 0])

def w2_Airy(t):
	return t
Airy["A"] = A_from_w2(w2_Airy)

def Airy_sol(t):
	Ai0, Aip0, Bi0, Bip0 = special.airy(0)
	M = (1/(Ai0*Bip0 - Aip0*Bi0))*np.matrix([[Bip0, -Bi0], [-Aip0, Ai0]])
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

def w2_burst(t):
	n = burst["n"]
	w = (n**2 - 1)/(1 + t**2)**2
	return w
burst["A"] = A_from_w2(w2_burst)

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

burst["x0"] = x0 = np.array([burst_soln(-burst["n"],burst["n"]), dburst_soln(-burst["n"],burst["n"])])
burst["true_sol"] = Burst_sol
burst["t_start"] = -burst["n"]
burst["t_stop"] = +burst["n"]
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
		
def Omega(t0, t, A, alpha, order):
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
	
########## Integrator #################

def RK4_Integrator(t_start, t_stop, n_step, x0, A):
	# An integrator using a 4th order RK method with fixed
	# step size
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	n_step = number of steps
	A = A(t) matrix function
	"""
	Ndim = x0.size
	x_ = np.zeros((n_step+1, x0.size)) # set up the array of x values
	t_ = np.zeros(n_step+1)
	t_[0] = t_start
	x_[0,:] = x0
	h = (t_stop - t_start) / n_step
	#
	n = 0
	t = t_start
	while n < n_step:
		x_n = x_[n,:].reshape(Ndim, 1)
		k1 = h*A(t) @ x_n
		k2 = h*A(t + 0.25*h) @ (x_n + 0.25*k1)
		k3 = h*A(t + (3/8)*h) @ (x_n + (3/32)*k1 + (9/32)*k2)
		k4 = h*A(t + (12/13)*h) @ (x_n + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
		k5 = h*A(t + h) @ (x_n + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
		x_np1 = x_n + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (11/40)*k5
		x_[n+1,:] = x_np1.reshape(Ndim)
		t = t + h
		t_[n+1] = t
		n = n + 1
		if np.round((t-t_start)*10000 / (t_stop-t_start)) % 100 == 0:
			print("\r" + "integrated {:.0%}".format((t-t_start)/(t_stop-t_start)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)

def Magnus_Integrator(t_start, t_stop, n_step, x0, A, alpha, order):
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	n_step = number of steps
	A = A(t) matrix function
	alpha = alpha generating function
	order = 4 or 6
	"""
	Ndim = x0.size
	x_ = np.zeros((n_step+1, Ndim)) # set up the array of x values
	t_ = np.zeros(n_step+1)
	t_[0] = t_start
	x_[0,:] = x0
	h = (t_stop - t_start) / n_step
	#
	n = 0
	t = t_start
	while n < n_step:
		x_[n+1,:] = (linalg.expm(Omega(t, t+h, A, alpha, order)) @ x_[n,:].reshape(Ndim, 1)).reshape(Ndim)
		t = t + h
		t_[n+1] = t
		n = n + 1
		if np.round((t-t_start)*10000 / (t_stop-t_start)) % 100 == 0:
			print("\r" + "integrated {:.0%}".format((t-t_start)/(t_stop-t_start)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)

###### set up #########################

"""
maybe put some other settings in here to make using different integrators easier?
"""

M4_GL = {
	"name" : "Magnus 4$^\\circ$, GL quad",
	"fname" : "M4GL"
}

M4_D = {
	"name" : "Magnus 4$^\\circ$, num. diff",
	"fname" : "M4D"
}

M6_D = {
	"name" : "Magnus 6$^\\circ$, num. diff",
	"fname" : "M6D"
}

M6_GL = {
	"name" : "Magnus 6$^\\circ$, GL quad",
	"fname" : "M6GL"
}

M4_SNC = {
	"name" : "Magnus 4$^\\circ$, Simpson quad",
	"fname" : "M4SNC"
}

M6_SNC = {
	"name" : "Magnus 6$^\\circ$, NC quad",
	"fname" : "M6SNC"
}

RK4 = {
	"name" : "RK 4th order",
	"fname" : "RK4"
}

###### plot graph #####################

def plot_graph():
	# function for plotting a graph of the results.
	Eq = Airy
	n_step = 50
	
	lines = [M4_GL, M4_D]
	######## Integration ##################
	M4_GL["data"] = Magnus_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"], alpha_GL, 4)
	M4_D["data"] = Magnus_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"], alpha_D, 4)
	#M4_SNC["data"] = Magnus_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"], alpha_SNC, 4)
	#M6_GL["data"] = Magnus_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"], alpha_GL, 6)
	#M6_SNC["data"] = Magnus_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"], alpha_SNC, 6)
	#RK4["data"] = RK4_Integrator(Eq["t_start"], Eq["t_stop"], n_step, Eq["x0"], Eq["A"])
	t_vec0 = np.linspace(Eq["t_start"], Eq["t_stop"], 1000)
	x_true = Eq["true_sol"](t_vec0)
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames + "nstep_" + str(n_step)
	#
	colours = ['r', 'm', 'b', 'c', 'g']
	markertypes = ['+', 'x', '+', '^', 'x']
	plt.plot(t_vec0, x_true[:,0], color="0.5", linewidth=1, linestyle="--", label="true soln.")
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = line["data"][1][:,0]
		T = line["data"][2]
		plt.plot(t, x, colours[i] + markertypes[i], markersize=3, linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		#plt.plot(data["t"], data["x"][:,0], colours[i] + '-', alpha=0.5, linewidth=1)
	plt.xlabel("t")
	plt.ylabel("x")
	plt.ylim(-1, 1)
	plt.title(Eq["name"] + " equation, numerical solutions")
	plt.legend(fontsize=8)
	plt.savefig("Plots/" + filename + ".pdf", transparent=True)
	plt.clf()
	print("made plot")
	
def plot_errors():
	# function for plotting the errors of the results
	# function for plotting a graph of the results.
	Eq = Airy
	
	SF = 10.0 # factor by which the time interval t_stop - t_start is increased
	h = 0.25 # stepsize
	new_t_stop = Eq["t_start"] + SF*(Eq["t_stop"] - Eq["t_start"])
	n_step = int((new_t_stop - Eq["t_start"])/h)
	
	lines1 = [M4_GL, M4_SNC, M6_GL,  M6_SNC]
	lines2 = [M4_D, M6_D]
	
	lines = lines2
	
	#M4_GL["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_GL, 4)
	M4_D["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_D, 4)
	#M4_SNC["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_SNC, 4)
	#M6_GL["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_GL, 6)
	M6_D["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_D, 6)
	#M6_SNC["data"] = Magnus_Integrator(Eq["t_start"], new_t_stop, n_step, Eq["x0"], Eq["A"], alpha_SNC, 6)
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames + "h" + str(h)
	#
	colours1 = ['rx', 'c^', 'mx', 'b.']
	colours2 = ['k+', 'gx'] 
	line_colours = ['r-', 'k-', 'b-', 'g-', 'm-']
	
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = line["data"][1][:,0]
		T = line["data"][2]
		x_true = Eq["true_sol"](t)[:,0]
		err = np.log(np.abs(x - x_true))
		plt.plot(t, err, colours2[i], markersize=1, linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		#plt.plot(data["t"], err, line_colours[i], alpha=0.1, linewidth=1)
	plt.xlabel("t")
	plt.ylim((-25, 0))
	plt.ylabel("ln(absolute error)")
	plt.title(Eq["name"] + " equation, numerical solutions, error vs time, h=" + str(h) + "s")
	lgnd = plt.legend()
	for i in range(0, len(lines)):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)
	plt.savefig("Plots/" + filename + "_error.pdf", transparent=True)
	plt.clf()
	print("made plot")
	print("saved in " + "Plots/" + filename + "_error.pdf")
	
	
#plot_graph()
plot_errors()

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

	