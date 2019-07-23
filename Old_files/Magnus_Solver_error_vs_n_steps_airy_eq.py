# Magnus Solver

"""
investigate error vs the number of steps used
For the Airy equation
"""
"""
To solve 

x'(t) = A(t) x(t)

"""

import numpy as np
import sympy as sym
import time
from scipy import special, linalg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_start = time.time()

########## Symbolics #################
#
#	Symbolic manipulation using sympy

"""
Define a function for the 2x2 matrix from a 1D equation of the form

x'' + ω^2(t)x = 0

[ x' ]     [ 0      1 ] [ x ]
[ x'']  =  [ -ω^2   0 ] [ x']

"""

def A_from_w2(w2):
	def f(t):
		M = sym.Matrix([[0, 1], [-w2(t), 0]])
		return M
	return f	

"""
create 2x2 A matrices corresponding to the Airy 
and burst equations respectively
"""
'''
def w2_Airy(t):
	return t
A_Airy = A_from_w2(w2_Airy)

n_burst = 20
def w2_burst(t):
	n = n_burst
	w = (n**2 - 1)/(1 + t**2)**2
	return w
A_burst = A_from_w2(w2_burst)

"""
define the first and second terms of the Magnus expansion (symbolic form)

Ω_1(t) = \int_t_0^t ( A(t') ) dt'

Ω_2(t) = 0.5 \int_t_0^t( \int_t_0^t'( [A(t'),A(t'')] )dt'' )dt'

"""

ts0 = sym.Symbol('ts0')
ts = sym.Symbol('ts')

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

def Omega_3_sym(A):
	ts1 = sym.Symbol('ts1')
	ts2 = sym.Symbol('ts2')
	ts3 = sym.Symbol('ts3')
	integral_1 = sym.integrate(Com(A(ts1),Com(A(ts2), A(ts3))), (ts3, ts0, ts2))
	integral_2 = sym.integrate(integral_1, (ts2, ts0, ts1))
	integral_3 = sym.integrate(integral_2, (ts1, ts0, ts))
	return (1/6)*integral_3

########### Numerics #################

"""
Convert symbolic expressions to 
numpy functions
"""

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']

A = A_Airy # choose the A(t) matrix

print("Omega_1 = ", Omega_1_sym(A))
print("Omega_2 = ", Omega_2_sym(A))
print("Omega_3 = ", Omega_3_sym(A))

Omega_1 = sym.lambdify((ts0, ts), Omega_1_sym(A), modules=array2mat)
Omega_2 = sym.lambdify((ts0, ts), Omega_2_sym(A), modules=array2mat)
Omega_3 = sym.lambdify((ts0, ts), Omega_3_sym(A), modules=array2mat)

T_sym = time.time()
print("Done symbolic manipulation, time taken = {:.5f}".format(T_sym - T_start))

"""
Define a stepping function
"""

def f_step(t0, t, N):
	# N = order to go up to, should be 1, 2 or 3
	if N == 1:
		return linalg.expm(Omega_1(t0, t)) # + Omega_2(t0, t) + Omega_3(t0, t))
	elif N == 2:
		return linalg.expm(Omega_1(t0, t) + Omega_2(t0, t)) # + Omega_3(t0, t))
	elif N == 3:
		return linalg.expm(Omega_1(t0, t) + Omega_2(t0, t) + Omega_3(t0, t))
	
########## Integrator #################

def Integrator_1(t_vec, x0, N):
	"""
	x0 = initial conditions
	t_vec = vector of times
	"""
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(2)
	for i in range(1,len(t_vec)):
		x[i,:] = (f_step(t_vec[i-1], t_vec[i], N) @ x[i-1,:].reshape(2, 1)).reshape(2)
		if (i*100) % (len(t_vec)-1) == 0:
			print("\r" + "integrated {:.0%}".format(i/(len(t_vec)-1)), end='')
	print('')
	
	return x

######## True solutions ###############

""" Airy function """

Ai0, Aip0, Bi0, Bip0 = special.airy(0)

####### Get true solution #############

x0 = np.array([0.75, 0]).reshape(2, 1)

T_max = 100

"""
find the coefficients for the true solution of the
Airy equation with the given initial conditions
"""

M = (1/(Ai0*Bip0 - Aip0*Bi0))*np.matrix([[Bip0, -Bi0], [-Aip0, Ai0]])
ab = M @ x0	
a = ab[0,0]
b = ab[1,0]

Ai, Aip, Bi, Bip = special.airy(-T_max)
x_true = a*Ai + b*Bi
dxdt_true = a*Aip + b*Bip
'''
######## Integration ##################

n_ = np.round(np.logspace(0, 5, 50))
x_ = np.zeros((50, 3))
'''
for N in [1, 2, 3]:
	for i in range(n_.size):
		t_vec = np.linspace(0, 100, n_[i])
		x = Integrator_1(t_vec, x0, N)
		x_[i, N-1] = x[-1, 0]
		print('#',i)
	
err = np.abs(x_true - x_)

T_num = time.time()
print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_sym))

######## Save Results #################
'''
filename = "Airy_error_vs_n_steps"
file_path = "Data/" + filename + ".txt"
'''
f = open(file_path, "w") # empty file
f.close()
f = open(file_path, "w")

f.write(time.strftime("%X") + "\n")
f.write("time to do symbolics	= {}s \n".format(T_sym - T_start))
f.write("time to do numerics	= {}s \n".format(T_num - T_sym))
f.write("n_steps	err (1st ord.)	err (2nd ord.)	err (3rd ord.)  \n")
for i in range(n_.size):
	f.write("{}	{}	{}	{} \n".format(n_[i], err[i, 0], err[i, 1], err[i, 2]))
f.close()

print("saved data")
'''
####### Open data ####################

f = open(file_path, "r")
err = np.loadtxt(f, skiprows=4, delimiter='	', usecols=(1, 2, 3))
print("read data")

###### plot graph #####################

plt.plot(n_, np.log10(err[:,0]), 'r-', linewidth=1, label="1st order")
plt.plot(n_, np.log10(err[:,1]), 'b-', linewidth=1, label="2nd order")
plt.plot(n_, np.log10(err[:,2]), 'g-', linewidth=1, label="3rd order")
plt.xlabel("no. steps")
plt.ylabel("log$_{10}$(err. at t = 100)")
plt.ylim(-15, 10)
plt.xscale('log')
plt.minorticks_on()
plt.title(filename)
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()
print("made plot")

	