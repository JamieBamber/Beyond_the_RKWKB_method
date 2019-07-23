# Magnus Solver

"""
To solve 

x'(t) = A(t) x(t)

for equations of the form 

x'' + ω^2x = 0

The simple case of ω^2 = const. i.e. the SHO

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

def w2_Airy(t):
	return t
A_Airy = A_from_w2(w2_Airy)

def w2_burst(t):
	n = 5
	w = (n**2 - 1)/(1 + t**2)**2
	return w
	
def w2_SHO(t):
	return 1
	
A_burst = A_from_w2(w2_burst)

A_SHO = A_from_w2(w2_SHO)

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
	
print("Omega_1 = ", Omega_1_sym(A_SHO))
	
def Omega_2_sym(A):
	ts1 = sym.Symbol('ts1')
	ts2 = sym.Symbol('ts2')
	integral_1 = sym.integrate(Com(A(ts1),A(ts2)), (ts2, ts0, ts1))
	integral_2 = sym.integrate(integral_1, (ts1, ts0, ts))
	return 0.5*integral_2

print("Omega_2 = ", Omega_2_sym(A_SHO))

def Omega_3_sym(A):
	ts1 = sym.Symbol('ts1')
	ts2 = sym.Symbol('ts2')
	ts3 = sym.Symbol('ts3')
	integral_1 = sym.integrate(Com(A(ts1),Com(A(ts2), A(ts3))), (ts3, ts0, ts2))
	integral_2 = sym.integrate(integral_1, (ts2, ts0, ts1))
	integral_3 = sym.integrate(integral_2, (ts1, ts0, ts))
	return (1/6)*integral_3

print("Omega_3 = ", Omega_3_sym(A_SHO))

########### Numerics #################

"""
Convert symbolic expressions to 
numpy functions
"""

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']

A = A_SHO # choose the A(t) matrix

Omega_1 = sym.lambdify((ts0, ts), Omega_1_sym(A), modules=array2mat)
Omega_2 = sym.lambdify((ts0, ts), Omega_2_sym(A), modules=array2mat)
Omega_3 = sym.lambdify((ts0, ts), Omega_3_sym(A), modules=array2mat)

T_sym = time.time()
print("Done symbolic manipulation, time taken = {:.5f}".format(T_sym - T_start))

"""
Define a stepping function
"""

def f_step(t0, t):
	return linalg.expm(Omega_1(t0, t)) # + Omega_2(t0, t) + Omega_3(t0, t))

########## Integrator #################

def Integrator_1(t_vec, x0):
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(2)
	for i in range(1,len(t_vec)):
		x[i,:] = (f_step(t_vec[i-1], t_vec[i]) @ x[i-1,:].reshape(2, 1)).reshape(2)
		if (i*100) % (len(t_vec)-1) == 0:
			print("\r" + "integrated {:.0%}".format(i/(len(t_vec)-1)), end='')
		
	print('')
		
	return x
	
######## Integration ##################
		
"""
trying the simple case of the SHO
"""


x0 = np.array([1, 0]).reshape(2, 1)
n_steps = 1000
t_vec = np.linspace(0, 35, n_steps)
t_vec0 = np.linspace(0, 35, 1000)

x = Integrator_1(t_vec, x0)

T_num = time.time()
print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_sym))

######## Save Results #################

filename = "SHO_1st_order_" + str(n_steps) + "_steps"
file_path = "Data/" + filename + ".txt"

f = open(file_path, "w") # empty file
f.close()
f = open(file_path, "w")

f.write(time.strftime("%X") + "\n")
f.write("time to do symbolics	= {}s \n".format(T_sym - T_start))
f.write("time to do numerics	= {}s \n".format(T_num - T_sym))
f.write("t	x	x' \n")
for i in range(len(t_vec)):
	f.write("{}	{}	{} \n".format(t_vec[i], x[i, 0], x[i, 1]))

f.close()

print("saved data")

####### Get true solution #############

x_true = x0[0,0]*np.cos(t_vec) + x0[1,0]*np.sin(t_vec0)
dxdt_true = -x0[0,0]*np.sin(t_vec) + x0[1,0]*np.cos(t_vec0)

###### plot graph #####################

plt.plot(t_vec, x[:,0], 'r', linewidth=1, label="Magnus stepping")
plt.plot(t_vec0, x_true, color="0.5", linewidth=1, linestyle="--", label="true soln.")
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-1.5, 1.5)
plt.title(filename)
plt.legend()
plt.savefig("Plots/" + filename + ".pdf", transparent=True)
plt.clf()

print("made plot")

	