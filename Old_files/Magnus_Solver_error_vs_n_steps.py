# Magnus Solver

"""
To solve 

x'(t) = A(t) x(t)

"""
#################

Choose the Equation
Equation = "Airy"

#################

import numpy as np
import sympy as sym
import time
from scipy import special, linalg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_start = time.time()

def Make_new_data():
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
	
	if Equation == "Airy":
		A = A_Airy # choose the A(t) matrix
		N_max = 3
		T_max = 35
	elif Equation == "burst":
		A = A_burst
		N_max = 2
		T_max = n_burst
		
	print("Omega_1 = ", Omega_1_sym(A))
	Omega_1 = sym.lambdify((ts0, ts), Omega_1_sym(A), modules=array2mat)
	if N_max >= 2:
		print("Omega_2 = ", Omega_2_sym(A))
		Omega_2 = sym.lambdify((ts0, ts), Omega_2_sym(A), modules=array2mat)
	if N_max >= 3:
		print("Omega_3 = ", Omega_3_sym(A))
		Omega_3 = sym.lambdify((ts0, ts), Omega_3_sym(A), modules=array2mat)
	
	T_sym = time.time()
	print("Done symbolic manipulation, time taken = {:.5f}".format(T_sym - T_start))
	
	"""
	Define a stepping function
	"""
	
	def f_step(t0, t, N):
		# N = order to go up to, should be 1, 2 or 3
		if N == 1:
			return linalg.expm(Omega_1(t0, t))
		elif N == 2:
			return linalg.expm(Omega_1(t0, t) + Omega_2(t0, t))
		elif N == 3:
			return linalg.expm(Omega_1(t0, t) + Omega_2(t0, t) + Omega_3(t0, t))
		
	########## Integrator #################
	
	def Integrator_1(t_vec, x0):
		"""
		x0 = initial conditions
		t_vec = vector of times  (N,) shape array
		"""
		x = np.zeros((len(t_vec), x0.shape[0], N_max)) # set up the array of x values
		
		for N in range(1, N_max+1):
			x[0, :, N-1] = x0.reshape(2)
			for i in range(1,len(t_vec)):
				x[i,:,N-1] = (f_step(t_vec[i-1], t_vec[i], N) @ x[i-1,:].reshape(2, 1)).reshape(2)
				if (i*100) % (len(t_vec)-1) == 0:
					print("\r" + "integrated {:.0%}".format(i/(len(t_vec)-1)), end='')
			
			print('done order ', N)
			
		return x
	
	######## True solutions ###############
	
	""" Airy function """
	
	Ai0, Aip0, Bi0, Bip0 = special.airy(0)
	
	""" burst equation """
	
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
	
	####### Get true solution #############
	
	if Equation == "Airy":
		"""
		find the coefficients for the true solution of the
		Airy equation with the given initial conditions
		"""
		
		M = (1/(Ai0*Bip0 - Aip0*Bi0))*np.matrix([[Bip0, -Bi0], [-Aip0, Ai0]])
		ab = M @ x0	
		print(ab)
		a = ab[0,0]
		b = ab[1,0]
		
		Ai, Aip, Bi, Bip = special.airy(-T_max)
		x_true = a*Ai + b*Bi
		dxdt_true = a*Aip + b*Bip
	
	if Equation == "burst"
		"""
		find the coefficients for the true solution of the
		burst equation with the given initial conditions
		"""
		x_true = burst_soln(t_vec0, n_burst)
	
	######## Integration ##################
			
	"""
	for the burst equation or the airy equation
	"""
	
	n_steps = 500
	
	if Equation == "Airy":
		#x0 = np.array([Ai0, -Aip0]).reshape(2, 1)
		x0 = np.array([0.75,0]).reshape(2, 1)
		t_vec = np.linspace(0, T_max, n_steps)
	elif Equation == "burst":
		x0 = np.array([burst_soln(-n_burst,n_burst), dburst_soln(-n_burst,n_burst)]).reshape(2, 1)
		t_vec = np.linspace(-n_burst, T_max, n_steps)
	
	n_ = np.round(np.logspace(0, 5, 50))
	x_ = np.zeros((50, N_max))
	
	for N in range(1, N_max+1):
		for i in range(n_.size):
			t_vec = np.linspace(0, 100, n_[i])
			x = Integrator_1(t_vec, x0)
			x_[i, N-1] = x[-1, 0, N-1]
		
	err = np.abs((x_true - x_)/x_true)
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_sym))
	
	data = np.hstack((n_.reshape(n_.size, 1), err))
	return data
	
def Save_data(data):
	######## Save Results #################
	n_ = data[:,0]
	err = data[:,1:]
	
	filename = Equation + "_error_vs_n_steps"
	file_path = "Data/" + filename + ".txt"
	
	f = open(file_path, "w") # empty file
	f.close()
	f = open(file_path, "w")
	
	f.write(time.strftime("%X") + "\n")
	f.write("time to do symbolics	= {}s \n".format(T_sym - T_start))
	f.write("time to do numerics	= {}s \n".format(T_num - T_sym))
	f.write("n_steps	err (1st ord.)	err (2nd ord.)	etc.  \n")
	for i in range(n_.size):
		f.write("{}	".format(n_[i]))
		for N in range(0, N_max):
			f.write("{}	".format(err[i, N]))
		f.write("\n")
	f.close()
	print("saved data")

def Open_data():
	####### Open data ####################
	filename = Equation + "_error_vs_n_steps"
	file_path = "Data/" + filename + ".txt"
	
	f = open(file_path, "r")
	data = np.loadtxt(f, skiprows=4, delimiter='	'))
	print("read data")
	return data

#######################################

NEW == True

if NEW == True:
	data = Make_new_data()
	Save_data(data)
elif NEW == False:
	data = Open_data()

###### plot graph #####################

n_ = data[:,0]
err = data[:,1:]

colours = ['r-', 'b-', 'g-']
for N in range(N_max):
	plt.plot(n_, np.log10(err[:,N]), colours[N], linewidth=1, label="order"+str(N))
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

	