
import numpy as np 
from scipy.sparse import random as sprandn
import matplotlib.pyplot as plt

#===============================
#Problem set
m = 500 	#number of examples
n = 2500	#number of features
A = np.matrix(np.random.randn(m,n))
A = (A-np.mean(A,0))/np.std(A,0) # normalize columns
x_r = sprandn(n,1,density=0.05)
b = A*x_r + np.sqrt(0.001)*np.matrix(np.random.randn(m,1))

# cached computations
AtA  = A.T*A #n*n
Atb  = A.T*b #n*1

#Global constant and defaults
Max_iter = 100
ABSTOL = 1e-3
RELTOL   = 1e-2 #error tolerance for ADMM
#===============================


def objective(A,x,b,gamma):
	'''
	objective function: f(x) + g(x)
	f(x) = 1/2||Ax-b||^2
	g(x) = gamma*|x|
	A : independt variables m*n
	x : parameters n*1
	b : m*1
	''' 
	return f(A,x,b) + gamma*np.sum(np.abs(x))

def f(A,x,b):
	'''
	f(x) = 1/2||Ax-b||^2 m*1
	'''
	return 0.5*(np.linalg.norm(A*x-b)**2)

def gradf(AtA,x,Atb):
	'''
	gradient of f(X) n*1
	'''
	return AtA*x-Atb 

def uppbd(A,AtA,Atb,x,x_1,b,lamdaK):
	'''
	m*1
	'''
	xDiff = x_1 - x
	return f(A,x,b) + gradf(AtA,x,Atb).T*xDiff + 1.0/(2.0*lamdaK)* np.sum(np.multiply(xDiff,xDiff))
    
def proxop(v,lamdaK):
	'''
	calculate proximal operater of l1 norm
	lamdaK: learning rate
	'''
	zero = np.matrix(np.zeros(np.shape(v)))
	return np.multiply(np.sign(v),np.maximum(np.abs(v)-lamdaK,zero))