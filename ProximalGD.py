#================================
# Proximal Gradient Descent
#=================================
def proxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL):

	x = np.matrix(np.zeros((np.shape(A)[1],1)))
	
	obj = []
	for k in range(Max_iter):
		while True:
			x_1 = proxop(x-lamdaK*gradf(AtA,x,Atb),lamdaK*gamma)
			if f(A,x_1,b) <= uppbd(A,AtA,Atb,x,x_1,b,lamdaK):
				break
			else:
				lamdaK = beta*lamdaK

		obj.append(objective(A,x_1,b,gamma))

		# terminating condition
		if k > 1 and np.linalg.norm(objective(A,x_1,b,gamma) - objective(A,x,b,gamma)) < ABSTOL:
			break

		x = x_1

	return x_1, obj

if __name__ == '__main__':
	lamdaK = 1
	beta = 0.5 #decreasing parameter for lambda
	# gamma =  0.1*np.linalg.norm(Atb,np.inf) 
	gamma =  0.01

	x_1,obj= proxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL)