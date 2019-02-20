#=================================
# Accelerated proximal GD
#=================================

def AccProxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL):

	x = np.matrix(np.zeros((np.shape(A)[1],1)))
	xprev = x
	
	obj = []
	for k in range(Max_iter):
		
		y = x + (1/(k+3)) * (x-xprev)
		
		while True:
			x_1 = proxop(y-lamdaK*gradf(AtA,y,Atb),lamdaK*gamma)
			if f(A,x_1,b) <= uppbd(A,AtA,Atb,y,x_1,b,lamdaK):
				break
			else:
				lamdaK = beta*lamdaK

		obj.append(objective(A,x_1,b,gamma))

		# terminating condition
		if k > 1 and np.linalg.norm(objective(A,x_1,b,gamma) - objective(A,x,b,gamma)) < ABSTOL:
			break

		xprev = x
		x = x_1
	
	return x_1, obj

if __name__ == '__main__':
	lamdaK = 1
	beta = 0.5 #decreasing parameter for lambda
	gamma =  0.01

	x_1,obj2 = AccProxgd(Max_iter,lamdaK,gamma,AtA,Atb,A,beta,b,ABSTOL)
