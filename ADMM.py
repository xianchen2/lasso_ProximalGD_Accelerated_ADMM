#=================================
#			ADMM
#=================================

def proxf(AtA,v,Atb,RELTOL,n):
	'''
	1/2||Ax-b||_2^2
	'''
	return np.linalg.inv(RELTOL*np.identity(n)+AtA)*(Atb+0.01*v)

def ADMM(Max_iter,lamdaK,gamma,AtA,Atb,A,b,ABSTOL,RELTOL,n):
    
	ldr = 1/lamdaK
	x = np.matrix(np.random.randn(n,1))
	z = np.matrix(np.random.randn(n,1))
	u = np.matrix(np.random.randn(n,1))
    
    
	obj = []
	for k in range(Max_iter):
		# update x
		x_1 = proxf(AtA,(z-u),Atb,RELTOL,n)
		# update z
		z_1 = proxop(x_1+u,(lamdaK*gamma)/RELTOL) 
		# update u
		u_1 = u + RELTOL*(x_1 - z_1)
        
		ls = f(A,x_1,b)+ gamma*np.sum(np.abs(z_1))
		obj.append(ls)
        
# 		# terminating condition
# 		xz_norm = np.linalg.norm(x_1-z_1)
# 		zz_norm = np.linalg.norm(-ldr*(z_1 - z))
# 		eps_pri = np.sqrt(n)*ABSTOL + RELTOL*np.maximum(np.linalg.norm(x_1),np.linalg.norm(-z_1))
# 		eps_dual = np.sqrt(n)*ABSTOL + RELTOL*np.linalg.norm(ldr*u_1)
# 		if xz_norm < eps_pri and zz_norm < eps_dual:
# 			break

		lsold = f(A,x,b)+ gamma*np.sum(np.abs(z))
		if k > 1 and np.linalg.norm(ls - lsold) < ABSTOL:
			break

		z = z_1
		x = x_1
		u = u_1        

	return x_1, obj

if __name__ == '__main__':
	lamdaK = 1
	gamma =  0.01

	x_1,obj3 = ADMM(Max_iter,lamdaK,gamma,AtA,Atb,A,b,ABSTOL,RELTOL,n)