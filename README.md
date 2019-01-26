# lasso_ProximalGD_Accelerated_ADMM
https://www.authorea.com/users/254721/articles/349097-lasso-proximal-accelerated-proximal-gradient-method-admm

minimize f(x) + g(x)

Proximal gradient descent:  minimize g(x) using proximal operator and performance gradient updates on f(x), which has convergence rate of O(1/k). 

Accelerated proximal gradient method:  include a momentum term to avoid overshooting with faster convergence rate of O(1/k^2). 

ADMM: Treats two functions, the objective and constraint, separately. 
The goal is to minimize f(x)+g(x) subject to (x = z). 
The u is the running sum of the errors(x - z). 


