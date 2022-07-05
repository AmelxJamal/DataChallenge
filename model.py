import pandas as pd
import numpy as np
# import cvxopt

np.random.seed(20)


# Model

def solveRR(y, X, lam):
    n, p = X.shape
    assert (len(y) == n)
    
    A = X.T.dot(X)
    # Adjust diagonal due to Ridge
    # A[np.diag_indices_from(A)] += lam * n
    A += n * lam * np.eye(p)
    b = X.T.dot(y)
    beta = np.linalg.solve(A, b)
    
    return (beta)


#Weighted Ridge Regression

def solveWRR(y, X, w, lam):
    n, p = X.shape
    assert (len(y) == len(w) == n)

    w_sqrt = np.sqrt(w)   
    y1 = w_sqrt * y
    X1 = X * w_sqrt[:, None] 
    beta = solveRR(y1, X1, lam)

    return (beta)
    

def solveLRR_newton(y, X, lam, max_iter=500, eps=1e-3):
    n, p = X.shape
    assert (len(y) == n)
    
    # Parameters
    max_iter = 500
    eps = 1e-3
    sigmoid = lambda a: 1/(1 + np.exp(-a))
    
    # Initialize
    beta = np.zeros(p)
            
    # Hint: Use IRLS
    for i in range(max_iter):
        beta_old = beta
        f = X.dot(beta_old)
        w = sigmoid(f) * sigmoid(-f)
        z = f + y / sigmoid(y*f)
        beta = solveWRR(z, X, w, 2*lam)
        # Break condition (achieved convergence)
        if np.sum((beta-beta_old)**2) < eps:
            break
    return (beta)




# def cvxopt_qp(P, q, G, h, A, b):
#     P = .5 * (P + P.T)
#     cvx_matrices = [
#         cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
#     ]
#     #cvxopt.solvers.options['show_progress'] = False
#     solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
#     return np.array(solution['x']).flatten()

# solve_qp = cvxopt_qp


def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the RBF kernel with parameter sigma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    sigma: float
    '''
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K



def laplace_kernel(X1, X2, gamma=1):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the laplace kernel with parameter gamma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    gamma: float
    '''
    diff = X1[:, None]-X2[None, :]
    # print(f'differenece shape {diff.shape}')
    diff_norm = np.sum(diff ** 2, axis = -1)
    # print(f'differenece norm shape {diff_norm.shape}')
    K = np.exp(- gamma * np.sqrt(diff_norm))
    return K

def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return X1@X2.T

def polynomial_kernel(X1, X2, degree=2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the polynomial kernel of degree `degree`
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return (X1@X2.T +1)**degree

def add_column_ones(X):
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1))])

class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    fit_K
    predict_K
    '''
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        'laplace': laplace_kernel,
        # 'mismatch': mismatch_kernel,
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        self.fit_intercept_ = False
        
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', 1.)
        if self.kernel_name == 'polynomial':
            params['degree'] = kwargs.get('degree', 2)
        if self.kernel_name == 'laplace':
            params['gamma'] = kwargs.get('gamma', 1.)
        return params

    def fit_K(self, K, y, **kwargs):
        pass
    def decision_function_K(self, K):
        pass
    
    def fit(self, X, y, fit_intercept=False, **kwargs):

        if fit_intercept:
            X = add_column_ones(X)
            self.fit_intercept_ = True
        self.X_train = X
        self.y_train = y

        K = self.kernel_function_(self.X_train, self.X_train, **self.kernel_parameters)

        return self.fit_K(K, y, **kwargs)
    
    def decision_function(self, X):

        if self.fit_intercept_:
            X = add_column_ones(X)

        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)

        return self.decision_function_K(K_x)

    def predict(self, X):
        pass
    
    def predict_K(self, K):
        pass

class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit_K(self, K, y, sample_weights=None):
        n = K.shape[0]
        assert (n == len(y))
        
        w_sqrt = np.ones_like(y) if sample_weights is None else sample_weights
        w_sqrt = np.sqrt(w_sqrt)

        # Rescale kernel matrix K to take weights into account
        A = K * w_sqrt * w_sqrt[:, None]
        # Add lambda to the diagonal of A (= add lambda*I):
        A += n * self.lambd * np.eye(n)
        # self.alpha = (K + n * lambda I)^-1 y
        self.alpha = w_sqrt * np.linalg.solve(A , y * w_sqrt)

        return self
    
    def decision_function_K(self, K_x):
        return K_x@self.alpha
    
    def predict(self, X):
        return self.decision_function(X)
    
    def predict_K(self, K_x):
        return self.decision_function_K(K_x)

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
# tanh version helps avoid overflow problems
    return .5 * (1 + np.tanh(.5 * x))

class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):

        self.lambd = lambd
        super().__init__(**kwargs)
        
    
    def fit_K(self, K, y, method='gradient', lr=0.1, max_iter=500, tol=1e-12):
        '''
        Find the dual variables alpha
        '''
        if method == 'gradient':
            self.fit_alpha_gradient_(K, y, lr=lr, max_iter=max_iter, tol=tol)
        elif method == 'newton':
            self.fit_alpha_newton_(K, y, max_iter=max_iter, tol=tol)
            
        return self
        
    def fit_alpha_gradient_(self, K, y, lr=0.01, max_iter=500, tol=1e-6):
        '''
        Finds the alpha of logistic regression by gradient descent
        
        lr: learning rate
        max_iter: Max number of iterations
        tol: Tolerance wrt. optimal solution
        '''
        n = K.shape[0]
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            M = y*sigmoid(-y*K@alpha)
            gradient = -(1/n) *K@M +2*self.lambd*K@alpha
            alpha = alpha_old - lr * gradient
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol**2:
                break
        self.n_iter = n_iter
        self.alpha = alpha

    def fit_alpha_newton_(self, K, y, max_iter=500, tol=1e-6):
        '''
        Finds the alpha of logistic regression by the Newton-Raphson method
        and Iterated Least Squares
        '''
        n = K.shape[0]
        # IRLS
        KRR = KernelRidgeRegression(lambd=2*self.lambd)
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            m = K.dot(alpha_old)
            w = sigmoid(m) * sigmoid(-m)
            z = m + y / sigmoid(y * m)
            alpha = KRR.fit_K(K, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol**2:
                break
        self.n_iter = n_iter
        self.alpha = alpha
        
    def decision_function_K(self, K_x):
        # print('K', K_x.shape, 'alpha', self.alpha.shape)
        return sigmoid(K_x@self.alpha)
        
    def predict(self, X):
        return np.sign(2*self.decision_function(X)-1)

