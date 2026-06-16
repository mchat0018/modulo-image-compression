import numpy as np

class MoRAM:

    def __init__(self, cs_matrix, val_range, omp_limit=100, tol=1e-5):
        self.A = cs_matrix
        self.M = cs_matrix.shape[0]
        self.N = cs_matrix.shape[1]
        self.R = val_range
        self.omp_limit = omp_limit
        self.tol = tol


    def initialize_bin_idx(self, y):
        return (y > self.R/2).astype('int')
    

    def orthogonal_matching_pursuit(self, y, A):
        # initializations
        basis_ind = []  # support vector indices
        r = y   # residual

        for _ in range(self.omp_limit):
            
            i = np.argmax(A.T @ r)  # selecting the column of A with the largest projection on the residual
            basis_ind.append(i) # adding this column to the set of basis vectors for the support set
            
            # getting the estimate of z defined on the existing support set
            A_s = A[:, basis_ind]
            z_s = A_s @ np.invert(A_s.T @ A_s) @ A_s.T @ r[basis_ind]
            z = np.zeros(self.N+self.M); z[basis_ind] = z_s

            # updating the residual
            r = y - A @ z

            # checking for convergence
            if np.linalg.norm(r) < self.tol: break

        # return the first N indices of 
        return z[self.N]


    def justice_pursuit(self,y):
        # augmenting the CS matrix to form the model for sparse errors
        A = np.stack([self.A, self.R*np.eye(self.M)], axis=1)
        return self.orthogonal_matching_pursuit(y, A)


    def descent(self, y, max_iter=100):
        
        # initializing bin index
        p = self.initialize_bin_idx(y)

        for t in range(max_iter):
            # getting the translated measurement value for linear sparse recovery
            y_c = y - p*self.R
            # running the Justice Pursuit (JP) algorithm
            x = self.justice_pursuit(y_c)
            # updating the estimate for bin index
            p = (np.ones(self.M) - np.sign(self.A @ x).astype('int')) / 2

        return x
    


class MultiScaleMoRAM:

    def __init__(self, cs_matrix, max_level=256, max_moram_iters=100, omp_limit=100, tol=1e-5):
        self.A = cs_matrix
        self.M = cs_matrix.shape[0]
        self.N = cs_matrix.shape[1]

        self.l_max = max_level
        self.max_moram_iters = max_moram_iters
        self.omp_limit = omp_limit
        self.tol = tol


    def recover(self, y):

        num_scales = (np.log(self.l_max)/np.log(2)) + 1
        alpha_list = [2**(l)/self.l_max for l in range(num_scales)]
        x_prev = None

        for l,alpha in enumerate(alpha_list):
            y_scaled = y*alpha
            A = alpha*self.A
            R = self.l_max*alpha
            moRAM = MoRAM(A, R, self.omp_limit, self.tol)
            
            if x_prev is not None:
                z_pred = A @ x_prev
                k_est = np.floor(z_pred - y_scaled)
                y_l = y_scaled + k_est
            
            else: y_l = y

            x = moRAM.descent(y_l, max_iter=self.max_moram_iters)
            x_prev = x

        return x