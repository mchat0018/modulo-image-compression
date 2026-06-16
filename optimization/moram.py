import numpy as np

class MoRAM:

    def __init__(self, cs_matrix, val_range, tol=1e-5):
        self.A = cs_matrix
        self.M = cs_matrix.shape[0]
        self.N = cs_matrix.shape[1]
        self.R = val_range
        self.tol = tol


    def initialize_bin_idx(self, y):
        return (y > self.R/2).astype('int')
    

    def orthogonal_matching_pursuit(self, y, A):
        # initializations
        basis_ind = []  # support vector indices
        r = y.copy()   # residual

        for k in range(self.M):
            
            i = np.argmax(np.abs(A.T @ r))  # selecting the column of A with the largest projection on the residual
            basis_ind.append(i) # adding this column to the set of basis vectors for the support set
            
            # getting the estimate of z defined on the existing support set
            A_s = A[:, basis_ind]
            z_s = np.linalg.lstsq(A_s, y, rcond=None)[0]
            
            z = np.zeros(self.N+self.M); z[basis_ind] = z_s

            # updating the residual
            r = y - A @ z

            # checking for convergence
            if np.linalg.norm(r) < self.tol: 
                break

        # return the first N indices of the estimate
        return z[:self.N]


    def justice_pursuit(self,y):
        # augmenting the CS matrix to form the model for sparse errors
        A = np.concatenate([self.A, self.R*np.eye(self.M)], axis=1)
        return self.orthogonal_matching_pursuit(y, A)


    def descent(self, y, max_iter=100):
        
        # initializing bin index
        p = self.initialize_bin_idx(y)

        for t in range(max_iter):
            # getting the translated measurement value for linear sparse recovery
            y_c = y - p*self.R
            # running the Justice Pursuit (JP) algorithm
            x = self.justice_pursuit(y_c)
            
            # checking for convergence
            norm_error = np.linalg.norm(y - (self.A @ x) - p*self.R) / (np.linalg.norm(y) + 1e-6)
            if norm_error < 0.05: 
                print("Early convergence in iteration {}".format(t))
                break

            # updating the estimate for bin index
            p = (np.ones(self.M) - np.sign(self.A @ x).astype('int')) / 2

        return x
    


class MultiScaleMoRAM:

    def __init__(self, cs_matrix, max_level=256, max_moram_iters=100, print_debug_errors=False):
        self.A = cs_matrix
        self.M = cs_matrix.shape[0]
        self.N = cs_matrix.shape[1]

        self.l_max = max_level
        self.max_moram_iters = max_moram_iters
        self.print_debug_errors = print_debug_errors


    def __get_order(self, alpha):
        if alpha == 0: return 0
        order = np.floor(np.log10(np.abs(alpha)))
        return order

    def recover(self, y):

        num_scales = ((np.log(self.l_max)/np.log(2)) + 1).astype('int')
        alpha_list = [2**(l)/self.l_max for l in range(num_scales)]
        x_prev = None

        for alpha in alpha_list:
            y_scaled = y*alpha
            A = alpha*self.A
            R = self.l_max*alpha
            tol = 10**self.__get_order(alpha)

            moRAM = MoRAM(A, R, tol)
            
            if x_prev is not None:
                z_pred = A @ x_prev
                k_est = np.floor(z_pred - y_scaled)
                y_l = y_scaled + k_est
            
            else: y_l = y_scaled

            x = moRAM.descent(y_l, max_iter=self.max_moram_iters)
            x_prev = x

            if self.print_debug_errors:
                print(f"AFTER {R}-FOLD-WIDTH SCALING, (relative) l2-norm error \
                       = {np.linalg.norm(y - (self.A @ x) - np.floor(self.A @ x)) / (np.linalg.norm(y) + 1e-6)}")

        return x