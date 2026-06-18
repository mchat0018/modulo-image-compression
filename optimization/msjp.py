import numpy as np
import pylops
from pylops.optimization.sparsity import fista
import pyproximal
from pyproximal.optimization.primal import ProximalGradient


class MultiScaleJusticePursuit:

    def __init__(self, y: np.ndarray, scales: list, A: np.ndarray, print_debug_errors=False):
        
        y_raw = y
        self.y = self.__center_modulo_measurements(y_raw)   # centering modulo measurements for robust no-fold initialization
        self.alpha_list = scales
        self.A = A
        self.M = A.shape[0]
        self.N = A.shape[1]

        self.print_debug_errors = print_debug_errors

    def __center_modulo_measurements(self, y:np.ndarray):
        y_centered = np.mod(y + 0.5, 1.0) - 0.5
        return y_centered

    def __error_diagnosis(self, y:np.ndarray, x_est:np.ndarray, e:np.ndarray, A:np.ndarray):
        print("||x||_1:", np.linalg.norm(x_est, 1))
        print("||e||_1:", np.linalg.norm(e, 1))
        print("number |e_i| > 0.25:", np.sum(np.abs(e) > 0.25))
        print(
            "fit residual:",
            np.linalg.norm(A @ x_est + e - y) / np.linalg.norm(y),
        )



    def basis_pursuit(self, y:np.ndarray, alpha:float, fista_config:dict) -> np.ndarray:
        A_op = pylops.MatrixMult(self.A)
        y_norm = y/alpha
        x, _, _ = fista(
            Op=A_op,
            y=y_norm,
            niter=fista_config.get('niter', 500),
            eps=fista_config.get('lam_x', 0.1),
            alpha=None,
            tol=fista_config.get('tol', 1e-6),
        )
        return x
    
    
    def justice_pursuit(self, y:np.ndarray, A:np.ndarray, alpha:float, x0:np.ndarray, fista_config:dict) -> tuple[np.ndarray, np.ndarray]:
        
        # Making the aggregated CS matrix for sparse error minimization
        A_op = pylops.MatrixMult(A)
        I_op = pylops.Identity(self.M)
        A_aggr = pylops.HStack([A_op, I_op])

        # smooth function in Proximal Gradient Descent 
        # f(x) = 0.5*||y-Ax-e||_2^2
        f = pyproximal.L2(
            Op=A_aggr,
            b=y,
            sigma=1.0
        ) 
        
        # Non-smooth L1 regularization term: 
        # g(x) = \lambda_x ||x||_1 + \lambda_e ||e||_1
        l1_weights = np.concatenate([
            fista_config['lam_x']*np.ones(self.N)*alpha,
            fista_config['lam_e']*np.ones(self.M)
        ])
        g = pyproximal.L1(sigma=l1_weights)

        # Running Fast Iterative Shrinkage Thresholding Algorithm
        z = ProximalGradient(
            proxf=f,
            proxg=g,
            epsg=1.0,
            x0=np.concatenate([x0, np.zeros(self.M)]),
            tau=None, # enable adaptive backtracking, since Lipschitz constant for f isn't known
            acceleration='fista',
            niter=fista_config.get('niter', 500),
            tol=fista_config.get('tol', 1e-6),
        )

        x = z[:self.N]; e=z[self.N:]
        return x, e

    
    def compute_robust_estimate(self, y:np.ndarray, A:np.ndarray, alpha:float, x_prev:np.ndarray, jp_iters:int=10, fista_config:dict={}) -> np.ndarray:
        
        z_pred = A @ x_prev  # using estimate obtained at previous scale for prediction
        k_init = np.round(z_pred - y) # getting an initial estimate of the integer value
        y_l = y + k_init  # getting the translated dense measurements to be used for sparse recovery

        k_prev = k_init
        for _ in range(jp_iters):
            # obtaining x and (fold-)error estimates using Justice Pursuit
            x_est, e = self.justice_pursuit(y_l, A, alpha, x0=x_prev, fista_config=fista_config)

            # diagnosing potential oversimplification
            if self.print_debug_errors: self.__error_diagnosis(y, x_est, e, A)

            # correct the fold estimate
            k_corr = np.round((A @ x_est) - y)
            if self.print_debug_errors: print("Mean unclipped fold update = {}".format((k_corr - k_prev).mean()))

            # the fold update suggested by the error, check for agreement among pixels with higher errors
            k_corr_from_error = k_prev - np.rint(e).astype('int')
            agree = (k_corr == k_corr_from_error) | (np.abs(e) <= 0.35)
            if self.print_debug_errors: print("{}/{} pixels are in agreement in with the error in fold-update".format(np.sum(agree), len(agree)))

            # clip to adjacent folds for stability (Zhao et. al.)
            k_corr_clipped = np.clip(k_corr, k_prev-1, k_prev+1)
            adjacent = k_corr == k_corr_clipped
            if self.print_debug_errors: print("{}/{} proposed fold updates were adjacent to each other".format(np.sum(adjacent), len(adjacent)))

            y_l = y + k_corr_clipped

            # terminate if fold estimate doesn't change
            if (k_corr_clipped == k_prev).all(): break

            k_prev = k_corr_clipped
            x_prev = x_est

        return x_est

    
    def recover_signal(self, jp_iters:int=10, fista_config:dict={}) -> np.ndarray:
        
        x_prev = None

        for l,alpha in enumerate(self.alpha_list):
            if self.print_debug_errors: print(f"RUNNING FISTA-BACKED JUSTICE PURSUIT FOR {512*alpha} FOLDS...")

            y_scaled = self.y[l,:]  # measurements obtained at alpha scale
            A_scaled = alpha*self.A # corresponding compressed sensing matrix

            if l == 0 and self.print_debug_errors: print("Centered modulo working correctly: {}".format(np.max(np.abs(y_scaled))<0.5))

            x = self.basis_pursuit(y_scaled, alpha, fista_config) if x_prev is None else \
                self.compute_robust_estimate(y_scaled, A_scaled, alpha, x_prev, jp_iters, fista_config)

            x_prev = x

            if self.print_debug_errors: 
                error = np.linalg.norm(
                    self.y[-1,:] - self.__center_modulo_measurements((self.A @ x) - np.floor(self.A @ x))
                ) / (np.linalg.norm(self.y[-1,:]) + 1e-6)

                print(f"FOR {512*alpha} FOLDS, (relative) l2-norm error in measurements = {error:4f}")
                print("===============================================================================")

        return x
        
