import numpy as np

import numpy as np

def omp(A, y, tol=1e-6):
    """
    Orthogonal Matching Pursuit (OMP) for sparse recovery.

    Solves: min ||z||_0  subject to  y = A z  (or approximately)
    using greedy selection of atoms.

    Parameters
    ----------
    A : np.ndarray, shape (M, N)
        Measurement matrix (usually fat, M < N).
    y : np.ndarray, shape (M,)
        Measurement vector.
    tol : float, optional
        Stopping tolerance on the residual norm ||r||_2.
        Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. If not provided, defaults to M
        (the number of measurements). Usually you would set it to the
        expected sparsity level.

    Returns
    -------
    z : np.ndarray, shape (N,)
        Reconstructed sparse vector.
    support : list
        Indices of the selected atoms (support set).
    """
    M, N = A.shape

    max_iter = M   # as per the pseudo-code: "or when we complete M iterations"

    # Initialization
    z = np.zeros(N)
    Lambda = []          # support set
    r = y.copy()         # residual

    # Pre-compute A^T for efficiency (optional)
    At = A.T

    for _ in range(max_iter):
        # 1. Correlation step
        h = At @ r          # shape (N,)

        # 2. Select atom with maximum absolute correlation
        k = np.argmax(np.abs(h))

        # 3. Update support
        Lambda.append(k)

        # 4. Solve least squares on the current support:
        #    z_Lambda = argmin || y - A_Lambda * z_Lambda ||_2
        A_Lambda = A[:, Lambda]          # (M, |Lambda|)
        # Using lstsq for numerical stability (handles ill-conditioned cases)
        z_Lambda, _, _, _ = np.linalg.lstsq(A_Lambda, y, rcond=None)

        # 5. Update full vector z
        z = np.zeros(N)
        z[Lambda] = z_Lambda

        # 6. Update residual
        r = y - A @ z

        # 7. Check stopping criterion
        if np.linalg.norm(r, 2) <= tol:
            break

    return z, Lambda