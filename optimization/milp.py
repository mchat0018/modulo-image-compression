import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def solve_modulo_cs_milp(A, z, v_bound=30, mip_rel_gap=0.1):
    """
    Solve the modulo-CS optimization problem using a non-negative MILP formulation.
    Optimized specifically for non-negative signals like MNIST images (x >= 0).

    Parameters:
    A : (m, n) measurement matrix
    z : (m) modulo-1 measurements (values in [0,1))
    v_bound : finite bound for integer variables v (keep small, e.g., 20-50 for scaled data)
    mip_rel_gap : relative MIP gap tolerance for early termination (0.1 = 10%)

    Returns:
    x_opt : reconstructed sparse vector (non-negative)
    v_opt : integer part vector
    res   : milp solution object
    """
    m, n = A.shape

    # 1. Objective function coefficients (c):
    # We want to minimize ||x||_1. Since x >= 0, ||x||_1 is simply the sum of x.
    # Decision variables layout: [x (n continuous), v (m integers)]
    c = np.concatenate([np.ones(n), np.zeros(m)])

    # 2. Equality constraints: A*x - v = z  =>  [A  -I] * [x; v] = z
    A_eq = np.hstack([A, -np.eye(m)])
    b_eq = z

    # 3. Variable Bounds:
    # x must be non-negative: [0, inf]
    # v must be bounded integers: [-v_bound, v_bound]
    lb = np.concatenate([np.zeros(n), -v_bound * np.ones(m)])
    ub = np.concatenate([np.inf * np.ones(n), v_bound * np.ones(m)])
    bounds = Bounds(lb, ub)

    # 4. Integrality: 0 for continuous variables (x), 1 for integer variables (v)
    integrality = np.array([0] * n + [1] * m)

    # 5. Solve using Highs MILP solver via SciPy
    res = milp(c=c,
               constraints=LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
               integrality=integrality,
               bounds=bounds,
               options={'disp': True, 'mip_rel_gap': mip_rel_gap})

    if res.success:
        x_opt = res.x[:n]      # First n elements belong to x
        v_opt = res.x[n:]      # Remaining m elements belong to v
        return x_opt, v_opt, res
    else:
        print("MILP failed to find a feasible solution:", res.message)
        return None, None, res