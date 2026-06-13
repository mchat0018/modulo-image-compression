import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def solve_modulo_cs_milp(A, z, v_bound=1000):
    """
    Solve the MILP optimization problem

    Parameters:
    A : (m, n) measurement matrix
    z : (m) modulo-1 measurements (values in [0,1))
    v_bound : large but finite bound for integer variables v
               Adjust based on expected magnitude of A*x.

    Returns:
    x_opt : reconstructed sparse vector
    v_opt : integer part
    res   : milp solution object
    """
    m, n = A.shape

    # Decision variables: [x+ (n), x- (n), v (m)]
    c = np.concatenate([np.ones(n), np.ones(n), np.zeros(m)])

    # Equality constraint: A*x+ - A*x- - v = z
    A_eq = np.hstack([A, -A, -np.eye(m)])
    b_eq = z

    # Bounds: x+, x- >= 0; v in [-v_bound, v_bound] (integer)
    lb = np.concatenate([np.zeros(2 * n), -v_bound * np.ones(m)])
    ub = np.concatenate([np.inf * np.ones(2 * n), v_bound * np.ones(m)])
    bounds = Bounds(lb, ub)

    # Integrality: 0 for continuous variables (x+, x-), 1 for v
    integrality = np.array([0] * (2 * n) + [1] * m)

    # Solve
    res = milp(c=c,
               constraints=LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
               integrality=integrality,
               bounds=bounds,
               options={'disp': True})  # show solver output

    if res.success:
        x_opt = res.x[:n] - res.x[n:2 * n]  # x = x+ - x-
        v_opt = res.x[2 * n:]
        return x_opt, v_opt, res
    else:
        print("MILP failed:", res.message)
        return None, None, res

# Generate test data (as in the paper)
N, s, m = 784, 15, 31   # m = 2s+1 = 31
A = np.random.randn(m, N) / np.sqrt(m)   # Gaussian measurements
print(f"A = {A}")
x_true = np.zeros(N)
support = np.random.choice(N, s, replace=False)
x_true[support] = np.random.uniform(-1, 1, s)
z = (A @ x_true) % 1    # modulo-1 measurements

# Solve
x_rec, v_rec, res = solve_modulo_cs_milp(A, z, v_bound=256)
# print("Reconstruction error:", np.linalg.norm(x_rec - x_true))
print(f"x_rec = {x_rec}")
print(f"x is {np.linalg.norm(x_rec, ord=0)}-sparse and {len(x_rec)} entries")