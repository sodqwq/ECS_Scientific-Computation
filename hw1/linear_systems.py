# ECS130 HW1 -- Solving Linear Systems
import numpy as np, scipy
import matplotlib


def lu(A):
    """ Constructs an LU decomposition of `A` without pivoting. """
    m = A.shape[0]
    if A.shape[1] != m: 
        raise Exception('A is not square!')
    L = np.identity(m)
    U = A.copy()

    for k in range(m - 1):
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k + 1, m):
                U[i, j] = U[i, j]- L[i, k] * U[k, j]

    return L, U

def forwardsub(L, b):
    """ Solves "L x = b" with a lower triangular matrix L. """
    m = L.shape[0]
    if L.shape[1] != m: raise Exception('A is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')
    # TODO (Problem 8): Implement the forward substitution algorithm
    # https://julianpanetta.com/teaching/ECS130/05-SolvingLinearSystems-deck.html#/forward-substitution/9
    x = b.copy()
    for i in range(m):
        for j in range(i):
            x[i] =x[i]- L[i, j] * x[j]
        x[i] = x[i]/L[i, i]

    return x

def backsub(U, b):
    """ Solves "U x = b" with an upper triangular matrix U. """
    m = U.shape[0]
    if U.shape[1] != m: raise Exception('A is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')

    # TODO (Problem 9): Implement the back substitution algorithm
    # https://julianpanetta.com/teaching/ECS130/05-SolvingLinearSystems-deck.html#/back-substitution
    x = b.copy()
    for i in range(m - 1, -1, -1):
            x[i] = x[i] / U[i, i]
            for j in range(i - 1, -1, -1):
                x[j] = x[j] - U[j, i] *x[i]


    return x
    

def solve(A, b):
    """ Solve "A x = b" using the LU decomposition approach """

    # TODO (Problem 10): use the methods you implemented above to compute the
    # LU decomposition and then solve for `x` using its factors.
    L, U = lu(A)
    y = forwardsub(L, b)
    x = backsub(U, y)
    return x

def forwardsub_vectorized(L, b):
    """ Solves "L x = b" with a lower triangular matrix L. """

    # TODO (optional): Implement a vectorized version of `forwardsub`
    
    return forwardsub(L, b)

def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def syntheticFactors(m):
    L = np.random.uniform(size=(m, m))
    U = np.random.uniform(size=(m, m))
    L[np.triu_indices(m, 1)] = 0
    L[np.diag_indices(m)] = 1
    U[np.tril_indices(m, -1)] = 0
    return L, U

from scipy.linalg.blas import dtrsv
def forwardsub_scipy(L, b):
    return scipy.linalg.solve_triangular(L, b, lower=True, check_finite=False)

forwardsub_implementations = {'loop': forwardsub,
                              'scipy': forwardsub_scipy}

import unittest
class TestCases(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-12):
        self.assertLess(relerror(a, b), tol)

    def test_lu_slides_example(self):
        A = np.array([[2, 1, 1, 0],
                      [4, 3, 3, 1],
                      [8, 7, 9, 5],
                      [6, 7, 9, 8]], dtype=float)
        L, U = lu(A)
        self.requireSame(np.tril(L), np.array([[1., 0., 0., 0.], [2., 1., 0., 0.], [4., 3., 1., 0.], [3., 4., 1., 1.]]))
        self.requireSame(np.triu(U), np.array([[2., 1., 1., 0.], [0., 1., 1., 1.], [0., 0., 2., 2.], [0., 0., 0., 2.]]))

    def test_lu_synthetic(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            L, U = syntheticFactors(m)
            Lc, Uc = lu(L @ U)
            self.requireSame(np.tril(Lc), L)
            self.requireSame(np.triu(Uc), U)

    def test_forwardsub_synthetic(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = L @ x
            for name, fs in forwardsub_implementations.items():
                self.requireSame(x, fs(L, b))

    def test_backsub_synthetic(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = U @ x
            self.requireSame(x, backsub(U, b))

    def test_solve_synthetic(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            A = np.random.uniform(size=(m, m))
            x = np.random.normal(size=m)
            b = A @ x
            self.requireSame(x, solve(A, b))

if __name__ == '__main__':
    import sys
    if '-b' in sys.argv:
        # Benchmarking
        from matplotlib import pyplot as plt
        import timeit

        setup_command = lambda m, impl: f'from __main__ import syntheticFactors, forwardsub_implementations; import numpy as np; L, U = syntheticFactors({m}); b = np.random.normal(size={m}); forwardsub = forwardsub_implementations["{impl}"]'
        times = {}
        for impl in forwardsub_implementations:
            sizes = [16, 32, 64, 128, 200, 256, 400, 512, 756, 1024]
            if impl != 'loop': # Include larger matrices for the faster variants
                sizes.extend([1600, 2048, 3000, 4096])
            time = [min(timeit.repeat(f'forwardsub(L, b)', setup_command(m, impl), number=10, repeat=2)) for m in sizes]
            plt.loglog(sizes, time,  '-s', ms=4, label=impl)

        plt.xlabel('Matrix size $m$')
        plt.ylabel('Time per substitution (s)')
        plt.title('Forward Substitution Benchmarks')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('forwardsub_benchmarks.pdf')

    # Run the tests defined above.
    unittest.main()
