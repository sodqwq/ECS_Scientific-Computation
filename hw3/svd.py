import eig as eig
import numpy as np

def svd(A, tol=1e-8, maxIterations=1000):
    """
    Construct the SVD of a potentially rectangular matrix A.
    """
    m, n = A.shape
    k = min(m, n)
    U, sigma, V = np.eye(m, k), np.zeros(k), np.eye(n, k) # TODO: Replace these with the actual SVD.
    # TODO (Problem 5): compute the "thin" SVD of A by performing an eigenvalue decomposition
    # of either A^T A or A A^T depending on the shape of A.

    return U, sigma, V

def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

import unittest
class TestCases(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-6):
        self.assertLess(relerror(a, b), tol)

    def test_svd(self):
        # Generate a random matrix.
        for i in range(100):
            m = np.random.randint(1, 30)
            n = np.random.randint(1, 30)
            A = np.random.normal(size=(m, n))
            k = min(m, n)
            U, sigma, V = svd(A)
            # Check that U is orthogonal.
            self.requireSame(U.T @ U, np.eye(k))
            # Check that V is orthogonal.
            self.requireSame(V.T @ V, np.eye(k))
            # Check that U, sigma, V reconstruct A.
            self.requireSame(A, U @ np.diag(sigma) @ V.T)

import unittest
if __name__ == '__main__':
    # Run the unit tests defined above.
    unittest.main()
