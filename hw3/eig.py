# ECS 130 HW3 -- Eigenvalue Decomposition via the QR Algorithm
import qr as qr
import numpy as np
from numpy.linalg import norm

def right_multiply_Q(B, v_list):
    """
    Replace the contents of `B` with the product `B Q`, where orthogonal matrix
    `Q` is represented implicitly by the list of Householder vectors `v_list`.
    """
    # TODO (Problem 2a): apply each Householder reflector in `v_list` to each *row* of `B`
    """y = B.T
    for k in range(len(v_list)):
        y[k:] -= 2 * np.outer(v_list[k],v_list[k] @ y[k:])
    B[:] = y.T"""
    
    for i in range(B.shape[0]):
        for k in range(len(v_list)):
            v = v_list[k]
            B[i,k:] -= 2 *v * np.dot(v,B[i,k:])
    
    return B
    
    

def qr_iteration(A, Q_accum):
    """
    Apply a single iteration of the QR Eigenvalue algorithm to symmetric
    matrix `A`, accumulating the iteration's Q factor to `Q_accum`
    """
    # TODO (Problem 2b): update A and Q_accum in-place!
 
    Q,R = qr.householder(A)
    A[:] = right_multiply_Q(R, Q)
    Q_accum[:] =  right_multiply_Q(Q_accum, Q)
    return Q_accum

def off_diag_size(A):
    """
    Compute the norm of the off-diagonal elements of a square matrix `A`.
    """
    return np.sqrt(2) * norm(A[np.triu_indices(A.shape[0], 1)])

def pure_qr(A, Q_accum, tol=1e-8, maxIterations=1000):
    """
    Run the simplest, barebones implementation of the QR algorithm
    (without shifts or deflation) to reduce `A` to a diagonal matrix
    via an orthogonal similarity transform that is multiplied into
    `Q_accum`.
    Iteration is terminated when the off-diagonal's relative magnitude
    shrinks below `tol` or when `maxIterations` iterations have been run
    (whichever comes first).
    """
    # Use the householder QR algorithm to compute the eigenvalues
    # and eigenvectors of the symmetric matrix A
    residuals = []
    for i in range(maxIterations):
        qr_iteration(A, Q_accum)
        odiag = off_diag_size(A)
        residuals.append(odiag)
        if odiag < tol:
            break
    return residuals

def full_qr(A, Q_accum, tol=1e-8):
    # TODO (Problem 3): Implement the QR algorithm with the Rayleigh quotient shift and deflation.
    # Also record the residuals at each step in `residuals` like done in `pure_qr` above.
    
    m = A.shape[0]
    residuals = []

    for m_deflated in range(m,1,-1):
        A_deflated = A[:m_deflated, :m_deflated]
        while norm(A_deflated[:(m_deflated - 1), m_deflated - 1]) > tol:
            mu = A_deflated[m_deflated-1 , m_deflated-1 ]
            np.fill_diagonal(A_deflated,A_deflated.diagonal() - mu)

            qr_iteration(A_deflated,Q_accum[ : ,:m_deflated])
            
            np.fill_diagonal(A_deflated,A_deflated.diagonal() + mu)
            residuals.append(off_diag_size(A))

    return residuals


def sorted_eigendecomposition(A, tol=1e-8, descending=True):
    """
    Compute the eigenvalue decomposition using `full_qr` and then sort
    the eigenvalues/permute the eigenvectors so that the diagonal of `A`
    is descending (like in the SVD) or ascending.
    """
    A = A.copy()
    m = A.shape[0]
    Q = np.eye(m)
    residuals = full_qr(A, Q, tol)
    p = np.argsort(np.diag(A))
    if descending: p = p[::-1]
    return np.diag(A)[p], Q[:, p]

import unittest
from matplotlib import pyplot as plt
import sys
import pickle, lzma

def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

class TestCases(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-8):
        self.assertLess(relerror(a, b), tol)

    def test_right_multiply_Q(self):
        right_multiply_Q_test_data = pickle.load(lzma.open('data/right_multiply_Q_test_data.pkl.lzma', 'rb'))
        for B_in, v_list_in, B_out in right_multiply_Q_test_data:
            right_multiply_Q(B_in, v_list_in)
            self.requireSame(B_in, B_out)

if __name__ == '__main__':
    # Run the unit tests defined above.
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=1)

    m = int(sys.argv[1]) if len(sys.argv) == 2 else 5

    # Generate a random symmetric matrix.
    A = np.random.normal(size=(m, m))
    A += A.T
    Q = np.eye(m)
    pure_qr_lambda = A.copy()
    pure_qr_Q = Q.copy()
    residuals_pure = pure_qr(pure_qr_lambda, pure_qr_Q)
    print(f'Computing eigendecomposition of a random symmetric {m}x{m} matrix...')
    print(f'Pure QR off-diagonal magnitude:\t{residuals_pure[-1]}')
    print(f'Pure QR reconstruction error:\t{norm(A - pure_qr_Q @ pure_qr_lambda @ pure_qr_Q.T)}')
    plt.semilogy(residuals_pure, label='pure_qr')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Off-diagonal magnitude')
    plt.title(f'QR Algorithm Convergence for a Random {m}x{m} Matrix')
    plt.savefig('residuals.pdf')
    plt.close()

    𝜦 = A.copy()
    residuals_full = full_qr(𝜦, Q)

    print(f'Full QR off-diagonal magnitude:\t{residuals_full[-1]}')
    print(f'Full QR reconstruction error:\t{norm(A - Q @ 𝜦 @ Q.T)}')

    np.set_printoptions(edgeitems=100, linewidth=1000)

    plt.semilogy(residuals_pure, label='pure_qr')
    plt.semilogy(residuals_full, label='full_qr')
    plt.xlabel('Iteration')
    plt.ylabel('Off-diagonal magnitude')
    plt.title(f'QR Algorithm Convergence for a Random {m}x{m} Matrix')
    plt.legend()
    plt.grid()
    plt.savefig('residuals_full_qr.pdf')
