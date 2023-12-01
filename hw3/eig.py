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
    '''for v in v_list:
        # Ensure v is a column vector and has the same number of elements as the columns in B
        if len(v) != B.shape[1]:
            raise ValueError("Dimension of v does not match the number of columns in B")
        
        v = np.array(v).reshape(-1, 1)
        
        # Apply the Householder reflector to each row of B
        for i in range(B.shape[0]):
            B[i, :] -= 2 * np.dot(v.T, B[i, :]) * v.flatten()
    
    for v in v_list:
    # Compute the Householder transformation
    #v = np.array(v).reshape(-1，1) # Ensure v is a column vector
        H = np.eye(B.shape[1]) - 2 * np.dot(v,v.T) / np.dot(v.T, v)
        B = np.dot(B,H)
    
    for v in v_list:
    # Compute the Householder transformation
    #v = np.array(v).reshape(-1，1) # Ensure v is a column vector
        H = np.eye(B.shape[1]) - 2 * np.dot(v,v.T) / np.dot(v.T, v)
        B = np.dot(B,H)'''
    '''for v in v_list:
        for i in range(len(B)):B[i] = qr.apply_householder(v,B[i])
    Q,R = qr.householder(B)
    y= B.T
    m,n = y.shape
    for V in v_list:
        for k in range(n):
            v_k=V
            y[k:] -= (2 * np.dot(v_k,y[k:])) * V
    B=y.T
    '''
    for v in v_list:
        v = np.array(v)  # Ensure v is a numpy array
        for k in range(B.shape[0]):
            # Apply reflector to the k-th column of B (which is B^T's row)
            B[k, :] -= 2 * np.dot(v, B[k, :]) * v
    
    return B

def qr_iteration(A, Q_accum):
    """
    Apply a single iteration of the QR Eigenvalue algorithm to symmetric
    matrix `A`, accumulating the iteration's Q factor to `Q_accum`
    """
    # TODO (Problem 2b): update A and Q_accum in-place!
    
    """Q,R = qr.householder(A)
    A[:] = right_multiply_Q(R, Q)
    Q_accum[:] =  right_multiply_Q(Q_accum, Q)
    
    Q, R = np.linalg.qr(A)   # Perform QR decomposition
    A[:] = np.dot(R, Q)      # Update A as RQ
    Q_accum[:] = np.dot(Q_accum, Q)  # Update Q_accum"""
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
    '''residuals = []
    m, n = A.shape
    for i in range(m - 1):
        # Check for convergence and perform deflation if needed
        if np.abs(A[-1, -2]) < tol:
            # Deflate A and continue with the smaller matrix
            A = A[:-1, :-1]
            continue
        # Compute the Rayleigh quotient shift
        mu = A[-1, -1]
        # Shift A
        A -= mu * np.identity(n)
        # Perform QR iteration
        qr_iteration(A, Q_accum)
        # Unshift A
        A += mu * np.identity(n)
        # Compute the residual (off-diagonal size)
        odiag = off_diag_size(A)
        residuals.append(odiag)

        # Check for convergence
        if odiag < tol:
            break
    
    '''
    n = A.shape[0]
    residuals = []
    
    for m_deflated in range(n, 1, -1):
        A_deflated = A[:m_deflated, :m_deflated]  # View of the top-left block

        # Run Rayleigh-quotient-shifted QR iteration on the submatrix A_deflated
        while np.max(np.abs(np.tril(A_deflated, -1))) > tol:
            mu = A_deflated[-1, -1]  # Get the Rayleigh quotient
            np.fill_diagonal(A_deflated, np.diagonal(A_deflated) - mu)  # Apply the shift
            qr_iteration(A_deflated, Q_accum[:, :m_deflated])
            np.fill_diagonal(A_deflated, np.diagonal(A_deflated) + mu)  # Undo the shift
            residuals.append(off_diag_size(A_deflated))

        # Padding Q_accum with identity matrix for the deflated portion
        I_pad = np.eye(n - m_deflated)
        Q_accum = np.dot(Q_accum, np.block([[np.eye(m_deflated), np.zeros((m_deflated, n - m_deflated))],
                                            [np.zeros((n - m_deflated, m_deflated)), I_pad]]))

    
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
