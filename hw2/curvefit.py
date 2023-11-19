# ECS130 HW2 -- Curve Fitting
import numpy as np
import linear_systems, qr

def vandermonde_matrix(d, x):
    """ Constructs a Vandermonde matrix for the sample coordinates in vector `x` and degree `d`. """

    # TODO (Problem 5): Implement the Vandermonde matrix construction
    numRows = len(x)
    numCols = d+1
    vandermonde = np.zeros((numRows, numCols))
    for i in range(numRows):
        for j in range(numCols):
            vandermonde[i, j] = x[i] ** j
    return vandermonde


def evaluate_polynomial(a, x):
    """ Evaluates the polynomial with coefficients `a` at the sample coordinates in vector `x`. """
    # TODO (Problem 6): Implement the polynomial evaluation
    y = np.zeros(len(x))
    for i, xi in enumerate(x):
        yi = 0
        for j, aj in enumerate(a):
            yi += aj * (xi**j)
        y[i] = yi
    return y

def curvefit_cholesky(x, y, d):
    """
    Fits a polynomial of degree `d` to the data points in `x` and `y` using the
    Cholesky factorization approach, returning the coefficients of the
    polynomial in a vector `a`.
    """
    # TODO (Problem 7): Implement the Cholesky factorization approach
    a = np.zeros(d)
    V = vandermonde_matrix(d, x)
    VtV = np.matmul(V.T,V)
    Vty = np.matmul(V.T,y)
    a = linear_systems.solve_cholesky(VtV, Vty)
    return a
    

def curvefit_qr(x, y, d, factorizer=qr.modified_gram_schmidt):

    """
    Fits a polynomial of degree `d` to the data points in `x` and `y` using the
    QR factorization approach, returning the coefficients of the polynomial in
    a vector `a`.
    """
    return qr.least_squares(vandermonde_matrix(d, x), y, factorizer)

approaches = { 'Cholesky': curvefit_cholesky }

# Create a curve fitting approach for each QR variant.
approaches.update({method: lambda x, y, d, m=method: curvefit_qr(x, y, d, m) for method in qr.methods})

################################################################################
# Test each method on a collection of data sets by printing the calculated
# coefficients and plotting the resulting polynomials.
################################################################################
if __name__ == '__main__':
    import matplotlib, numpy as np
    matplotlib.use('Agg') # use a DISPLAY-free backend
    from matplotlib import pyplot as plt
    import subprocess as sp

    np.set_printoptions(linewidth=200, edgeitems=100)

    def writeFunctionSamples(name, x, f):
        np.savetxt(name, x, f(x))

    datasets = {
        'linear':           [np.linspace(0.4, 0.6, 25),    lambda x: x],
        'quadratic':        [np.linspace(0, 1,  25),       lambda x: 0.5 * x * x],
        'noisy_quadratic':  [np.linspace(0, 1,  15),       lambda x: 0.5 * x * x + np.random.normal(scale=1e-2, size=x.size)],
        'trig':             [np.linspace(0, 1,  25),       lambda x: np.cos(4 * x)],
        'trig_exp':         [np.linspace(0, 1, 100),       lambda x: np.exp(np.cos(4 * x))],
        'lecture_8':        [np.linspace(-0.25, 1.25, 11), lambda x: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]]
    }

    # Parse command line arguments.
    # Example usage: python curvefit.py 3 --example trig --method ModifiedGramSchmidt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('degree', type=int, help='degree of polynomial to fit')
    parser.add_argument('--example', type=str, default=None, help='example data set to fit')
    parser.add_argument('--method', type=str, default=None, help='method to use for fitting')
    args = parser.parse_args()

    examples = args.example.split(';') if args.example else datasets.keys()
    methods  = args.method.split(';')  if args.method  else approaches.keys()
    degree   = args.degree

    for name in examples:
        if name not in datasets:
            print(f"Invalid example name '{name}'; must be one of ", list(datasets.keys()))
            continue

        x, f = datasets[name]
        sample_data = np.column_stack([x, f(x)])

        print(f'Data set {name}:')

        plt.xlabel('x')
        plt.ylabel('y')

        plt.plot(*sample_data.T, 'x', markersize=5)

        for method_name in methods:
            if method_name not in approaches:
                print(f"Invalid method name '{method_name}'; must be one of ", list(approaches.keys()))
                continue

            fitter = approaches[method_name]
            eval_x = np.linspace(-0.25, 1.25, 1000)
            coeffs = fitter(*sample_data.T, degree)
            if np.isnan(coeffs).all(): continue # Skip unimplemented methods.
            eval_y = evaluate_polynomial(coeffs, eval_x)

            plt.plot(eval_x, eval_y, label=method_name)

            ground_truth = sample_data[:, 1]
            eval_y_sample = evaluate_polynomial(coeffs, sample_data[:, 0])
            print(f'  {method_name}:  {coeffs}; error = {np.linalg.norm(ground_truth - eval_y_sample) / np.linalg.norm(ground_truth)}')

        plt.grid()
        plt.legend()
        plt.title(f'Degree {degree} Polynomial Fit to Data Set {name}')
        plt.tight_layout()
        plt.savefig(f'result_{name}_deg_{degree}.pdf')
        plt.close()
