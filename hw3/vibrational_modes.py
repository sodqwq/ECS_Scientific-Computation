# ECS130 HW3 -- Modal Analysis for Truss Structures
import truss_simulation
import numpy as np
import visualization
import eig
import pickle
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 vibrational_modes.py num_modes <data file>")
        sys.exit(1)

    num_modes = int(sys.argv[1])
    path = sys.argv[2]

    data = pickle.load(open(path, 'rb'))

    name = data['name']
    X = data['X']
    E = data['E']
    C = np.array(data['C'])
    F = data['F']
    k = data['k']

    C = [] # TODO: make this optional
    F[:] = 0

    K = truss_simulation.construct_K(X, E, k)
    l, Q = eig.sorted_eigendecomposition(K, descending=False)
    print(l[:num_modes])

    for i in range(num_modes):
        U = -Q[:, i].reshape((-1, 2))
        visualization.visualizeTruss(X + 0 * U, E, C, F, newfig=True,  alpha=0.05)
        visualization.visualizeTruss(X + 1 * U, E, C, F, newfig=False, alpha=0.15)
        visualization.visualizeTruss(X + 2 * U, E, C, F, newfig=False, alpha=1.0)
        plt.title(f'Mode {i}, energy = {max(l[i], 0):0.3e}, frequency = {np.sqrt(max(l[i], 0)):.3f}')
        plt.tight_layout()
        plt.savefig(f'{name}_mode_{i}.pdf')
        plt.close()
