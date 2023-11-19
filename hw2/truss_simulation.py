# ECS 130 HW2: Truss Simulation
import visualization
from matplotlib import pyplot as plt
import numpy as np
import pickle
import linear_systems

def construct_K(X, E, k):
    """
    Constructs the stiffness matrix K for a truss structure
    given the rest joint positions in the rows of X,
    edge endpoint indices in the rows of E,
    and the stiffness k of each edge.
    """
    numNodes = X.shape[0]  # referred to as n in the handout
    numEdges = E.shape[0]  # referred to as m in the handout
    # TODO (Problem 14): Construct the stiffness matrix K
    # Initialize matrix B
    B = np.zeros((2 * numNodes, numEdges))
    for e in range(numEdges):
        # Get the indices of the nodes for the e-th edge
        node_indices = E[e, :]
        ai, aj = node_indices[0], node_indices[1]

        # Compute vector for edge
        a = X[aj , :] - X[ai, :]
        length = np.linalg.norm(a)
        unit_a  = a / length

        # Place B for the start node
        B[2 * ai:2 * ai + 2, e] = unit_a
        B[2 * aj :2 * aj + 2, e] = -unit_a

    # BDB.T matrix multiplication = K
    BD = B * k
    K = BD @ B.T
    return K

def simulate(X, E, C, F, k):
    """
    Simulates the deformation of a truss structure given the rest joint positions
    in the rows of X, edge endpoint indices in the rows of E, indices of fixed
    joints in the array C, and the stiffness k of each edge.
    """
    numNodes = X.shape[0]  # referred to as n in the handout
    numEdges = E.shape[0]  # referred to as m in the handout
    # Construct the stiffness matrix K
    K = construct_K(X, E, k)
    K_tilde = K.copy()
    F_tilde = F.copy()
    for ci in C:
        #zeroing out rows and columns
        K_tilde[2 * ci:2 * ci + 2, :] = 0
        K_tilde[:, 2 * ci:2 * ci + 2] = 0
        #placing 1's on the diagonal for these rows
        K_tilde[2 * ci, 2 * ci] = 1
        K_tilde[2 * ci + 1, 2 * ci + 1] = 1

        F_tilde[2 * ci:2 * ci + 2] = 0

    # Solve the linear system  ̃KU =  ̃F
    U = linear_systems.solve_cholesky(K_tilde, F_tilde)
    return U


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 truss_simulation.py <data file>")
        sys.exit(1)

    data = pickle.load(open(sys.argv[1], 'rb'))

    name = data['name']
    X = data['X']
    E = data['E']
    C = np.array(data['C'])
    F = data['F']
    k = data['k']

    # Visualize the truss's rest configuration
    axisLimits = visualization.visualizeTruss(X, E, C, F)
    plt.savefig(f'{name}_init.pdf')
    plt.close()

    # Simulate the truss's deformation
    U = simulate(X, E, C, F.ravel(), k).reshape((-1, 2))
    x = X + U

    # Visualize the truss's deformed configuration
    visualization.visualizeTruss(x, E, C, F, axisLimits=axisLimits)
    plt.savefig(f'{name}_deformed.pdf')
    plt.close()
