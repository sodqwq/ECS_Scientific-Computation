import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections  as mc

def visualizeTruss(X, E, C, F, annotate=False, axisLimits=None, newfig=True, alpha=1.0):
    """
    Visualizes a truss structure and its boundary conditions.

    Parameters
    ----------
    X : array_like
        An N x 2 array of node coordinates.
    E : array_like
        An M x 2 array of indices into X, representing the edges of the truss.
    C : array_like
        An array of indices into X, representing the nodes with fixed position.
    F : array_like
        An N x 2 array of forces acting on each node.
    annotate : bool, optional
        Whether or not to label each node with its index.
    axisLimits : array_like, optional
        The axis limits of the plot, as [xmin, xmax, ymin, ymax].
    """
    lc = mc.LineCollection(X[E], linewidths=3, color='dodgerblue', zorder=0, alpha=alpha)
    if newfig: plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.add_collection(lc)
    
    # Determine which nodes belong to which boundary condition types
    freeNodes = np.ones(len(X), dtype=bool)
    forceNodes = np.where(np.linalg.norm(F, axis=1))
    freeNodes[C] = False
    freeNodes[forceNodes] = False
    
    plt.scatter(X[freeNodes, 0], X[freeNodes, 1], zorder=1, color='k', s=60, alpha=alpha)
    plt.scatter(X[C, 0], X[C, 1], zorder=1, color='orange', s=60, alpha=alpha)
    plt.scatter(X[forceNodes, 0], X[forceNodes, 1], zorder=1, color='red', s=60, alpha=alpha)
    plt.quiver(X[:, 0], X[:, 1], F[:, 0], F[:, 1], width=0.005, scale=10, zorder=0, color='red', alpha=alpha)
    
    if annotate:
        for i, pt in enumerate(X):
            plt.annotate(i, pt)
            
    plt.axis('equal')
    plt.axis('off')
    if axisLimits is not None:
        ax.set_xlim(axisLimits[0], axisLimits[1])
        ax.set_ylim(axisLimits[2], axisLimits[3])
    plt.tight_layout()
    plt.draw()
    [xmin, xmax, ymin, ymax] = plt.axis()
    return [xmin, xmax, ymin, ymax]
