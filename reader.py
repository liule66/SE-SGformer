import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
def read_graph(path):
    '''
    Read the signed network from the path

    inputs
        path: string
            path for input file

    outputs
        A: csr_matrix
            signed adjacency matrix
        base: int
            base number of node id (start from base)
    '''

    df = pd.read_csv(path, comment='#', delimiter=',', header=None, usecols=range(3), dtype=int)

    X = df.to_numpy()

    X[:, 2] = np.where(X[:, 2] < 0, -1, 1)

    # directed->undirected
    X = np.concatenate((X, X[:, (1, 0, 2)]))

    m, n = X.shape

    if n <= 2 or n >= 4:
        print('Invalid input format')

    base = np.amin(X[:, 0:2])
    # base = 1

    if base < 0:
        raise ValueError('Out of range of node id: negative base')

    X[:, 0:2] = X[:, 0:2] - base
    rows = X[:, 0]
    cols = X[:, 1]
    data = X[:, 2]

    n = int(np.amax(X[:, 0:2]) + 1)

    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    return A, int(base)
