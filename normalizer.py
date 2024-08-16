import numpy as np
from scipy.sparse import csr_matrix, spdiags, find
def semi_row_normalize(A):
    '''
    Perform the semi row-normalization for given adjacency matrix

    inputs
        A: csr_matrix
            adjacency matrix of given graph

    outputs
        nAp: csr_matrix
            positive semi row-normalized adjacency matrix
        nAn: csr_matrix
            negative semi row-normalized adjacency matrix
    '''

    m, n = A.shape

    # row-wise sum, d is out-degree for each node
    d = abs(A).sum(axis=1)
    d = np.asarray(d).flatten()

    d = np.maximum(d, np.ones(n))

    # 求D的逆
    invd = 1.0 / d
    invD = spdiags(invd, 0, m, n)
    snA = invD * A

    I, J, K = find(snA)

    pos = K > 0
    neg = K < 0

    nAp = csr_matrix((abs(K[pos]), (I[pos], J[pos])), shape=(m, n))
    nAn = csr_matrix((abs(K[neg]), (I[neg], J[neg])), shape=(m, n))

    return nAp, nAn
