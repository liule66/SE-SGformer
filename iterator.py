import numpy as np
from tqdm import tqdm
from numpy.linalg import norm


def iterate(nApT, nAnT, seed, c, epsilon, beta, gamma, max_iters,
            handles_deadend, verbose):
    '''
    Perform power iteration for SRWR query

    inputs
        nApT: csr_matrix
            positive semi row-normalized adjacency matrix (transpose)
        nAnT: csr_matrix
            negative semi row-normalized adjacency matrix (transpose)
        seed: int
            seed (query) node
        c: float
            restart probability
        epsilon: float
            error tolerance for power iteration
        beta: float
            balance attenuation factor
        gamma: float
            balance attenuation factor
        max_iters: int
            maximum number of iterations for power iteration
        handles_deadend: bool
            if true, it will handle the deadend issue in power iteration
            otherwise, it won't, i.e., no guarantee for sum of SRWR scores
            to be 1 in directed graphs
        verbose: bool
            if true, it will show a progress bar over iterations

    outputs:
        rd: ndarray
            relative trustworthiness score vector w.r.t. seed
        rp: ndarray
            positive SRWR vector w.r.t. seed
        rn: ndarray
            negative SRWR vector w.r.t. seed
        residuals: list
            list of residuals of power iteration,
            e.g., residuals[i] is i-th residual
    '''

    m, n = nApT.shape
    q = np.zeros((n, 1))
    q[seed] = 1.0

    rp = q
    rn = np.zeros((n, 1))
    rt = np.row_stack((rp, rn))

    residuals = np.zeros((max_iters, 1))

    # pbar = tqdm(total=max_iters, leave=True, disable=not verbose)
    for i in range(max_iters):
        if handles_deadend:
            new_rp = (1 - c) * (nApT.dot(rp + (1.0 - gamma) * rn) +
                                beta * (nAnT.dot(rn)))
            new_rn = (1 - c) * (gamma * (nApT.dot(rn)) +
                                nAnT.dot(rp + (1.0 - beta) * rn))
            P = np.sum(new_rp) + np.sum(new_rn)
            new_rp = new_rp + (1.0 - P) * q
        else:
            new_rp = (1 - c) * (nApT.dot(rp + (1.0 - gamma) * rn) +
                                beta * (nAnT.dot(rn))) + c * q
            new_rn = (1 - c) * (gamma * (nApT.dot(rn)) +
                                nAnT.dot(rp + (1.0 - beta) * rn))

        new_rt = np.row_stack((new_rp, new_rn))

        residuals[i] = norm(new_rt - rt, 1)

        # pbar.set_description("Residual at %d-iter: %e" % (i, residuals[i]))
        if residuals[i] <= epsilon:
            # pbar.set_description("SRWR scores have converged")
            # pbar.update(max_iters)
            break

        rp = new_rp
        rn = new_rn
        rt = new_rt

    rd = rp - rn

    return rd, rp, rn, residuals
