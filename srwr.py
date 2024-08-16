import iterator
import normalizer
import reader
class SRWR:
    normalized = False

    def __init__(self):
        pass

    def read_graph(self, input_path):
        '''
        Read a graph from the given input path
        This function performs the normalization as well

        inputs
            input_path: string
                path for input file
        '''

        self.A, self.base = reader.read_graph(input_path)
        self.d = abs(self.A).sum(axis=1)
        self.normalize()

    def normalize(self):
        '''
        Normalize the given graph
        '''

        if self.normalized is False:
            self.nAp, self.nAn = normalizer.semi_row_normalize(self.A)
            self.nApT = self.nAp.T
            self.nAnT = self.nAn.T
            self.normalized = True

    def query(self, seed, c=0.15, epsilon=1e-9, beta=0.5, gamma=0.5,
              max_iters=300, handles_deadend=True, verbose=True):
        '''
        Compute an SRWR query for given seed

        inputs
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

        seed = seed - self.base

        rd, rp, rn, residuals = iterator.iterate(self.nApT, self.nAnT, seed, c,
                                                 epsilon, beta, gamma,
                                                 max_iters, handles_deadend,
                                                 verbose)

        return rd, rp, rn, residuals
