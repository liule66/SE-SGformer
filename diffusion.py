import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from srwr import SRWR

import os
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Diffusion:
    """The method of generating diffusion graph for the training dataset"""

    def __init__(self, train_pos_edge_index, train_neg_edge_index, max_iters = 100, c = 0.15) -> None:
        self.count = 0
        self.max_iters = max_iters
        self.c = c
        self.input_path = "D:\\PycharmProjects\\Graphormr\\Data\\BitcoinOTC\\raw\\soc-sign-bitcoinotc.csv"
        # concat pos & neg training data to find the invovled nodes in training dataset
        edge_index = torch.concat((train_pos_edge_index, train_neg_edge_index), dim=1)  # shape (2, n)

        # select all node_id invovled in training dataset
        self.node_id_selected = torch.unique(edge_index).to(device)  # shape (m)
        self.m = len(self.node_id_selected)

        # signed random walk with restart
        self.srwr = SRWR()
        self.srwr.read_graph(self.input_path)  # read graph from input_path
        self.srwr.normalize()


    def single_node_srwr(self, seed, epsilon = 1e-9, beta = 0.5, gamma = 0.5, handles_deadend = True):
        """
        input_path : origin graph file path, input format: src(int)\tdst(int)\tsign(1/-1)
        P.S. original file is undirected, and I covert to direct in the "./srwr/reader.py"
        """

        # rp: relevance of pos; rn: relevance of neg
        _, rp, rn, _ = self.srwr.query(seed, self.c, epsilon, beta, gamma, self.max_iters, handles_deadend)

        self.count += 1
        print("\r" + f"{self.count} done", end="", flush=True)

        return rp.astype(np.float16), rn.astype(np.float16)


    def generate_diffusion_relevance_graph(self):
        """
        generate diffusion graph, and
        save the relevance data ( shape == (invovled_node_num, invovled_node_num) )
        save pos & neg edge index data respectively
        """

        # generate the probability matrix after SRWR
        # print(f"generating {self.percent}-0-{self.times}-d_training...")

        # srwr
        relevance_res = [self.single_node_srwr(node_id) for node_id in self.node_id_selected]
        print(f"\n direct srwr output data to undirected...")
        rp_mat, rn_mat = zip(*relevance_res)

        rp_mat = torch.tensor(np.concatenate(rp_mat, axis=1)).to(device)
        rn_mat = torch.tensor(np.concatenate(rn_mat, axis=1)).to(device)

        # remove the uninvovled node
        rp_mat = rp_mat[self.node_id_selected - 1]
        rn_mat = rn_mat[self.node_id_selected - 1]

        # rp = max(rp, rp.T)
        rp = torch.max(rp_mat, rp_mat.T)
        rn = torch.max(rn_mat, rn_mat.T)

        # rd = rp - rn
        diffusion_graph = rp - rn

        torch.save(diffusion_graph.cpu(), f"D:\PycharmProjects\Graphormr\srwr/d_training")


    def generate_diffusion_graph(self, thresholds_p = 0.025, thresholds_n = -0.0002):

        diffusion_file_path = f"D:\PycharmProjects\Graphormr\srwr\d_training"

        if not os.path.exists(diffusion_file_path):
            print(f"relevance file not exists, start generating...")
            self.generate_diffusion_relevance_graph()

        # diffusion relevance matrix
        diffusion_graph = torch.load(diffusion_file_path).to(device)

        # suitable index
        pos_idx = diffusion_graph >= thresholds_p
        neg_idx = diffusion_graph <= thresholds_n
        all_idx = pos_idx | neg_idx

        # diffusion graph adjacency matrix
        diffusion_graph[pos_idx] = 1
        diffusion_graph[neg_idx] = -1
        diffusion_graph[~all_idx] = 0  # else 0
        diffusion_graph = diffusion_graph.type(torch.int8)
        diffusion_graph = diffusion_graph.fill_diagonal_(0)  # remove self-loop

        # convert to Triad
        mask = torch.triu(torch.ones(diffusion_graph.shape), diagonal=1).bool().to(device)  # upper triangle mask matrix
        edge_index, edge_value = dense_to_sparse(diffusion_graph * mask)
        edge_index[0] = self.node_id_selected[edge_index[0]]
        edge_index[1] = self.node_id_selected[edge_index[1]]

        # concat to triad
        diffusion_graph = torch.concatenate((edge_index, edge_value.reshape(1, -1)), dim=0).T
        return diffusion_graph




    # def load_diffusion_data(self):
    #     """load the diffusion training data and split the pos and neg"""
    #
    #     diffusion_graph = torch.load(f"../../data/{self.percent}-0-{self.times}-d_training").to(device)
    #
    #     data_index = diffusion_graph[:, :2].T
    #     data_value = diffusion_graph[:, 2]
    #
    #     diff_pos_edge_index = data_index[:, data_value > 0]
    #     diff_neg_edge_index = data_index[:, data_value < 0]
    #
    #     return diff_pos_edge_index, diff_neg_edge_index


# if __name__ == "__main__":
#
#         Diffusion(train_pos_edge_index, train_neg_edge_index=).generate_diffusion_graph()

