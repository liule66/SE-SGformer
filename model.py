#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Union, Tuple, Dict, List, Optional
import scipy.sparse
import torch
import networkx as nx
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, structured_negative_sampling
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
    structured_negative_sampling,
)
from layer import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding, EdgeEncoding
from tmp import genWalk, generate_data
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class Graphormer(nn.Module):
    def __init__(self, args):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_pos_degree: max pos degree of nodes
        :param max_neg_degree: max neg degree of nodes

        """
        super().__init__()

        self.num_layers = args.num_layers
        self.num = args.num
        self.input_node_dim = args.num_node_features
        self.node_dim = args.node_dim
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.max_degree = args.max_degree

        self.length = args.length
        self.max_hop = args.max_hop
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)

        self.centrality_encoding = CentralityEncoding(
            max_degree=self.max_degree,
            node_dim=self.node_dim
        )
        self.spatial_matrix = SpatialEncoding(

            num=self.num,
        )

        self.adj_matrix = EdgeEncoding()

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                num_heads=self.num_heads,
            ) for _ in range(self.num_layers)
        ])
        self.lin = torch.nn.Linear(2 * args.output_dim, 3)
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def split_edges(
        self,
        edge_index: Tensor,
        test_ratio: float = 0.2,
    ) -> Tuple[Tensor, Tensor]:
        r"""Splits the edges :obj:`edge_index` into train and test edges.

        Args:
            edge_index (LongTensor): The edge indices.
            test_ratio (float, optional): The ratio of test edges.
                (default: :obj:`0.2`)
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def create_spectral_features(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Creates :obj:`in_channels` spectral node features based on
        positive and negative edges.

        Args:
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`pos_edge_index` and
                :attr:`neg_edge_index`. (default: :obj:`None`)
        """
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, num_nodes=N)
        val = val - 1

        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.input_node_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def forward(self, x: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, dataset_name: str, i: int) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        # file = f'./train/{dataset_name}-{i}-train_edges.csv'
        # train_edge_index, train_val, adj_unsigned, adj_signed, degree, offset = generate_data(file, x.shape[0])
        # feature = genWalk(train_edge_index, adj_signed, degree, offset, x.shape[0], self.length, self.max_hop, self.num)
        file = f'./train/train.txt'
        feature_file = f"./srwr/spatial_pos"
        if not os.path.isfile(feature_file):
            train_edge_index, train_val, adj_unsigned, adj_signed, degree, offset = generate_data(file, x.shape[0])
            spatial_pos = genWalk(train_edge_index, adj_signed, degree, offset, x.shape[0], self.length, self.max_hop, self.num)
            torch.save(spatial_pos, feature_file)

        feature = torch.load(feature_file).to(device)

        feature_abs  = torch.abs(feature)
        epsilon = 1e-10
        feature_reciprocal = torch.reciprocal(feature_abs + epsilon)
        row_sum = feature.sum(dim=2, keepdim=True)
        normalized_feature = (feature_reciprocal / row_sum).to(device)
        x = self.node_in_lin(x)
        x = self.centrality_encoding(x, pos_edge_index, neg_edge_index)

        spatial_matrix = self.spatial_matrix(normalized_feature).to(device)
        adj_matrix = self.adj_matrix(pos_edge_index, neg_edge_index, x.shape[0]).to(device)

        for layer in self.layers:
            x = layer(x, adj_matrix, spatial_matrix)

        x = self.node_out_lin(x)
        torch.save(x, f'./output/{dataset_name}-{i}-embedding.pt')
        return x

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
    ) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (torch.Tensor): The node embeddings.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes the overall objective.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + 5 * (loss_1 + loss_2)

    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[float, float,float]:
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        acc = accuracy_score(y, pred.round())

        return auc, f1,acc
