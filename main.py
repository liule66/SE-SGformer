import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import BitcoinOTC
import time

from model import SE_SGformer
from parameter import parse_args


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def train():
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(1147)
    dataset = BitcoinOTC(root = '/Data/BitcoinOTC')
    pos_edge_indices, neg_edge_indices = [], []
    for data in dataset:
        pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
        neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])

    pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
    neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)
    model = SE_SGformer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
    train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)

    x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, dataset.num_nodes+1)
    # x = torch.randn((dataset.num_nodes + 1, args.num_node_feasures)).to(device)

    file = f'./train/train_edges.csv'
    if not os.path.isfile(file):
        pos_data_tensor_cpu = train_pos_edge_index[0].cpu().numpy()
        pos_data_tensor_cpu = np.column_stack((pos_data_tensor_cpu, train_pos_edge_index[1].cpu().numpy()))

        neg_data_tensor_cpu = train_neg_edge_index[0].cpu().numpy()
        neg_data_tensor_cpu = np.column_stack((neg_data_tensor_cpu, train_neg_edge_index[1].cpu().numpy()))

        pos_data = np.column_stack((pos_data_tensor_cpu, np.ones(train_pos_edge_index.shape[1])))
        neg_data = np.column_stack((neg_data_tensor_cpu, -np.ones(train_neg_edge_index.shape[1])))

        all_data = np.vstack((pos_data, neg_data))

        np.savetxt(file, all_data, delimiter=',', fmt='%d', comments='')

        print(f"train_edges have been saved")

    total_time_start = time.time()

    for epoch in range(2000):
        epoch_time_start = time.time()

        loss = train()
        auc, f1 = test()
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start
        print(f'Epoch: {epoch:03d}, Time: {epoch_time:.2f}s, Loss: {loss:.4f}, AUC: {auc:.4f}, '
              f'F1: {f1:.4f}')

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print(f'Total Time: {total_time:.2f}s')
