import os
import random
import re
import json
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
import time

from model import Graphormer
from parameter import parse_args
from tmp import generate_data, genWalk

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(device)

def train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, dataset_name, i):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, dataset_name, i):
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)

def get_data(model, dataset_name, i):
    file = f'./train/train.txt'
    test_file = f'./test/test.txt'
    train_data = load_data(file)
    test_data = load_data(test_file)
    offset=4
    train_edge_index, train_edge_attr = process_edges(train_data)
    train_pos_edge_index = train_edge_index[:, train_edge_attr >=offset ]
    train_neg_edge_index = train_edge_index[:, train_edge_attr <offset]

    test_edge_index, test_edge_attr = process_edges(test_data)
    test_pos_edge_index = test_edge_index[:, test_edge_attr >=offset]
    test_neg_edge_index = test_edge_index[:, test_edge_attr <offset]

    train_pos_edge_index = train_pos_edge_index.to(device)
    train_neg_edge_index = train_neg_edge_index.to(device)
    test_pos_edge_index = test_pos_edge_index.to(device)
    test_neg_edge_index = test_neg_edge_index.to(device)
    pos_edge_index=torch.cat((train_pos_edge_index,test_pos_edge_index),dim=0)
    neg_edge_index=torch.cat((train_neg_edge_index,test_neg_edge_index),dim=0)
    print("pos_edge_index",pos_edge_index)
    print("neg_edge_index",neg_edge_index)
    max_node_value = max(torch.max(train_edge_index).item(), torch.max(test_edge_index).item())

    x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index, max_node_value + 1)


    return train_pos_edge_index, test_pos_edge_index, train_neg_edge_index, test_neg_edge_index, x


def Search(args, dataset_name, seed_list):
    aucs = []
    accs = []
    for i in range(1):
        torch.manual_seed(seed_list[i])
        model = Graphormer(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        train_pos_edge_index, test_pos_edge_index, train_neg_edge_index, test_neg_edge_index,  x = get_data(model, dataset_name, i)

        total_time_start = time.time()

        for epoch in range(2000):
            loss = train(model, optimizer, x, train_pos_edge_index, train_neg_edge_index, dataset_name, i)
            auc, f1, acc = test(model, x, train_pos_edge_index, train_neg_edge_index, test_pos_edge_index, test_neg_edge_index, dataset_name, i)
            print(f'Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f},ACC:{acc:.4f}')
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        print(f'Total Time: {total_time:.2f}s, Loss: {loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f},ACC:{acc:.4f}')
        aucs.append(auc)
        accs.append(acc)
        filename = f"./ablation/{args.num_layers}-{args.output_dim}-{args.max_degree}-{i}.json"
        params = {
        'num_layer': args.num_layers,
        'output_dim': args.output_dim,
        'max_degree': args.max_degree,
        'numbers':i,
        'auc':auc,
        'acc':acc
        }
        save_params(params, filename)
    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accs)
    std_auc = np.std(aucs)
    std_acc = np.std(accs)
    return mean_auc, mean_acc, std_auc, std_acc,total_time

def load_data(file_path):
    if file_path.endswith('.txt'):
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data


def process_edges(data):
    # Ensure all values are stripped of leading/trailing whitespace
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Convert the first two columns to int and the third column to float
    edge_index = torch.tensor(data.iloc[:, :2].values.astype(np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(data.iloc[:, 2].values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

if __name__ == '__main__':
    args = parse_args()
    seed_list = [1145, 14, 68, 9810, 187]

    # param
    num_layers_list = [1]
    output_dim_list = [128]
    max_degree_list = [12]
    dataset1_name = "amazon-music"
    for num_layers in num_layers_list:
        for output_dim in output_dim_list:
            for max_degree in max_degree_list:
                args.num_layers = num_layers
                args.output_dim = output_dim
                args.max_degree = max_degree
                auc_mean,acc_mean,auc_std,acc_std,total_time = Search(args,dataset1_name, seed_list)
    print(f"{dataset1_name}'s auc_mean: {auc_mean},auc_std:{auc_std}")
    print(f"{dataset1_name}'s acc_mean: {acc_mean},acc_std:{acc_std}")
    print(f"{dataset1_name}'s total_time: {total_time}")

