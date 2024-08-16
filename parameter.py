#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
def parse_args():
    parser = argparse.ArgumentParser()  

    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment')
    parser.add_argument('--exp_name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate of AdamW')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of Graphormer layers')
    parser.add_argument('--num', type=int, default=4, help='The number of rank walker')
    parser.add_argument('--num_node_features', type=int, default=128, help='Hidden dimensions of node features')
    parser.add_argument('--node_dim', type=int, default=128, help='Hidden dimensions of node features')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--output_dim', type=int, default=128, help='Number of output node features')
    parser.add_argument('--max_degree', type=int, default=10, help='Max pos degree of nodes')
    parser.add_argument('--max_hop', type=int, default=7, help='Max distance between two nodes')
    parser.add_argument('--length', type=int, default=50, help='Length of rank walker')
    args = parser.parse_args()  

    return args


