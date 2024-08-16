import json
import os.path as osp
import time
from math import fabs
from random import sample
import numpy as np
import pandas as pd
from torch_geometric.nn import SignedGCN
import torch
from torch import Tensor
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# def load_data(file_path):
#     if file_path.endswith('.txt'):
#         data = pd.read_csv(file_path, sep='\t', header=None)
#     elif file_path.endswith('.csv'):
#         data = pd.read_csv(file_path)
#     else:
#         raise ValueError("Unsupported file format")
#     return data
#
# def process_edges(data):
#     edge_index = torch.tensor(data.iloc[:, :2].values.T, dtype=torch.long)
#     edge_attr = torch.tensor(data.iloc[:, 2].values, dtype=torch.float)
#     return edge_index, edge_attr

def load_data(file_path):
    if file_path.endswith('.txt'):
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data
def process_edges(data):
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    edge_index = torch.tensor(data.iloc[:, :2].values.astype(np.int64).T, dtype=torch.long)
    edge_attr = torch.tensor(data.iloc[:, 2].values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr


def main(name: str, offset: float, i: int):
    train_data_path = f'./train/{name}_{i}-train_edges.csv'
    test_data_path = f'./test/{name}_{i}-test_edges.csv'
    embedding_save_path = f'./output/{name}_{i}-embedding.pt'

    # Load and process training and test data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    train_edge_index, train_edge_attr = process_edges(train_data)
    test_edge_index, test_edge_attr = process_edges(test_data)

    pos_edge_indices, neg_edge_indices = [], []
    pos_edge_indices.append(train_edge_index[:, train_edge_attr >= offset])
    neg_edge_indices.append(train_edge_index[:, train_edge_attr < offset])

    train_pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
    train_neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

    pos_edge_indices = []
    neg_edge_indices = []
    pos_edge_indices.append(test_edge_index[:, test_edge_attr >= offset])
    neg_edge_indices.append(test_edge_index[:, test_edge_attr < offset])

    test_pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
    test_neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

    def discriminate(z: Tensor, pos_neg_edge_index: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        k = 40
        m = 200
        device = z.device
        data_dir = f"Dist/{name}/"
        os.makedirs(data_dir, exist_ok=True)

        truth_k_pos_neighbors_path = os.path.join(data_dir, "truth_k_pos_neighbors_tensor.pt")
        truth_k_neg_neighbors_path = os.path.join(data_dir, "truth_k_neg_neighbors_tensor.pt")
        k_avg_pos_distance_path = os.path.join(data_dir, "k_avg_pos_distance.pt")
        k_avg_neg_distance_path = os.path.join(data_dir, "k_avg_neg_distance.pt")
        k_pos_neighbors_path = os.path.join(data_dir, "k_pos_neighbors_tensor.pt")
        k_neg_neighbors_path = os.path.join(data_dir, "k_neg_neighbors_tensor.pt")

        z = z.to(device)
        print("num_nodes", z.size(0))
        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)

        pos_neighbors = [[] for _ in range(z.size(0))]
        neg_neighbors = [[] for _ in range(z.size(0))]

        # Find neighbors from pos_edge_index and neg_edge_index
        for edge in pos_edge_index.t():
            pos_neighbors[edge[1]].append(edge[0].item())

        for edge in neg_edge_index.t():
            neg_neighbors[edge[1]].append(edge[0].item())

        pos_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in
                                pos_neighbors]
        neg_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in
                                neg_neighbors]

        truth_k_pos_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]
        truth_k_neg_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]

        for i in range(z.size(0)):
            if len(pos_neighbors_tensor[i]) > k:
                pos_distances = torch.norm(z[i] - z[pos_neighbors_tensor[i]], dim=1)
                truth_k_pos_neighbors_tensor[i] = pos_neighbors_tensor[i][
                    pos_distances.topk(k, largest=False).indices]
            else:
                truth_k_pos_neighbors_tensor[i] = pos_neighbors_tensor[i]

            if len(neg_neighbors_tensor[i]) > k:
                neg_distances = torch.norm(z[i] - z[neg_neighbors_tensor[i]], dim=1)
                truth_k_neg_neighbors_tensor[i] = neg_neighbors_tensor[i][
                    neg_distances.topk(k, largest=True).indices]
            else:
                truth_k_neg_neighbors_tensor[i] = neg_neighbors_tensor[i]

        torch.save(truth_k_pos_neighbors_tensor, truth_k_pos_neighbors_path)
        torch.save(truth_k_neg_neighbors_tensor, truth_k_neg_neighbors_path)

        pos_neighbors = [[] for _ in range(z.size(0))]
        neg_neighbors = [[] for _ in range(z.size(0))]

        for edge in pos_edge_index.t():
            pos_neighbors[edge[1]].append(edge[0].item())

        for edge in neg_edge_index.t():
            neg_neighbors[edge[1]].append(edge[0].item())

        pos_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in pos_neighbors]
        neg_neighbors_tensor = [torch.tensor(neighbors, dtype=torch.long, device=device) for neighbors in neg_neighbors]

        k_pos_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]
        k_neg_neighbors_tensor = [torch.tensor([], dtype=torch.long, device=device) for _ in range(z.size(0))]
        k_avg_pos_distance = torch.zeros(z.size(0), device=device)
        k_avg_neg_distance = torch.zeros(z.size(0), device=device)

        for i in range(z.size(0)):
            if len(pos_neighbors_tensor[i]) > 0:
                random_sample_pos_nei = pos_neighbors_tensor[i][torch.randperm(len(pos_neighbors_tensor[i]))[:m]]
                pos_distances = torch.norm(z[i] - z[random_sample_pos_nei], dim=1)
                if len(pos_distances) > k:
                    k_pos_neighbors_tensor[i] = random_sample_pos_nei[pos_distances.topk(k, largest=False).indices]
                    k_avg_pos_distance[i] = pos_distances.topk(k, largest=False).values.mean().item()
                else:
                    k_pos_neighbors_tensor[i] = random_sample_pos_nei
                    k_avg_pos_distance[i] = pos_distances.mean().item()
            else:
                k_avg_pos_distance[i] = float('0')

            if len(neg_neighbors_tensor[i]) > 0:
                random_sample_neg_nei = neg_neighbors_tensor[i][torch.randperm(len(neg_neighbors_tensor[i]))[:m]]
                neg_distances = torch.norm(z[i] - z[random_sample_neg_nei], dim=1)
                if len(neg_distances) > k:
                    k_neg_neighbors_tensor[i] = random_sample_neg_nei[neg_distances.topk(k, largest=True).indices]
                    k_avg_neg_distance[i] = neg_distances.topk(k, largest=True).values.mean().item()
                else:
                    k_neg_neighbors_tensor[i] = random_sample_neg_nei
                    k_avg_neg_distance[i] = neg_distances.mean().item()
            else:
                k_avg_neg_distance[i] = float('inf')

        torch.save(k_avg_pos_distance, k_avg_pos_distance_path)
        torch.save(k_avg_neg_distance, k_avg_neg_distance_path)
        torch.save(k_pos_neighbors_tensor, k_pos_neighbors_path)
        torch.save(k_neg_neighbors_tensor, k_neg_neighbors_path)

        Dist_ij = torch.norm(z[pos_neg_edge_index[0]] - z[pos_neg_edge_index[1]], dim=1)

        logits = torch.zeros((Dist_ij.size(0), 3), device=device)
        a = 0.05
        b = 0
        for i in range(Dist_ij.size(0)):
            node = pos_neg_edge_index[1, i].item()
            dist = Dist_ij[i]
            dist_1 = abs(dist - k_avg_pos_distance[node]) - a
            dist_2 = abs(dist - k_avg_neg_distance[node]) - b

            if dist_1 < dist_2:
                logits[i] = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
            else:
                logits[i] = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

        return torch.log_softmax(logits, dim=1)

    def explainer(pos_neg_edge_index: Tensor, is_positive: bool) -> float:
        """
        Explains the predictions by comparing the expected neighbors with the ground truth neighbors.

        Args:
            pos_neg_edge_index (Tensor): The edge indices to explain.
            is_positive (bool): Flag indicating whether the edges are positive or negative.

        Returns:
            float: The explanation accuracy.
        """
        z = torch.load(embedding_save_path)
        device = pos_neg_edge_index.device
        num_nodes = pos_neg_edge_index.max().item() + 1

        # Initialize ground truth and expected neighbors for each node
        ground_truth = [set() for _ in range(num_nodes)]
        exp_nei = [set() for _ in range(num_nodes)]

        # Load precomputed neighbor tensors
        truth_k_pos_neighbors_tensor = torch.load(f"Dist/{name}/truth_k_pos_neighbors_tensor.pt")
        truth_k_neg_neighbors_tensor = torch.load(f"Dist/{name}/truth_k_neg_neighbors_tensor.pt")
        k_pos_neighbors_tensor = torch.load(f"Dist/{name}/k_pos_neighbors_tensor.pt")
        k_neg_neighbors_tensor = torch.load(f"Dist/{name}/k_neg_neighbors_tensor.pt")

        # Predict the edge types
        predictions = discriminate(z, pos_neg_edge_index, train_pos_edge_index, train_neg_edge_index)[:, :2].max(dim=1)[
            1]

        for i, edge in enumerate(pos_neg_edge_index.t()):
            node = edge[1].item()
            if is_positive:
                ground_truth[node].update(truth_k_pos_neighbors_tensor[node].tolist())
                if predictions[i] == 0:  # Predicted as positive
                    exp_nei[node].update(k_pos_neighbors_tensor[node].tolist())
                else:  # Predicted as negative
                    exp_nei[node].update(k_neg_neighbors_tensor[node].tolist())
            else:
                ground_truth[node].update(truth_k_neg_neighbors_tensor[node].tolist())
                if predictions[i] == 1:  # Predicted as negative
                    exp_nei[node].update(k_neg_neighbors_tensor[node].tolist())
                else:  # Predicted as positive
                    exp_nei[node].update(k_pos_neighbors_tensor[node].tolist())

        # Calculate explanation accuracy for each node and take the mean
        accuracies = []
        for i in range(num_nodes):
            if len(ground_truth[i]) > 0:
                correct_predictions = ground_truth[i].intersection(exp_nei[i])
                accuracy = len(correct_predictions) / len(ground_truth[i])
                accuracies.append(accuracy)

        # If there are no valid ground truth sets, return 0.0
        if len(accuracies) == 0:
            return 0.0
        else:
            explanation_accuracy = sum(accuracies) / len(accuracies)

        return explanation_accuracy

    from typing import Tuple
    import torch
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    def test() -> Tuple[float, float, float, float]:
        """Evaluates node embeddings :obj:z on positive and negative test
        edges by computing AUC, F1 scores, and Accuracy.

        Args:
            z (torch.Tensor): The node embeddings.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        with torch.no_grad():
            z = torch.load(embedding_save_path)
            pos_p = discriminate(z, test_pos_edge_index, train_pos_edge_index, train_neg_edge_index)[:, :2].max(dim=1)[1]
            neg_p = discriminate(z, test_neg_edge_index, train_pos_edge_index, train_neg_edge_index)[:, :2].max(dim=1)[1]
            exp_pos_p = explainer(test_pos_edge_index, is_positive=True)
            exp_neg_p = explainer(test_neg_edge_index, is_positive=False)
            exp_total = (1.5 * exp_pos_p + 0.5 * exp_neg_p) / 2

        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat([pred.new_ones((pos_p.size(0))), pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)

        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0
        acc = accuracy_score(y, pred.round())
        return auc, f1, acc, exp_total

    auc, f1, acc, exp_total = test()
    return auc, f1, acc, exp_total


seed_list = [1145, 14, 191, 9810, 721]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the discriminator.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--offset', type=float, default=4.0, help="Offset value for edge classification")
    args = parser.parse_args()

    for i in range(5):
        print(f"Running iteration {i}")
        main(name=args.dataset, offset=args.offset, i=i)

