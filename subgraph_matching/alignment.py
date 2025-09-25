"""Build an alignment matrix for matching a query subgraph in a target graph.
Subgraph matching model needs to have been trained with the node-anchored option
(default)."""

import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

from common import data
from common import models
from common import utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.test import validation
from subgraph_matching.train import build_model
from pkl_visualizer import *

def gen_alignment_matrix(model, query, target, method_type="order"):
    """Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: the subgraph matching model. Must have been trained with
            node anchored setting (--node_anchored, default)
        query: the query graph (networkx Graph)
        target: the target graph (networkx Graph)
        method_type: the method used for the model.
            "order" for order embedding or "mlp" for MLP model
    """

    mat = np.zeros((len(query), len(target)))
    for i, u in enumerate(query.nodes):
        for j, v in enumerate(target.nodes):
            batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)
            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]
            mat[i][j] = raw_pred.item()
    return mat

def main():
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    parser = argparse.ArgumentParser(description='Alignment arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    args = parser.parse_args()
    args.test = True
    if args.query_path:
        with open(args.query_path, "rb") as f:
            query = pickle.load(f)[0]
    else:
        query = nx.gnp_random_graph(5, 0.5)
    if args.target_path:
        with open(args.target_path, "rb") as f:
            target = pickle.load(f)


            # If your target is a dict of edges or adjacency:
            if isinstance(target, dict):
                target = nx.from_dict_of_dicts(target)  # or from_dict_of_lists depending on structure

    else:
        target = nx.gnp_random_graph(10, 0.5)

    model = build_model(args)
    mat = gen_alignment_matrix(model, query, target,
        method_type=args.method_type)

    np.save("results/alignmentq.npy", mat)
    print("Saved alignment matrix in results/alignmentq.npy")

    plt.imshow(mat, interpolation="nearest")
    plt.savefig("plots/alignmentq.png")
    print("Saved alignment matrix plot in plots/alignmenqt.png")

    ## Simple existence test, whether each query is subgraph of target
    threshold = 0.5
    score_avg = mat.max(axis=1).mean()
    exists_avg = score_avg > threshold
    print(f"[Option B] Average-max score: {score_avg:.3f}, Query exists in target? {exists_avg}")

    # ------------------------
    # OPTION A: Max matching (one-to-one mapping)

    binary_mat = (mat > threshold).astype(int)
    B = nx.Graph()
    for i in range(binary_mat.shape[0]):
        for j in range(binary_mat.shape[1]):
            if binary_mat[i, j]:
                B.add_edge(f"q{i}", f"t{j}")

    # Only query nodes that actually exist in B
    top_nodes = {n for n in B.nodes if n.startswith("q")}

    matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=top_nodes)

    # Boolean result
    exists_matching = all(f"q{i}" in matching for i in range(mat.shape[0]))
    print(f"[Option A] One-to-one matching exists? {exists_matching}")

    # Print mapping
    print("Query node → Target node mapping:")
    for q in range(mat.shape[0]):
        tgt = matching.get(f"q{q}")
        if tgt:
            print(f"  q{q} → {tgt}")
        else:
            print(f"  q{q} → None")

    
    # Visualize query and target graphs
    visualize_graphs_auto_labels(query)
    visualize_graphs_auto_labels(target)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()

