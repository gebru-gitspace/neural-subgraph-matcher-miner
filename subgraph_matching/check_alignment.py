import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from subgraph_matching.alignment import gen_alignment_matrix, build_model
from subgraph_matching.config import parse_encoder
from common import utils
import argparse

def visualize_graph(graph, title="", path=None):
    plt.figure(figsize=(5,5))
    nx.draw(graph, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.title(title)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parse_encoder(parser)
    utils.parse_optimizer(parser)
    parser.add_argument('--query_dir', type=str, default='graphs/queries')
    parser.add_argument('--target_dir', type=str, default='graphs/targets')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    model = build_model(args)

    query_files = sorted(os.listdir(args.query_dir))
    target_files = sorted(os.listdir(args.target_dir))

    for qf in query_files:
        with open(os.path.join(args.query_dir, qf), 'rb') as f:
            query = pickle.load(f)
        visualize_graph(query, title=f"Query: {qf}", path=f"plots/query_{qf}.png")

        for tf in target_files:
            with open(os.path.join(args.target_dir, tf), 'rb') as f:
                target = pickle.load(f)
            visualize_graph(target, title=f"Target: {tf}", path=f"plots/target_{tf}.png")

            # Alignment matrix
            mat = gen_alignment_matrix(model, query, target, method_type=args.method_type)
            exists = all(mat[i].max() > args.threshold for i in range(mat.shape[0]))
            print(f"Query {qf} exists in Target {tf}? {exists}")

            # Save alignment plot
            plt.figure(figsize=(6,6))
            plt.imshow(mat, interpolation='nearest')
            plt.colorbar()
            plt.title(f"Alignment: {qf} -> {tf}")
            plt.savefig(f"plots/alignment_{qf}_{tf}.png")
            plt.close()

if __name__ == "__main__":
    main()
