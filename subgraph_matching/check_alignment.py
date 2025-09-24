"""
Subgraph alignment: robust handling for multiple graph formats
- Queries: list of nx.Graph
- Target: list of tuples (node_id, node_data_dict)
"""

import argparse
import os
import pickle
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

from common import utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.train import build_model


def gen_alignment_matrix(model, query, target, method_type="order"):
    """Generate subgraph matching alignment matrix."""
    mat = np.zeros((len(query), len(target)))

    for i, u in enumerate(query.nodes):
        for j, v in enumerate(target.nodes):
            print(f"Processing query node {u} -> target node {v}")  # debug
            batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)

            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]

            mat[i, j] = raw_pred.item()

    return mat


def make_hashable_dynamic(obj):
    """
    Recursively convert obj into a hashable type.
    Works for dict, list, tuple, int, str, float.
    """
    if isinstance(obj, dict):
        # Use 'id' if present
        if "id" in obj:
            return obj["id"]
        # Flatten dict into sorted tuple of key/value hashables
        return tuple((k, make_hashable_dynamic(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(make_hashable_dynamic(x) for x in obj)
    else:
        return obj  # already hashable


def load_graph(path, fallback_size=8, p=0.25, is_target=False):
    """
    Smart graph loader:
    - Queries: list of nx.Graph
    - Target: nx.Graph from list of tuples/dicts/complex structures
    """
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, using random fallback graph.")
        return nx.gnp_random_graph(fallback_size, p)

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Queries: list of nx.Graph
    if not is_target:
        if isinstance(data, list) and all(isinstance(g, nx.Graph) for g in data):
            return data
        else:
            # fallback: try to convert each item to nx.Graph
            graphs = []
            for item in data:
                if isinstance(item, nx.Graph):
                    graphs.append(item)
                elif isinstance(item, dict):
                    g = nx.Graph()
                    for node, nbrs in item.items():
                        node_id = make_hashable_dynamic(node)
                        g.add_node(node_id)
                        if isinstance(nbrs, dict):
                            for nbr in nbrs.keys():
                                g.add_edge(node_id, make_hashable_dynamic(nbr))
                        elif isinstance(nbrs, list):
                            for nbr in nbrs:
                                g.add_edge(node_id, make_hashable_dynamic(nbr))
                    graphs.append(g)
                elif isinstance(item, tuple) and len(item) == 2:
                    g = nx.Graph()
                    node_idx, node_data = item
                    node_id = make_hashable_dynamic(node_data.get("id", node_idx) if isinstance(node_data, dict) else node_idx)
                    g.add_node(node_id)
                    neighbors = node_data.get("neighbors", []) if isinstance(node_data, dict) else []
                    for nbr in neighbors:
                        g.add_edge(node_id, make_hashable_dynamic(nbr))
                    graphs.append(g)
                else:
                    raise ValueError(f"Cannot convert query item {item} to nx.Graph")
            return graphs

    # Target graph: build nx.Graph dynamically
    G = nx.Graph()
    if isinstance(data, nx.Graph):
        return data

    elif isinstance(data, list):
        # List of tuples or dict-like nodes
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                node_idx, node_data = item
                node_id = make_hashable_dynamic(node_data.get("id", node_idx) if isinstance(node_data, dict) else node_idx)
                G.add_node(node_id, **(node_data if isinstance(node_data, dict) else {}))
                neighbors = node_data.get("neighbors", []) if isinstance(node_data, dict) else []
                for nbr in neighbors:
                    nbr_id = make_hashable_dynamic(nbr)
                    # Only add edge if nbr_id is hashable and valid
                    if nbr_id != node_id:  # avoid self-loop accidentally
                        G.add_edge(node_id, nbr_id)
                            # G.add_edge(node_id, make_hashable_dynamic(nbr))
            else:
                node_id = make_hashable_dynamic(item)
                G.add_node(node_id)

    elif isinstance(data, dict):
        for node, nbrs in data.items():
            node_id = make_hashable_dynamic(node)
            G.add_node(node_id)
            if isinstance(nbrs, dict):
                for nbr in nbrs.keys():
                    G.add_edge(node_id, make_hashable_dynamic(nbr))
            elif isinstance(nbrs, list):
                for nbr in nbrs:
                    G.add_edge(node_id, make_hashable_dynamic(nbr))

    else:
        raise ValueError(f"Unsupported target graph format in {path}")

    return G




def main():
    parser = argparse.ArgumentParser(description="Subgraph alignment")
    utils.parse_optimizer(parser)
    parse_encoder(parser)

    parser.add_argument(
        "--query_path", type=str, default="graphs/queries/patterns.pkl",
        help="Path to query graph(s) pickle"
    )
    parser.add_argument(
        "--target_path", type=str, default="graphs/targets/graph.pkl",
        help="Path to target graph pickle"
    )

    args = parser.parse_args()

    # Load queries + target
    queries = load_graph(args.query_path, is_target=False)
    target = load_graph(args.target_path, is_target=True)

    # print("Loaded queries:", type(queries), len(queries) if isinstance(queries, list) else 1)
    # if isinstance(queries, list):
    #     print("First query nodes:", list(queries[0].nodes))
    # print("Target nodes:", list(target.nodes))


    # Build model
    # target = queries[-1]
    model = build_model(args)
    print("[INFO] Model built:")

    # Ensure output dirs
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Take only the first query safely
    if isinstance(queries, list):
        query = queries[0]
    else:
        query = queries  # single nx.Graph


    # Generate alignment matrix
    mat = gen_alignment_matrix(model, query, target)
    print("Alignment matrix shape:", mat.shape)
    print("Alignment matrix sample:", mat[:min(5, mat.shape[0]), :min(5, mat.shape[1])])

    np.save("results/alignment_0.npy", mat)
    print("[INFO] Saved alignment matrix to results/alignment_0.npy")

    plt.imshow(mat, interpolation="nearest", cmap="viridis")
    plt.colorbar()
    plt.title("Alignment Matrix (Query 0)")
    plt.savefig("plots/alignment_0.png")
    plt.close()
    print("[INFO] Saved alignment matrix plot to plots/alignment_0.png")

    # Simple existence test
    exists = all(mat[i].max() > 0.5 for i in range(mat.shape[0]))
    print(f"[RESULT] Query 0 exists in target? {exists}")


    # # Run alignment for each query
    # for idx, query in enumerate(queries):
    #     mat = gen_alignment_matrix(model, query, target)
    #     np.save(f"results/alignment_{idx}.npy", mat)
    #     print(f"[INFO] Saved alignment matrix to results/alignment_{idx}.npy")

    #     plt.imshow(mat, interpolation="nearest", cmap="viridis")
    #     plt.colorbar()
    #     plt.title(f"Alignment Matrix (Query {idx})")
    #     plt.savefig(f"plots/alignment_{idx}.png")
    #     plt.close()
    #     print(f"[INFO] Saved alignment matrix plot to plots/alignment_{idx}.png")

    #     exists = all(mat[i].max() > 0.5 for i in range(mat.shape[0]))
    #     print(f"[RESULT] Query {idx} exists in target? {exists}")


if __name__ == "__main__":
    main()
