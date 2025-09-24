# visualizer_show_all_detailed.py
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def load_pkl(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"✅ Loaded {path}, type: {type(obj)}")
    return obj


def visualize_graphs_detailed(graphs, node_label_fields=None, edge_label_field="name"):
    """
    Visualize a list of NetworkX graphs with detailed labels.
    - node_label_fields: list of node attributes to show (in order)
    - edge_label_field: edge attribute to display as label
    """
    node_label_fields = node_label_fields or ["id", "label"]

    for i, g in enumerate(graphs):
        if not isinstance(g, (nx.Graph, nx.DiGraph)):
            print(f"⚠️ Skipping item {i}, not a graph: {type(g)}")
            continue

        print(f"🔍 Graph {i}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

        pos = nx.spring_layout(g, seed=42)

        # Build node labels
        node_labels = {}
        for n, data in g.nodes(data=True):
            parts = [str(data.get(f, "")) for f in node_label_fields if f in data]
            node_labels[n] = "\n".join(parts) if parts else str(n)

        # Build edge labels
        edge_labels = {}
        for u, v, data in g.edges(data=True):
            label = data.get(edge_label_field, "")
            edge_labels[(u, v)] = label

        plt.figure(figsize=(6, 6))
        nx.draw(
            g,
            pos,
            with_labels=False,
            node_color="skyblue",
            node_size=600,
            edge_color="gray",
            arrows=isinstance(g, nx.DiGraph),
        )
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8)
        if edge_labels:
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)

        plt.title(f"Graph {i}: {g.number_of_nodes()}n-{g.number_of_edges()}e")
        plt.show()


if __name__ == "__main__":
    path = "graphs/queries/patterns.pkl"
    path = Path(path)

    if not path.exists():
        print(f"❌ File not found: {path}")
    else:
        obj = load_pkl(path)
        if isinstance(obj, list) and all(isinstance(x, (nx.Graph, nx.DiGraph)) for x in obj):
            # Adjust node fields based on your YAML schema
            visualize_graphs_detailed(obj, node_label_fields=["id", "label", "gene_name", "transcript_name"])
        else:
            print("⚠️ Pickle is not a list of NetworkX graphs.")
