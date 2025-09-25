# visualizer_show_all_auto_labels.py
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def load_pkl(path: str):
    """Load a pickle file containing a NetworkX graph or a list of graphs."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"✅ Loaded {path}, type: {type(obj)}")
    return obj


def format_attributes(attrs: dict):
    """Format node or edge attributes as multi-line string for labels."""
    if not attrs:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in attrs.items())


def visualize_graphs_auto_labels(graphs):
    """Visualize single or list of NetworkX graphs with all attributes as labels."""

    # Ensure graphs is a list
    if isinstance(graphs, (nx.Graph, nx.DiGraph)):
        graphs = [graphs]

    num_graphs = len(graphs)

    # --- ✅ Create subplots (1 row, N columns) ---
    fig, axes = plt.subplots(1, num_graphs, figsize=(7 * num_graphs, 7))

    # If only 1 graph, axes is not a list
    if num_graphs == 1:
        axes = [axes]

    for i, (g, ax) in enumerate(zip(graphs, axes)):
        if not isinstance(g, (nx.Graph, nx.DiGraph)):
            print(f"⚠️ Skipping item {i}, not a graph: {type(g)}")
            continue

        print(f"🔍 Graph {i}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

        pos = nx.spring_layout(g, seed=42)

        # Node labels
        node_labels = {n: format_attributes(data) for n, data in g.nodes(data=True)}

        # Edge labels (show only "type")
        edge_labels = {(u, v): data.get("type", "") for u, v, data in g.edges(data=True)}

        # --- Draw graph in subplot ---
        nx.draw(
            g,
            pos,
            with_labels=False,
            node_color="skyblue",
            node_size=600,
            edge_color="gray",
            arrows=isinstance(g, nx.DiGraph),
            ax=ax,   # ✅ draw in specific subplot
        )
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=7, ax=ax)
        if edge_labels:
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        ax.set_title(f"Graph {i}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Path to your pkl file
    path = Path("graphs/queries/patterns.pkl")

    if not path.exists():
        print(f"❌ File not found: {path}")
    else:
        obj = load_pkl(path)
        visualize_graphs_auto_labels(obj)
