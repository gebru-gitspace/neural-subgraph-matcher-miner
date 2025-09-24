import os
import pickle
import networkx as nx
# from html_processor import HTMLTemplateProcessor, validate_graph_data
from visualizer import extract_graph_data, HTMLTemplateProcessor, validate_graph_data  # your existing extractor

def load_graph_from_pkl(pkl_file: str) -> nx.Graph:
    """Load a NetworkX graph from a pickle file."""
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, nx.Graph):
        return data

    # Convert list of edges/nodes into a graph
    if isinstance(data, list):
        G = nx.Graph()
        try:
            # Try edges first
            G.add_edges_from(data)
        except Exception:
            # If fails, assume nodes only
            G.add_nodes_from(data)
        return G

    raise TypeError(f"Cannot convert object of type {type(data)} to NetworkX graph")


def visualize_pkl_graph(pkl_file: str, template_path: str = "template.html", output_dir: str = "./plots") -> str:
    """Visualize a pickle graph using your HTML template."""
    graph = load_graph_from_pkl(pkl_file)
    graph_data = extract_graph_data(graph)

    if not validate_graph_data(graph_data):
        raise ValueError(f"Graph data extracted from {pkl_file} is invalid")

    processor = HTMLTemplateProcessor(template_path)
    filename = processor.generate_filename(graph_data, base_name=os.path.basename(pkl_file).split(".")[0])
    output_path = processor.process_template(graph_data, output_filename=filename, output_dir=output_dir)

    print(f"Graph visualization saved to: {output_path}")
    return output_path

def visualize_folder(folder_path, template_path: str = "template.html", output_dir: str = "./plots"):
    """Visualize all pickle graphs in a folder."""
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pkl"):
            pkl_file = os.path.join(folder_path, file)
            visualize_pkl_graph(pkl_file, template_path, output_dir)

if __name__ == "__main__":
    path = "../graphs/queries"
    visualize_folder(path)
