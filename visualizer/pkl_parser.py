import os
import pickle
import networkx as nx
from visualizer import HTMLTemplateProcessor, validate_graph_data
from visualizer import extract_graph_data  # your existing extractor

def load_graph_from_pkl(pkl_file: str) -> nx.Graph:
    """Load a NetworkX graph from a pickle file."""
    with open(pkl_file, "rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(graph)}")
    return graph

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

def visualize_folder(folder_path: str = "graphs/queries", template_path: str = "new_template.html", output_dir: str = "./plots"):
    """Visualize all pickle graphs in a folder."""
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pkl"):
            pkl_file = os.path.join(folder_path, file)
            visualize_pkl_graph(pkl_file, template_path, output_dir)

if __name__ == "__main__":
    visualize_folder()
