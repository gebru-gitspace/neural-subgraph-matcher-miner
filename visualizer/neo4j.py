import os
import pickle
import networkx as nx
from neo4j import GraphDatabase

# Load credentials from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def load_graph_from_pkl(pkl_file: str) -> nx.Graph:
    """Load a NetworkX graph from a pickle file."""
    with open(pkl_file, "rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(graph)}")
    return graph

def push_graph_to_neo4j(graph: nx.Graph, graph_name: str):
    """Push a NetworkX graph to Neo4j."""
    with driver.session() as session:
        # Create nodes
        for node_id, attrs in graph.nodes(data=True):
            session.run(
                "MERGE (n:Node {id: $id, graph: $graph}) SET n += $attrs",
                id=node_id, graph=graph_name, attrs=attrs
            )
        # Create edges
        for src, tgt, attrs in graph.edges(data=True):
            session.run(
                "MATCH (a:Node {id: $src, graph: $graph}), "
                "(b:Node {id: $tgt, graph: $graph}) "
                "MERGE (a)-[r:CONNECTED]->(b) "
                "SET r += $attrs",
                src=src, tgt=tgt, graph=graph_name, attrs=attrs
            )
    print(f"Pushed graph '{graph_name}' to Neo4j with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

def load_folder_to_neo4j(folder_path: str = "graphs/queries"):
    """Load all .pkl graphs in a folder to Neo4j."""
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pkl"):
            graph_file = os.path.join(folder_path, file)
            graph = load_graph_from_pkl(graph_file)
            graph_name = os.path.splitext(file)[0]
            push_graph_to_neo4j(graph, graph_name)

if __name__ == "__main__":
    load_folder_to_neo4j()
    driver.close()
