from neo4j import GraphDatabase
import os, pickle, networkx as nx

# load driver from environment
import os
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def load_graphs_from_pkl(pkl_file: str):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    graphs = []
    if isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        graphs.append(data)
    elif isinstance(data, list):
        for g in data:
            if isinstance(g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                graphs.append(g)
    else:
        raise TypeError(f"Loaded object is not a NetworkX graph or list: {type(data)}")
    return graphs

def push_graph_to_neo4j(graph, session):
    """Example: push nodes and edges into Neo4j."""
    for node_id, attrs in graph.nodes(data=True):
        session.run(
            "MERGE (n:Node {id:$id}) SET n += $attrs",
            id=node_id, attrs=attrs
        )
    for src, tgt, attrs in graph.edges(data=True):
        session.run(
            "MATCH (a:Node {id:$src}), (b:Node {id:$tgt}) "
            "MERGE (a)-[r:CONNECTS]->(b) SET r += $attrs",
            src=src, tgt=tgt, attrs=attrs
        )

def load_folder_to_neo4j(folder_path):
    with driver.session() as session:
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".pkl"):
                graphs = load_graphs_from_pkl(os.path.join(folder_path, file))
                for graph in graphs:
                    push_graph_to_neo4j(graph, session)

if __name__ == "__main__":
    path = "../graphs/queries"
    load_folder_to_neo4j(path)
