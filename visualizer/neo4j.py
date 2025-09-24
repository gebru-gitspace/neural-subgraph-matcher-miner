from neo4j import GraphDatabase
import pickle
import networkx as nx

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def upload_graph(graph: nx.Graph, name: str):
    with driver.session() as session:
        session.run(f"MATCH (n) DETACH DELETE n")  # optional: clear old graph
        for node in graph.nodes(data=True):
            session.run("CREATE (n:Node {id: $id, label: $label})", id=node[0], label=node[1].get("label", str(node[0])))
        for u, v, d in graph.edges(data=True):
            session.run("""
            MATCH (a:Node {id:$u}), (b:Node {id:$v})
            CREATE (a)-[:CONNECTS]->(b)
            """, u=u, v=v)
