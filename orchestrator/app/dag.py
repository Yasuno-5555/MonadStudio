"""DAG analysis and validation."""
import networkx as nx
from typing import List, Tuple
from .models import DAG

def build_dag(dag_def: DAG) -> Tuple[List[str], nx.DiGraph, List[str]]:
    """
    Build and validate DAG from scenario definition.
    Returns (execution_order, graph, orphan_node_ids).
    
    Orphan nodes (not connected by any edge) are excluded from execution.
    """
    G = nx.DiGraph()

    # Add nodes
    for node in dag_def.nodes:
        G.add_node(node.id, type=node.type, params=node.params)

    # Add edges
    for e in dag_def.edges:
        G.add_edge(e.from_, e.to, port_out=e.port_out, port_in=e.port_in)

    # Validate: must be acyclic
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("DAG contains cycles - invalid scenario")

    # Identify connected nodes (have at least one edge)
    connected_nodes = set()
    for e in dag_def.edges:
        connected_nodes.add(e.from_)
        connected_nodes.add(e.to)
    
    # Orphan nodes = all nodes - connected nodes
    all_node_ids = {node.id for node in dag_def.nodes}
    orphan_nodes = list(all_node_ids - connected_nodes)
    
    # Topological sort only connected nodes
    if connected_nodes:
        # Create subgraph of only connected nodes
        connected_subgraph = G.subgraph(connected_nodes).copy()
        order = list(nx.topological_sort(connected_subgraph))
    else:
        order = []

    return order, G, orphan_nodes

def validate_connections(G: nx.DiGraph) -> List[str]:
    """
    Validate economic connections.
    Returns list of warnings/errors.
    """
    warnings = []
    
    for node_id in G.nodes:
        node_type = G.nodes[node_id].get("type")
        
        # Household cannot connect directly to Household
        for successor in G.successors(node_id):
            succ_type = G.nodes[successor].get("type")
            if node_type == "Household" and succ_type == "Household":
                warnings.append(f"Invalid: Household {node_id} -> Household {successor}")
    
    return warnings
