"""DAG analysis and validation."""
import networkx as nx
from typing import List, Tuple
from .models import DAG

def build_dag(dag_def: DAG) -> Tuple[List[str], nx.DiGraph]:
    """
    Build and validate DAG from scenario definition.
    Returns execution order and the graph.
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

    # Topological sort for execution order
    order = list(nx.topological_sort(G))

    return order, G

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
