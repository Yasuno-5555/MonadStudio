"""Execution pipeline for scenarios."""
from typing import Dict, Any
from .dag import build_dag, validate_connections
from .engine_bridge import EngineBridge
from .models import Scenario

def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    """
    Execute a scenario.
    
    1. Parse and validate DAG
    2. Execute nodes in topological order (only connected nodes)
    3. Return aggregated results
    """
    # Build DAG
    order, G, orphan_nodes = build_dag(scenario.dag)
    
    # Validate connections
    warnings = validate_connections(G)
    if warnings:
        print(f"Warnings: {warnings}")
    
    # Report orphan nodes
    if orphan_nodes:
        print(f"Orphan nodes (not connected, skipped): {orphan_nodes}")
    
    # Initialize engine
    engine = EngineBridge(grid=(50, 50, 7))
    
    # State storage
    state: Dict[str, Any] = {}
    
    # Execute in order (only connected nodes)
    for node_id in order:
        node_data = G.nodes[node_id]
        node_type = node_data.get("type")
        node_params = node_data.get("params", {})
        
        if node_type == "Household":
            ss = engine.solve_ss(node_params)
            state[node_id] = {
                "type": "Household",
                "steady_state": ss
            }
        elif node_type == "CentralBank":
            state[node_id] = {
                "type": "CentralBank",
                "params": node_params
            }
        elif node_type == "Firm":
            state[node_id] = {
                "type": "Firm",
                "params": node_params
            }
        elif node_type == "MarketClearing":
            state[node_id] = {
                "type": "MarketClearing",
                "status": "pending"
            }
        else:
            state[node_id] = {
                "type": node_type,
                "status": "unknown"
            }
    
    # Add orphan nodes as "not executed"
    for node_id in orphan_nodes:
        node_data = G.nodes[node_id]
        state[node_id] = {
            "type": node_data.get("type"),
            "status": "orphan (not connected)"
        }
    
    return {
        "execution_order": order,
        "orphan_nodes": orphan_nodes,
        "nodes": state,
        "simulation": {
            "horizon": scenario.simulation.horizon,
            "shock": scenario.simulation.shock.model_dump()
        }
    }
