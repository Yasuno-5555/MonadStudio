"""Execution pipeline for scenarios."""
from typing import Dict, Any
from .dag import build_dag, validate_connections
from .engine_bridge import EngineBridge
from .models import Scenario

def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    """
    Execute a scenario.
    
    1. Parse and validate DAG
    2. Execute nodes in topological order
    3. Return aggregated results
    """
    # Build DAG
    order, G = build_dag(scenario.dag)
    
    # Validate connections
    warnings = validate_connections(G)
    if warnings:
        # For now, just log warnings; don't fail
        print(f"Warnings: {warnings}")
    
    # Initialize engine
    engine = EngineBridge(grid=(50, 50, 7))
    
    # State storage
    state: Dict[str, Any] = {}
    
    # Execute in order
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
            # Phase 1: stub
            state[node_id] = {
                "type": "CentralBank",
                "params": node_params
            }
        elif node_type == "Firm":
            # Phase 1: stub
            state[node_id] = {
                "type": "Firm",
                "params": node_params
            }
        elif node_type == "MarketClearing":
            # Phase 1: stub - would aggregate all
            state[node_id] = {
                "type": "MarketClearing",
                "status": "pending"
            }
        else:
            state[node_id] = {
                "type": node_type,
                "status": "unknown"
            }
    
    return {
        "execution_order": order,
        "nodes": state,
        "simulation": {
            "horizon": scenario.simulation.horizon,
            "shock": scenario.simulation.shock.model_dump()
        }
    }
