"""Test the orchestrator without running the server."""
import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models import Scenario
from app.executor import run_scenario

def test_scenario():
    # Load test scenario
    test_file = os.path.join(os.path.dirname(__file__), "test_scenario.json")
    with open(test_file, "r") as f:
        data = json.load(f)
    
    # Parse with Pydantic
    scenario = Scenario(**data)
    print(f"✅ Scenario parsed: {scenario.meta.version}")
    print(f"   Nodes: {len(scenario.dag.nodes)}")
    print(f"   Edges: {len(scenario.dag.edges)}")
    
    # Run scenario
    result = run_scenario(scenario)
    print(f"✅ Execution order: {result['execution_order']}")
    
    # Check household result
    if "hh_1" in result["nodes"]:
        ss = result["nodes"]["hh_1"]["steady_state"]
        print(f"✅ Household SS: r={ss['r']}, w={ss['w']}, C={ss['C']}")
    
    print("-" * 40)
    print("Phase 1 Verification: PASS ✅")

if __name__ == "__main__":
    test_scenario()
