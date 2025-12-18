"""
Verify Phase 2 Features
- Reproducibility (Metadata export)
- Determinacy API
- Immutability
"""
from monad import Monad
import os
import json

def verify_reproducibility():
    print("\n--- Testing Reproducibility Layer ---")
    m = Monad("us_normal")
    m.shock("monetary", -0.01)
    res = m.solve(nonlinear=False)
    
    # Check Metadata existence in object
    assert 'timestamp' in res.meta
    assert 'shocks' in res.meta
    
    # Check Export Sidecar
    export_path = "test_export.csv"
    res.export(export_path)
    
    meta_path = "test_export.meta.json"
    if os.path.exists(meta_path):
        print(f"[OK] Sidecar found: {meta_path}")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            # Verify contents
            print("Sidecar Determinacy:", meta['determinacy']['status'])
            print("Sidecar Shocks:", meta['shock_definitions'].keys())
    else:
        print("[FAIL] Sidecar NOT found.")

def verify_determinacy():
    print("\n--- Testing Determinacy API ---")
    m = Monad("us_normal")
    res = m.shock("monetary", -0.01).solve()
    
    det = res.determinacy()
    print("Determinacy Dict:", det)
    assert 'status' in det
    assert 'notes' in det
    # With dummy backend, status might be 'unique' (default fallback)

def verify_immutability():
    print("\n--- Testing Immutability ---")
    params = {'phi_pi': 1.5}
    m = Monad(params)
    m.shock("monetary", -0.01)
    res = m.solve()
    
    # Change original dict; result should not change
    params['phi_pi'] = 999.0
    
    print(f"Original Param: {params['phi_pi']}")
    print(f"Result Param:   {res.params['phi_pi']}")
    
    if res.params['phi_pi'] == 1.5:
        print("[OK] Result parameters are immutable.")
    else:
        print("[FAIL] Result parameters changed with source!")

if __name__ == "__main__":
    verify_determinacy()
    verify_reproducibility()
    verify_immutability()
