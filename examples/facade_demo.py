"""
Facade API Demo
Verifies the "Thinking Library" functionality.
"""
from monad import Monad, Study
import numpy as np
import matplotlib.pyplot as plt

def demo_single_thought():
    print("\n--- Demo 1: Single Thought (Monad) ---")
    
    # 1. Initialize (The Agent)
    # Using a dict config for demo speed/independence
    m = Monad({
        'name': 'Demo US Model',
        'T': 40,
        'kappa': 0.1,
        'phi_pi': 1.5,
        'open_economy': False
    })

    # 2. Setup & Shock & Solve (The Stream of Thought)
    print("Agent is thinking...")
    res = (
        m.setup(phi_pi=2.0)            # Adjust policy
         .shock("monetary", size=-0.01) # Contractionary shock
         .solve(nonlinear=True)        # Deep thought (nonlinear)
    )

    # 3. Output (The Insight)
    print("Insight generated.")
    # res.plot(title="Monetary Shock (Nonlinear)") 
    # Commented out plot to run headless, but API exists.
    
    # Check data integrity
    print("Variables:", res.data.keys())
    print("Y[0]:", res.data['Y'][0])

def demo_study_comparison():
    print("\n--- Demo 2: Comparative Study ---")
    
    base_config = {'T': 40, 'kappa': 0.1, 'open_economy': False}
    
    # Create Study
    study = Study("Policy Robustness")
    
    # Add cases
    study.add("Hawk (phi=2.5)", 
              Monad(base_config).setup(phi_pi=2.5).shock("monetary", -0.01))
              
    study.add("Dove (phi=1.1)", 
              Monad(base_config).setup(phi_pi=1.1).shock("monetary", -0.01))
    
    # Run batch
    study.run(nonlinear=False) # Quick linear run
    
    # Check results
    hawk_y = study.results["Hawk (phi=2.5)"].data['Y'][0]
    dove_y = study.results["Dove (phi=1.1)"].data['Y'][0]
    
    print(f"Hawk Impact: {hawk_y:.4f}")
    print(f"Dove Impact: {dove_y:.4f}")
    
    # Ideally Dove has larger fluctuations (less stable) -> larger Y impact?
    # Actually higher phi_pi stabilizes inflation better.
    
if __name__ == "__main__":
    try:
        demo_single_thought()
        demo_study_comparison()
        print("\n[SUCCESS] Facade API Verified.")
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        import traceback
        traceback.print_exc()
