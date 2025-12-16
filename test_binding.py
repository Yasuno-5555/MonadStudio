import sys
import os
import time
import numpy as np

# Ensure we can import the .so/.pyd file generated in the current directory
sys.path.append(os.getcwd())

try:
    import monad_core
    print("✅ Successfully imported monad_core!")
except ImportError as e:
    print(f"❌ Failed to import monad_core: {e}")
    sys.exit(1)

def test_run():
    print("-" * 40)
    print("🚀 Starting Monad Engine Test (Phase 0: Refactored)")
    
    # 1. Initialize Engine
    Nm, Na, Nz = 50, 50, 7
    print(f"🔹 Initializing MonadEngine({Nm}, {Na}, {Nz})...")
    engine = monad_core.MonadEngine(Nm, Na, Nz)
    
    # 2. Call solve_steady_state
    # Params: beta, sigma, chi0, chi1, chi2
    print("🔹 Calling solve_steady_state()...")
    start_time = time.time()
    ss = engine.solve_steady_state(0.986, 2.0, 0.0, 5.0, 0.0)
    end_time = time.time()

    # 3. Verify Output
    print(f"⏱️  Execution Time: {(end_time - start_time)*1000:.4f} ms")
    print(f"🔹 r: {ss.r}")
    print(f"🔹 w: {ss.w}")
    print(f"🔹 Y: {ss.Y}")
    print(f"🔹 C: {ss.C}")

    # 4. Check Distribution
    dist = ss.distribution
    print(f"🔹 Distribution Shape: {dist.shape}")
    print(f"🔹 Distribution Sample (center): {dist[25, 25, 3]:.6f}")

    if dist.shape == (50, 50, 7):
        print("✅ Distribution Shape OK")
    else:
        print("❌ Distribution Shape Mismatch")

    # Simple verify if data is accessible
    if dist.sum() > 0:
        print("✅ Distribution data contains non-zero values")
    else:
        print("❌ Distribution data seems empty or zero")

    print("-" * 40)

if __name__ == "__main__":
    test_run()
