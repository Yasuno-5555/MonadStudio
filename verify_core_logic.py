import monad_core
import sys

def test():
    print("Testing monad_core...")
    print(f"Module file: {monad_core.__file__}")
    engine = monad_core.MonadEngine(50, 50, 7)
    
    # Test 1: Low beta
    res1 = engine.solve_steady_state(0.90, 2.0, 0.0, 5.0, 0.0)
    print(f"Beta=0.90 -> r={res1.r:.6f}, Y={res1.Y:.6f}, w={res1.w:.6f}")

    # Test 2: High beta
    res2 = engine.solve_steady_state(0.99, 2.0, 0.0, 5.0, 0.0)
    print(f"Beta=0.99 -> r={res2.r:.6f}, Y={res2.Y:.6f}, w={res2.w:.6f}")

    if abs(res1.r - res2.r) > 1e-6:
        print("SUCCESS: Result changed with parameter!")
        print(f"  Delta r = {res1.r - res2.r:.6f}")
    else:
        print("FAILURE: Result is CONSTANT!")

if __name__ == "__main__":
    test()
