import os
import shutil
import sys
import time

def clean_build():
    print("Cleaning build artifacts...")
    dirs_to_clean = ["build", "monad_core.egg-info", "dist"]
    for d in dirs_to_clean:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"Removed {d}")
            except Exception as e:
                print(f"Failed to remove {d}: {e}")
        else:
            print(f"{d} not found")

def verify_source():
    print("Verifying source code...")
    with open("src/monad/engine.cpp", "r", encoding="utf-8") as f:
        content = f.read()
        if "AnalyticalSolver::solve_steady_state" in content:
            print("Source code CONFIRMED: Contains dynamic logic.")
        else:
            print("Source code ERROR: Does NOT contain dynamic logic!")
            print("First 500 chars:")
            print(content[:500])
            sys.exit(1)

if __name__ == "__main__":
    clean_build()
    verify_source()
