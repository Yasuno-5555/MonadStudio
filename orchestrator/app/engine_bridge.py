"""Bridge to monad_core C++ engine."""
import sys
import os

# Add parent directory to path for monad_core import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import monad_core
from typing import Tuple, Dict, Any

class EngineBridge:
    """Wrapper around monad_core.MonadEngine."""
    
    def __init__(self, grid: Tuple[int, int, int] = (50, 50, 7)):
        """Initialize engine with grid dimensions (Nm, Na, Nz)."""
        self.grid = grid
        self.engine = monad_core.MonadEngine(*grid)
    
    def solve_ss(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve steady state with given parameters.
        
        Expected params:
            beta: discount factor
            sigma: risk aversion
            chi0, chi1, chi2: adjustment cost parameters (unused in One-Asset)
        """
        # Extract params with defaults
        beta = params.get("beta", 0.986)
        sigma = params.get("sigma", 2.0)
        chi0 = params.get("chi0", 0.0)
        chi1 = params.get("chi1", 5.0)
        chi2 = params.get("chi2", 0.0)
        
        # Call C++ engine
        result = self.engine.solve_steady_state(beta, sigma, chi0, chi1, chi2)
        
        # Extract distribution data (flattened, for JSON serialization)
        # Distribution is 3D (Nm x Na x Nz), but for One-Asset we only use (Na x Nz)
        # Mass is normalized to sum to 1.0 (probability distribution)
        dist_shape = result.distribution.shape  # (Nm, Na, Nz)
        Nm, Na, Nz = dist_shape
        
        # Flatten distribution for JSON (row-major: z changes slowest)
        # Extract slice at m=0 (where all mass is for One-Asset model)
        dist_2d = []
        for z_idx in range(Nz):
            for a_idx in range(Na):
                dist_2d.append(float(result.distribution[0, a_idx, z_idx]))
        
        # Generate grid_a (asset grid: 0 to 50 with power spacing, curvature=2)
        # This matches make_grid(Na, 0.0, 50.0, 2.0) in engine.cpp
        grid_a = []
        for i in range(Na):
            t = i / (Na - 1) if Na > 1 else 0.0
            grid_a.append(0.0 + (50.0 - 0.0) * (t ** 2.0))
        
        # grid_z (income states: hardcoded in make_income())
        grid_z = [0.8, 1.2]  # Must match engine.cpp make_income()
        
        return {
            "r": result.r,
            "w": result.w,
            "Y": result.Y,
            "C": result.C,
            "distribution_shape": list(dist_shape),
            # Heatmap data (mass is normalized, sums to 1.0)
            "distribution_data": dist_2d,  # Flattened (Nz * Na), z-major
            "grid_a": grid_a,              # Asset grid values
            "grid_z": grid_z,              # Income states
            "Na": Na,
            "Nz": Nz
        }

