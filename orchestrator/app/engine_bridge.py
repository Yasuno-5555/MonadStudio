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
            chi0, chi1, chi2: adjustment cost parameters
        """
        # Extract params with defaults
        beta = params.get("beta", 0.986)
        sigma = params.get("sigma", 2.0)
        chi0 = params.get("chi0", 0.0)
        chi1 = params.get("chi1", 5.0)
        chi2 = params.get("chi2", 0.0)
        
        # Call C++ engine
        result = self.engine.solve_steady_state(beta, sigma, chi0, chi1, chi2)
        
        # Return as dict
        return {
            "r": result.r,
            "w": result.w,
            "Y": result.Y,
            "C": result.C,
            "distribution_shape": list(result.distribution.shape)
        }
