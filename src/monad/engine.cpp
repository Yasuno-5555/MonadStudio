#include "engine.h"
#include <cmath>
#include <iostream>

namespace monad {

MonadEngine::MonadEngine(int Nm, int Na, int Nz)
    : Nm_(Nm), Na_(Na), Nz_(Nz) {}

SteadyStateResult MonadEngine::solve_steady_state(
    double beta,
    double sigma,
    double chi0,
    double chi1,
    double chi2
) {
    // Phase 0: Mock Implementation
    // Eventually this will call logic from household.cpp, market.cpp etc.

    SteadyStateResult res;
    
    // Mock Scalars
    res.r = 0.02; // 2% interest rate
    res.w = 1.05; // Wage
    res.Y = 1.0;  // Output
    res.C = res.Y; // C = Y in simplest closed economy

    // Initialize Distribution
    res.distribution = Distribution3D(Nm_, Na_, Nz_);
    
    // Fill with a dummy distribution (Gaussian bump)
    double center_m = Nm_ / 2.0;
    double center_a = Na_ / 2.0;

    for(int i=0; i<Nm_; ++i) {
        for(int j=0; j<Na_; ++j) {
            for(int k=0; k<Nz_; ++k) {
                double dm = (i - center_m) / (double)Nm_;
                double da = (j - center_a) / (double)Na_;
                double val = std::exp(-10.0 * (dm*dm + da*da));
                res.distribution(i, j, k) = val;
            }
        }
    }

    return res;
}

}
