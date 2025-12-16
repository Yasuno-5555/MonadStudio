#pragma once
#include "steady_state.h"
#include <string>

namespace monad {

// Impulse Response Function Result
struct IRFResult {
    std::vector<double> dC;  // Consumption response
    std::vector<double> dY;  // Output response
    std::vector<double> dr;  // Interest rate response
    int T;                   // Horizon
};

class MonadEngine {
public:
    MonadEngine(int Nm, int Na, int Nz);

    SteadyStateResult solve_steady_state(
        double beta,
        double sigma,
        double chi0,
        double chi1,
        double chi2
    );
    
    // Compute Impulse Response using SSJ
    // Requires solve_steady_state to be called first
    IRFResult compute_irf(int T, const std::string& shock_type);

private:
    int Nm_, Na_, Nz_;
    
    // Cached steady state data for SSJ (set by solve_steady_state)
    bool ss_computed_ = false;
    double r_ss_ = 0.0;
    double w_ss_ = 0.0;
    std::vector<double> c_pol_;    // Na * Nz
    std::vector<double> a_pol_;    // Na * Nz
    std::vector<double> mu_pol_;   // Na * Nz (marginal utility)
    std::vector<double> D_;        // Na * Nz (distribution)
};

}

