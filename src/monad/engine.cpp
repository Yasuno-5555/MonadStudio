#include "engine.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// Include Real Solver
#include "../AnalyticalSolver.hpp"

// Include SSJ Components
#include "../ssj/jacobian_builder.hpp"
#include "../ssj/aggregator.hpp"

namespace monad {

// Helper: Power Grid Generator (concentration near min) - from main_two_asset.cpp
static UnifiedGrid make_grid(int size, double min, double max, double curv) {
    UnifiedGrid g;
    g.resize(size);
    for(int i=0; i<size; ++i) {
        double t = (double)i / (size - 1);
        g.nodes[i] = min + (max - min) * std::pow(t, curv);
    }
    return g;
}

// Helper: Simple Income Process (Example) - from main_two_asset.cpp
static IncomeProcess make_income() {
    // 2-state simple process
    // z = [0.8, 1.2], Pi = [[0.9, 0.1], [0.1, 0.9]]
    IncomeProcess p;
    p.n_z = 2;
    p.z_grid = {0.8, 1.2};
    p.Pi_flat = {0.9, 0.1, 
                 0.1, 0.9};
    // Stationary distribution [0.5, 0.5] implicitly
    return p;
}

// Cached grid and params (static for now, will make member later if needed)
static UnifiedGrid cached_grid_;
static MonadParams cached_params_;

MonadEngine::MonadEngine(int Nm, int Na, int Nz)
    : Nm_(Nm), Na_(Na), Nz_(Nz) {}

SteadyStateResult MonadEngine::solve_steady_state(
    double beta,
    double sigma,
    double chi0,
    double chi1,
    double chi2
) {
    SteadyStateResult res;

    // 1. Setup Parameters
    MonadParams params;
    params.scalars["beta"] = beta;
    params.scalars["sigma"] = sigma;
    
    // Defaults for Solver
    params.scalars["alpha"] = 0.33;
    params.scalars["A"] = 1.0;
    params.scalars["tax_lambda"] = 1.0;

    // Setup Income Process
    params.income = make_income();

    // 2. Setup Grid
    UnifiedGrid grid = make_grid(Na_, 0.0, 50.0, 2.0);

    // 3. Call Solver
    double r_guess = 0.02;
    if (beta > 0.99) r_guess = 0.005;

    try {
        // A. Solve for Equilibrium Price (r)
        AnalyticalSolver::solve_steady_state(grid, r_guess, params);
        
        res.r = r_guess;
        
        // B. Recover Aggregate Variables & Distribution
        std::vector<double> c_pol, mu_pol, a_pol, D_flat;
        AnalyticalSolver::get_steady_state_policy(grid, res.r, params, c_pol, mu_pol, a_pol, D_flat);

        // *** CACHE FOR SSJ ***
        c_pol_ = c_pol;
        a_pol_ = a_pol;
        mu_pol_ = mu_pol;
        D_ = D_flat;
        r_ss_ = res.r;
        
        // Calculate w_ss for later use
        double alpha = params.get("alpha", 0.33);
        double A = params.get("A", 1.0);
        double K_dem = std::pow(res.r / (alpha * A), 1.0/(alpha-1.0));
        w_ss_ = (1.0 - alpha) * A * std::pow(K_dem, alpha);
        
        // Cache grid and params for SSJ
        cached_grid_ = grid;
        cached_params_ = params;
        
        ss_computed_ = true;
        // *** END CACHE ***

        // C. Calculate Aggregates
        double K = 0.0;
        for(size_t i=0; i<D_flat.size(); ++i) {
            int a_idx = i % Na_;
            K += D_flat[i] * grid.nodes[a_idx];
        }
        
        res.Y = A * std::pow(K, alpha);
        res.w = (1.0 - alpha) * res.Y;
        res.C = res.Y;

        // D. Pack Distribution (2D -> 3D Mock)
        res.distribution = Distribution3D(Nm_, Na_, Nz_);
        
        int solver_nz = params.income.n_z;
        
        for(int j=0; j<solver_nz && j<Nz_; ++j) {
            for(int i=0; i<Na_; ++i) {
                int solver_idx = j*Na_ + i;
                double mass = D_flat[solver_idx];
                res.distribution(0, i, j) = mass;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Solver Error: " << e.what() << std::endl;
        res.r = -999.0; 
        res.Y = 0.0;
        ss_computed_ = false;
    }

    return res;
}

// ============================================================================
// SSJ: Compute Impulse Response Function
// ============================================================================
IRFResult MonadEngine::compute_irf(int T, const std::string& shock_type) {
    IRFResult result;
    result.T = T;
    result.dC.resize(T, 0.0);
    result.dY.resize(T, 0.0);
    result.dr.resize(T, 0.0);
    
    if (!ss_computed_) {
        std::cerr << "[SSJ] Error: Steady state not computed. Call solve_steady_state first." << std::endl;
        return result;
    }
    
    std::cout << "[SSJ] Computing IRF for shock type: " << shock_type << ", T=" << T << std::endl;
    
    try {
        // 1. Compute Policy Partials using Dual numbers
        auto partials = Monad::JacobianBuilder::compute_partials_2d(
            cached_grid_, cached_params_, mu_pol_, r_ss_, w_ss_
        );
        
        std::cout << "[SSJ] Partials computed. da_dr[0]=" << partials.da_dr[0] 
                  << ", dc_dr[0]=" << partials.dc_dr[0] << std::endl;
        
        // 2. Build Transition Matrix
        auto Lambda = Monad::JacobianBuilder::build_transition_matrix_2d(
            a_pol_, cached_grid_, cached_params_.income
        );
        
        std::cout << "[SSJ] Transition matrix: " << Lambda.rows() << "x" << Lambda.cols() 
                  << ", nnz=" << Lambda.nonZeros() << std::endl;
        
        // 3. Convert SS distribution to Eigen Vector
        Eigen::VectorXd D_vec = Eigen::Map<Eigen::VectorXd>(D_.data(), D_.size());
        Eigen::VectorXd c_ss_vec = Eigen::Map<Eigen::VectorXd>(c_pol_.data(), c_pol_.size());
        
        // 4. Compute Consumption IRF (Partial Equilibrium)
        //    This is the response of aggregate C to a shock in r at t=0
        auto dC_irf = Monad::JacobianAggregator::build_consumption_impulse_response(
            T, cached_grid_, D_vec, Lambda, a_pol_, c_ss_vec,
            partials.da_dr, partials.dc_dr, cached_params_.income
        );
        
        std::cout << "[SSJ] dC_irf computed. dC[0]=" << dC_irf[0] << std::endl;
        
        // 5. Copy results
        for(int t=0; t<T; ++t) {
            result.dC[t] = dC_irf[t];
            // For PE analysis, dr is the shock itself (decaying)
            result.dr[t] = (t == 0) ? 0.01 : result.dr[t-1] * 0.8;  // 1% shock, rho=0.8
            // In PE, dY ~ dC (closed economy)
            result.dY[t] = dC_irf[t];
        }
        
        std::cout << "[SSJ] IRF complete. Peak |dC|=" << dC_irf.cwiseAbs().maxCoeff() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[SSJ] Error computing IRF: " << e.what() << std::endl;
    }
    
    return result;
}

} // namespace monad

