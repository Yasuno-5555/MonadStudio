#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Core Components
#include "Params.hpp"
#include "UnifiedGrid.hpp"
#include "AnalyticalSolver.hpp"
#include "ssj/jacobian_builder.hpp"
#include "ssj/ssj_solver.hpp"

using namespace emscripten;
using json = nlohmann::json;

// Helper: Convert Eigen::VectorXd to std::vector for Embind
std::vector<double> eigen_to_std(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

class MonadWasm {
private:
    UnifiedGrid grid;
    MonadParams params;
    
    // Steady State Results
    double r_ss;
    std::vector<double> c_ss; 
    std::vector<double> mu_ss;
    std::vector<double> a_pol_ss; 
    std::vector<double> D_ss; // Flat distribution
    
    // SSJ Objects (Persistent for fast IRF recalc)
    Eigen::SparseMatrix<double> Lambda_ss;
    Monad::JacobianBuilder::PolicyPartials partials;

public:
    MonadWasm() {
        std::cout << "[Wasm] MonadEngine Initialized." << std::endl;
        // Set default grid to avoid uninitialized state
        grid.resize(100);
        for(int i=0; i<100; ++i) grid.nodes[i] = (double)i;
    }

    void load_config(std::string json_str) {
        try {
            auto data = json::parse(json_str);
            std::cout << "[Wasm] Loading Config..." << std::endl;
            
            // 1. Load Parameters
            if (data.contains("parameters")) {
                for (auto& [key, val] : data["parameters"].items()) {
                    if (val.is_number()) {
                        params.scalars[key] = val.get<double>();
                    }
                }
            }

            // 2. Load Grid
            if (data.contains("agents") && data["agents"].size() > 0) {
                auto grid_def = data["agents"][0]["grids"]["asset_a"];
                int size = grid_def.value("n_points", 100);
                if(grid_def.contains("size")) size = grid_def["size"];
                
                double min_val = grid_def.value("min", 0.0);
                double max_val = grid_def.value("max", 100.0);
                std::string type = grid_def.value("type", "Log-spaced");
                double curv = grid_def.value("potency", 2.0);
                if(grid_def.contains("curvature")) curv = grid_def["curvature"];

                grid.resize(size);
                if (type == "Log-spaced") {
                    for(int i=0; i<size; ++i) {
                        double ratio = (double)i / (size - 1);
                        grid.nodes[i] = min_val + (max_val - min_val) * std::pow(ratio, curv);
                    }
                } else {
                    for(int i=0; i<size; ++i) {
                        grid.nodes[i] = min_val + (max_val - min_val) * ((double)i / (size - 1));
                    }
                }
            }
            
            // 3. Load Income
            if (data.contains("income_process")) {
                auto inc_def = data["income_process"];
                params.income.n_z = inc_def.value("n_z", 1);
                
                if (inc_def.contains("z_grid")) 
                    params.income.z_grid = inc_def["z_grid"].get<std::vector<double>>();
                else 
                    params.income.z_grid = {1.0};
                
                if (inc_def.contains("transition_matrix")) 
                    params.income.Pi_flat = inc_def["transition_matrix"].get<std::vector<double>>();
                else 
                    params.income.Pi_flat = {1.0};
            }
        } catch (const std::exception& e) {
            std::cout << "[Wasm Error] Parsing config: " << e.what() << std::endl;
        }
    }

    double solve_ss() {
        try {
            double r_guess = params.get("r_guess", 0.02);
            AnalyticalSolver::solve_steady_state(grid, r_guess, params);
            r_ss = r_guess;
            
            // Retrieve SS policy
            AnalyticalSolver::get_steady_state_policy(grid, r_ss, params, c_ss, mu_ss, a_pol_ss, D_ss);
            
            // Build SSJ Matrices
            Lambda_ss = Monad::JacobianBuilder::build_transition_matrix_2d(a_pol_ss, grid, params.income);
            
            double alpha = params.get("alpha", 0.33); 
            double A = params.get("A", 1.0);
            double K_dem = std::pow(r_ss / (alpha * A), 1.0 / (alpha - 1.0));
            double w_ss = (1.0 - alpha) * A * std::pow(K_dem, alpha);
            
            partials = Monad::JacobianBuilder::compute_partials_2d(grid, params, mu_ss, r_ss, w_ss);
            
            return r_ss;
        } catch (const std::exception& e) {
            std::cout << "[Wasm Error] Solve SS: " << e.what() << std::endl;
            return -1.0;
        }
    }

    // Return view to distribution data (Float32Array compatible if needed, but here double)
    val get_distribution() {
        return val(typed_memory_view(D_ss.size(), D_ss.data()));
    }
    
    val get_asset_grid() {
         return val(typed_memory_view(grid.size, grid.nodes.data()));
    }
    
    int get_nz() {
        return params.income.n_z;
    }

    val solve_irf() {
        try {
            int T = 200;
            double theta_w = params.get("theta_w", 0.75);
            double phi_pi = params.get("phi_pi", 1.5);
            
            auto res = Monad::SsjSolver::solve_nk_wage_transition(
                T, grid, 
                Eigen::Map<Eigen::VectorXd>(D_ss.data(), D_ss.size()), 
                Lambda_ss, a_pol_ss, c_ss, partials, params.income, r_ss, theta_w, phi_pi
            );
            
            val res_obj = val::object();
            res_obj.set("dY", val::array(eigen_to_std(res.dY)));
            res_obj.set("dC", val::array(eigen_to_std(res.dC)));
            res_obj.set("dw", val::array(eigen_to_std(res.dw)));
            res_obj.set("dpi", val::array(eigen_to_std(res.dpi)));
            res_obj.set("di", val::array(eigen_to_std(res.di)));
            return res_obj;
        } catch (const std::exception& e) {
             std::cout << "[Wasm Error] Solve IRF: " << e.what() << std::endl;
             return val::null();
        }
    }
};

EMSCRIPTEN_BINDINGS(monad_module) {
    class_<MonadWasm>("MonadWasm")
        .constructor<>()
        .function("load_config", &MonadWasm::load_config)
        .function("solve_ss", &MonadWasm::solve_ss)
        .function("solve_irf", &MonadWasm::solve_irf)
        .function("get_distribution", &MonadWasm::get_distribution)
        .function("get_asset_grid", &MonadWasm::get_asset_grid)
        .function("get_nz", &MonadWasm::get_nz)
        ;
}
