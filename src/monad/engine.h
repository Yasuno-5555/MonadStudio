#pragma once
#include "steady_state.h"

namespace monad {

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

private:
    int Nm_, Na_, Nz_;
};

}
