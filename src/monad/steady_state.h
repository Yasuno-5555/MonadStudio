#pragma once
#include "types.h"

namespace monad {

struct SteadyStateResult {
    double r;
    double w;
    double Y;
    double C;
    Distribution3D distribution;
};

}
