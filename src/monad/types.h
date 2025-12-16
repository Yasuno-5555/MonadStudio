#pragma once
#include <vector>

namespace monad {

using Vector = std::vector<double>;

struct Distribution3D {
    int Nm, Na, Nz;
    std::vector<double> data; // row-major

    // Constructor to initialize vector size
    Distribution3D() : Nm(0), Na(0), Nz(0) {}
    Distribution3D(int nm, int na, int nz) : Nm(nm), Na(na), Nz(nz), data(nm * na * nz) {}

    inline double& operator()(int i, int j, int k) {
        return data[(i*Na + j)*Nz + k];
    }
    
    inline const double& operator()(int i, int j, int k) const {
        return data[(i*Na + j)*Nz + k];
    }
};

}
