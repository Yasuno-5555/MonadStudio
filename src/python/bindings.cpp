#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../monad/engine.h"

namespace py = pybind11;
using namespace monad;

PYBIND11_MODULE(monad_core, m) {
    py::class_<MonadEngine>(m, "MonadEngine")
        .def(py::init<int,int,int>())
        .def("solve_steady_state", &MonadEngine::solve_steady_state,
             py::arg("beta"), py::arg("sigma"), 
             py::arg("chi0"), py::arg("chi1"), py::arg("chi2"));

    py::class_<SteadyStateResult>(m, "SteadyStateResult")
        .def_readonly("r", &SteadyStateResult::r)
        .def_readonly("w", &SteadyStateResult::w)
        .def_readonly("Y", &SteadyStateResult::Y)
        .def_readonly("C", &SteadyStateResult::C)
        .def_property_readonly(
            "distribution",
            [](const SteadyStateResult& ss) {
                // Zero-copy access to the underlying vector
                // Note: The python object must keep 'ss' alive if we were just viewing it,
                // but here we are creating a copy of the view or return-by-value?
                // Actually, 'ss' is likely a temporary returned by solve_steady_state.
                // If usage is: res = engine.solve_ss(); dist = res.distribution
                // Then 'res' holds the C++ object. pybind11 keeps it alive.
                // The array_t will share memory if we pass the pointer carefully or copy.
                // For Phase 0 Safety: Let's create a copy py::array to be safe or use appropriate owner.
                // Spec says "view only", but simple vector data pointer is dangerous if the parent dies.
                // Since SteadyStateResult is returned by value to Python, Python owns it.
                // So mapping the data pointer is safe as long as the SteadyStateResult object is alive.
                // To keep SteadyStateResult alive while array is used, we can use 'base' argument in array_t?
                // Or just copy for now to be absolutely safe (and not premature optimization).
                // "Python 側に渡す時は py::array_t<double> に view するだけ" -> implies view.
                
                return py::array_t<double>(
                    {ss.distribution.Nm, ss.distribution.Na, ss.distribution.Nz}, // shape
                    {ss.distribution.Na * ss.distribution.Nz * sizeof(double),    // strides (bytes)
                     ss.distribution.Nz * sizeof(double),
                     sizeof(double)},
                    ss.distribution.data.data(), // pointer
                    py::cast(ss) // base object (keep alive)
                );
            }
        );
}
