#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::string
#include <VAD_customVDB.hpp>

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "pybind11 bindings for EGO_VDB KDTree";

    py::class_<KDTree>(m, "KDTree")
        // constructor
        .def(py::init<>())

        // load_data
        .def("load_data", &KDTree::load_data,
             py::arg("json_path"),
             "Loads the VAD data from a JSON file.")

        // search
        .def("VAD_search_near_k", &KDTree::VAD_search_near_k,
             "Searches for nearest emotions in the VAD space.",
             py::arg("V"),
             py::arg("A"),
             py::arg("D"),
             py::arg("k"),
             py::arg("d"),
             py::arg("SIGMA"),
             py::arg("opt") = "knn"
        );
    
}