#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>   
#include <utility>   
#include <cstring>

namespace null {

// supported data types 
enum class DType : int32_t {
    UNDEFINED = 0,
    FLOAT32   = 1,
    UINT8     = 2,
    INT8      = 3,
    UINT16    = 4,
    INT16     = 5,
    INT32     = 6,
    INT64     = 7,
    BOOL      = 9,
    FLOAT64   = 11,
};


inline std::string dtype_to_str(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return "float";
        case DType::FLOAT64: return "double";
        case DType::INT32:   return "int32_t";
        case DType::INT64:   return "int64_t";
        case DType::INT8:    return "int8_t";
        case DType::UINT8:   return "uint8_t";
        case DType::INT16:   return "int16_t";
        case DType::UINT16:  return "uint16_t";
        case DType::BOOL:    return "uint8_t";
        default:             return "void";
    }
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::INT32:   return 4;
        case DType::INT64:   return 8;
        case DType::INT8:    return 1;
        case DType::UINT8:   return 1;
        case DType::INT16:   return 2;
        case DType::UINT16:  return 2;
        case DType::BOOL:    return 1;
        default:             return 0;
    }
} 

// core tensor structure, metadata + optional constant data
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;    // -1 is dynamic / unknown
    DType dtype = DType::UNDEFINED;
    std::vector<uint8_t> data;     // raw bytes for constant tensors (initalizers)
    bool is_constant = false;

    Tensor() = default;
    Tensor(std::string n, std::vector<int64_t> s, DType dt)
        : name(std::move(n)), shape(std::move(s)), dtype(dt) {}

    // total number of elements
    int64_t num_elements() const {
        if (shape.empty()) return 1; // scalar
        int64_t total = 1;
        for (int64_t d : shape) {
            if (d < 0) return -1; // dynamic
            total *= d;
        }
        return total;
    }

    // total bytes needed for this tensor
    int64_t byte_size() const {
        int64_t n = num_elements();
        if (n < 0) return -1;
        return n * static_cast<int64_t>(dtype_size(dtype));
    }

    // set float data as constant
    void set_float_data(const std::vector<float>& values) {
        dtype = DType::FLOAT32;
        is_constant = true;
        data.resize(values.size() * 4);
        std::memcpy(data.data(), values.data(), data.size());
        if (shape.empty()) shape = {static_cast<int64_t>(values.size())};
    }

    void set_int64_data(const std::vector<int64_t>& values) {
        dtype = DType::INT64;
        is_constant = true;
        data.resize(values.size() * 8);
        std::memcpy(data.data(), values.data(), data.size());
        if (shape.empty()) shape = {static_cast<int64_t>(values.size())};
    }

    // access constant data as typed view
    const float* float_ptr() const {
        if (dtype != DType::FLOAT32 || data.empty()) return nullptr;
        return reinterpret_cast<const float*>(data.data());
    }

    const int64_t* int64_ptr() const {
        if (dtype != DType::INT64 || data.empty()) return nullptr;
        return reinterpret_cast<const int64_t*>(data.data());
    }

    std::string shape_str() const {
        if (shape.empty()) return "[]";
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        return s + "]";
    }

    bool same_shape(const Tensor& other) const {
        return shape == other.shape;
    }
};

} // namespace null
