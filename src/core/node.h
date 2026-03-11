#pragma once
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
#include <string>
#include "tensor.h"

namespace null {

// supported attribute value types
using AttrValue = std::variant<
    int64_t,            // INT
    float,              // FLOAT
    std::string,        // STRING
    std::vector<int64_t>,   // INTS
    std::vector<float>,     // FLOATS
    Tensor              // TENSOR (embedded constant)
>;

// attribute type tag for printing or debugging
enum class AttrType { INT, FLOAT, STRING, INTS, FLOATS, TENSOR };

struct Attribute {
    std::string name;
    AttrValue   value;

    Attribute() = default;
    Attribute(std::string n, AttrValue v) : name(std::move(n)), value(std::move(v)) {}

    int64_t as_int() const {
        if (auto* p = std::get_if<int64_t>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not INT");
    }
    float as_float() const {
        if (auto* p = std::get_if<float>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not FLOAT");
    }
    const std::string& as_string() const {
        if (auto* p = std::get_if<std::string>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not STRING");
    }
    const std::vector<int64_t>& as_ints() const {
        if (auto* p = std::get_if<std::vector<int64_t>>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not INTS");
    }
    const std::vector<float>& as_floats() const {
        if (auto* p = std::get_if<std::vector<float>>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not FLOATS");
    }
    const Tensor& as_tensor() const {
        if (auto* p = std::get_if<Tensor>(&value)) return *p;
        throw std::runtime_error("Attribute '" + name + "' is not TENSOR");
    }

    bool has_int()    const { return std::holds_alternative<int64_t>(value); }
    bool has_float()  const { return std::holds_alternative<float>(value); }
    bool has_string() const { return std::holds_alternative<std::string>(value); }
    bool has_ints()   const { return std::holds_alternative<std::vector<int64_t>>(value); }
    bool has_floats() const { return std::holds_alternative<std::vector<float>>(value); }
    bool has_tensor() const { return std::holds_alternative<Tensor>(value); }
};

// Supported operator types
// Mirrors a subset of ONNX standard operators
enum class OpType {
    // Arithmetic
    Add, Sub, Mul, Div,
    // Activation
    Relu, Sigmoid, Tanh, Softmax,
    // Linear algebra
    Gemm, MatMul,
    // Convolution
    Conv,
    // Pooling
    MaxPool, AveragePool, GlobalAveragePool,
    // Normalization
    BatchNormalization,
    // Shape
    Reshape, Flatten, Transpose, Squeeze, Unsqueeze,
    // Elementwise
    Exp, Log, Sqrt, Abs, Neg, Pow,
    // Reduction
    ReduceMean, ReduceSum, ReduceMax,
    // Data
    Gather, Concat, Split, Slice,
    // Misc
    Dropout, Cast, Clip,
    // Fused (created by optimizer)
    ReluFused,    // e.g., Gemm + Relu
    // Constant
    Constant,
    // Identity (no-op, used internally)
    Identity,
    // Unknown / passthrough
    Unknown
};

inline std::string optype_to_str(OpType op) {
    switch (op) {
        case OpType::Add:               return "Add";
        case OpType::Sub:               return "Sub";
        case OpType::Mul:               return "Mul";
        case OpType::Div:               return "Div";
        case OpType::Relu:              return "Relu";
        case OpType::Sigmoid:           return "Sigmoid";
        case OpType::Tanh:              return "Tanh";
        case OpType::Softmax:           return "Softmax";
        case OpType::Gemm:              return "Gemm";
        case OpType::MatMul:            return "MatMul";
        case OpType::Conv:              return "Conv";
        case OpType::MaxPool:           return "MaxPool";
        case OpType::AveragePool:       return "AveragePool";
        case OpType::GlobalAveragePool: return "GlobalAveragePool";
        case OpType::BatchNormalization:return "BatchNormalization";
        case OpType::Reshape:           return "Reshape";
        case OpType::Flatten:           return "Flatten";
        case OpType::Transpose:         return "Transpose";
        case OpType::Squeeze:           return "Squeeze";
        case OpType::Unsqueeze:         return "Unsqueeze";
        case OpType::Exp:               return "Exp";
        case OpType::Log:               return "Log";
        case OpType::Sqrt:              return "Sqrt";
        case OpType::Abs:               return "Abs";
        case OpType::Neg:               return "Neg";
        case OpType::Pow:               return "Pow";
        case OpType::ReduceMean:        return "ReduceMean";
        case OpType::ReduceSum:         return "ReduceSum";
        case OpType::ReduceMax:         return "ReduceMax";
        case OpType::Gather:            return "Gather";
        case OpType::Concat:            return "Concat";
        case OpType::Split:             return "Split";
        case OpType::Slice:             return "Slice";
        case OpType::Dropout:           return "Dropout";
        case OpType::Cast:              return "Cast";
        case OpType::Clip:              return "Clip";
        case OpType::ReluFused:         return "ReluFused";
        case OpType::Constant:          return "Constant";
        case OpType::Identity:          return "Identity";
        default:                        return "Unknown";
    }
}

inline OpType str_to_optype(const std::string& s) {
    if (s == "Add")               return OpType::Add;
    if (s == "Sub")               return OpType::Sub;
    if (s == "Mul")               return OpType::Mul;
    if (s == "Div")               return OpType::Div;
    if (s == "Relu")              return OpType::Relu;
    if (s == "Sigmoid")           return OpType::Sigmoid;
    if (s == "Tanh")              return OpType::Tanh;
    if (s == "Softmax")           return OpType::Softmax;
    if (s == "Gemm")              return OpType::Gemm;
    if (s == "MatMul")            return OpType::MatMul;
    if (s == "Conv")              return OpType::Conv;
    if (s == "MaxPool")           return OpType::MaxPool;
    if (s == "AveragePool")       return OpType::AveragePool;
    if (s == "GlobalAveragePool") return OpType::GlobalAveragePool;
    if (s == "BatchNormalization") return OpType::BatchNormalization;
    if (s == "Reshape")           return OpType::Reshape;
    if (s == "Flatten")           return OpType::Flatten;
    if (s == "Transpose")         return OpType::Transpose;
    if (s == "Squeeze")           return OpType::Squeeze;
    if (s == "Unsqueeze")         return OpType::Unsqueeze;
    if (s == "Exp")               return OpType::Exp;
    if (s == "Log")               return OpType::Log;
    if (s == "Sqrt")              return OpType::Sqrt;
    if (s == "Abs")               return OpType::Abs;
    if (s == "Neg")               return OpType::Neg;
    if (s == "Pow")               return OpType::Pow;
    if (s == "ReduceMean")        return OpType::ReduceMean;
    if (s == "ReduceSum")         return OpType::ReduceSum;
    if (s == "ReduceMax")         return OpType::ReduceMax;
    if (s == "Gather")            return OpType::Gather;
    if (s == "Concat")            return OpType::Concat;
    if (s == "Split")             return OpType::Split;
    if (s == "Slice")             return OpType::Slice;
    if (s == "Dropout")           return OpType::Dropout;
    if (s == "Cast")              return OpType::Cast;
    if (s == "Clip")              return OpType::Clip;
    if (s == "Constant")          return OpType::Constant;
    if (s == "Identity")          return OpType::Identity;
    return OpType::Unknown;
}


} // namespace null
