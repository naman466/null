#pragma once
#include <vector>
#include <variant>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// supported operator types
// mirrors a subset of ONNX standard operators
enum class OpType {
    // arithmetic
    Add, Sub, Mul, Div,
    // activation
    Relu, Sigmoid, Tanh, Softmax,
    // linear algebra
    Gemm, MatMul,
    // convolution
    Conv,
    // pooling
    MaxPool, AveragePool, GlobalAveragePool,
    // normalization
    BatchNormalization,
    // shape
    Reshape, Flatten, Transpose, Squeeze, Unsqueeze,
    // elementwise
    Exp, Log, Sqrt, Abs, Neg, Pow,
    // reduction
    ReduceMean, ReduceSum, ReduceMax,
    // data
    Gather, Concat, Split, Slice,
    // misc
    Dropout, Cast, Clip,
    // fused (created by optimizer)
    ReluFused,    // e.g., Gemm + Relu
    // constant
    Constant,
    // identity (no-op, used internally)
    Identity,
    // unknown / passthrough
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

// a single computation node in the graph
struct Node {
    std::string name;
    OpType op;
    std::vector<std::string> inputs;   // tensor names consumed
    std::vector<std::string> outputs;  // tensor names produced
    std::unordered_map<std::string, Attribute> attrs;
    bool is_dead = false; // marked for DCE

    Node() : op(OpType::Unknown) {}
    Node(std::string n, OpType o,
         std::vector<std::string> ins,
         std::vector<std::string> outs)
        : name(std::move(n)), op(o),
          inputs(std::move(ins)), outputs(std::move(outs)) {}

    void add_attr(const std::string& key, AttrValue val) {
        attrs.emplace(key, Attribute{key, std::move(val)});
    }

    bool has_attr(const std::string& key) const {
        return attrs.count(key) > 0;
    }

    const Attribute& get_attr(const std::string& key) const {
        auto it = attrs.find(key);
        if (it == attrs.end())
            throw std::runtime_error("Node '" + name + "' missing attribute '" + key + "'");
        return it->second;
    }

    // Convenience: try to get attr with default
    int64_t attr_int(const std::string& key, int64_t def = 0) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return def;
        return it->second.as_int();
    }

    float attr_float(const std::string& key, float def = 0.0f) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return def;
        return it->second.as_float();
    }

    std::string attr_str(const std::string& key, std::string def = "") const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return def;
        return it->second.as_string();
    }

    std::vector<int64_t> attr_ints(const std::string& key,
                                    std::vector<int64_t> def = {}) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return def;
        return it->second.as_ints();
    }

    std::vector<float> attr_floats(const std::string& key,
                                    std::vector<float> def = {}) const {
        auto it = attrs.find(key);
        if (it == attrs.end()) return def;
        return it->second.as_floats();
    }
};

} // namespace null
