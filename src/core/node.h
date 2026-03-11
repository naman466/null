#pragma once
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
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

} // namespace null
