#include "analysis/shape_inference.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_map>

namespace null {

namespace {

// the elements of a small int64 tensor known at compile time, row-major. an
// element is nullopt where it depends on a dynamic dim: Shape of [?, 784] is
// {nullopt, 784}, and becomes {1, 784} once --batch 1 fixes the dim
using Value = std::vector<std::optional<int64_t>>;

// values are computed for shape arithmetic, never for weights: larger results
// are left unknown
constexpr int64_t kMaxValueElements = 4096;

// result of a shape rule for one output
struct TypeInfo {
    DType dtype = DType::UNDEFINED;
    std::vector<int64_t> shape;
    std::optional<Value> value;     // only for int64 outputs computed at compile time

    TypeInfo() = default;
    TypeInfo(DType d, std::vector<int64_t> s, std::optional<Value> v = std::nullopt)
        : dtype(d), shape(std::move(s)), value(std::move(v)) {}
};

std::string where(const Node& n) {
    return "node '" + n.name + "' (" + optype_to_str(n.op) + ")";
}

// everything a rule needs about one node
class Context {
public:
    // values: the partly known values computed so far in this pass, if any
    Context(const Node& n, const Graph& g, const std::unordered_map<std::string, Value>* values = nullptr)
        : node(n), graph_(g), values_(values) {
        for (const std::string& name : n.inputs) {
            if (name.empty()) {          // omitted optional input
                inputs_.push_back(nullptr);
                continue;
            }
            const Tensor* t = g.get_tensor_info(name);
            if (!t) fail("input '" + name + "' has no type or shape");
            inputs_.push_back(t);
        }
    }

    const Node& node;

    size_t num_inputs() const { return inputs_.size(); }
    bool has(size_t i) const { return i < inputs_.size() && inputs_[i]; }

    const Tensor& in(size_t i) const {
        if (!has(i)) fail("missing required input " + std::to_string(i));
        return *inputs_[i];
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw ShapeError(where(node) + ": " + msg);
    }

    void require_inputs(size_t min, size_t max) const {
        if (inputs_.size() < min || inputs_.size() > max) {
            std::string range = min == max ? std::to_string(min)
                                           : std::to_string(min) + " to " + std::to_string(max);
            fail("expects " + range + " inputs, got " + std::to_string(inputs_.size()));
        }
    }

    void require_float(size_t i) const {
        DType dt = in(i).dtype;
        if (dt != DType::FLOAT32 && dt != DType::FLOAT64)
            fail("input " + std::to_string(i) + " must be float32 or float64, got " +
                 dtype_name(dt));
    }

    void require_numeric(size_t i) const {
        DType dt = in(i).dtype;
        if (dt == DType::BOOL || dt == DType::UNDEFINED)
            fail("input " + std::to_string(i) + " must be numeric, got " + dtype_name(dt));
    }

    void require_same_dtype(size_t i, size_t j) const {
        if (in(i).dtype != in(j).dtype)
            fail("inputs " + std::to_string(i) + " and " + std::to_string(j) +
                 " have different types (" + dtype_name(in(i).dtype) + " and " +
                 dtype_name(in(j).dtype) + ")");
    }

    void require_scalar(size_t i) const {
        if (has(i) && !in(i).shape.empty() && in(i).num_elements() != 1)
            fail("input " + std::to_string(i) + " must be a scalar, got shape " + in(i).shape_str());
    }

    // the value of an int64 input known at compile time: an initializer, a
    // Constant node (value, value_int or value_ints), or a value shape
    // inference computed, like the output of Shape and what is built from it.
    // nullopt if the input has none (a graph input, an int32 tensor, ...)
    std::optional<Value> value_of(size_t i) const {
        if (!has(i)) return std::nullopt;
        const std::string& name = node.inputs[i];
        auto ints = [](const std::vector<int64_t>& v) { return Value(v.begin(), v.end()); };
        const Tensor* t = nullptr;
        if (auto it = graph_.initializers.find(name); it != graph_.initializers.end()) {
            t = &it->second;
        } else if (const Node* n = constant_node(name)) {
            if (n->has_attr("value_ints")) return ints(n->attr_ints("value_ints"));
            if (n->has_attr("value_int"))  return ints({n->attr_int("value_int")});
            if (n->has_attr("value"))      t = &n->get_attr("value").as_tensor();
        } else if (values_ && values_->count(name)) {
            return values_->at(name);
        } else if (!inputs_[i]->data.empty()) {
            t = inputs_[i];                 // computed by an earlier pass and kept with the tensor
        }
        if (!t || t->dtype != DType::INT64) return std::nullopt;
        int64_t n = t->num_elements();
        if (n <= 0) return Value{};
        return ints(std::vector<int64_t>(t->int64_ptr(), t->int64_ptr() + n));
    }

    // the value of an int64 input that must be fully known at compile time
    std::vector<int64_t> const_ints(size_t i) const {
        const std::string& name = node.inputs.at(i);
        std::optional<Value> v = value_of(i);
        if (!v) {
            if (in(i).dtype != DType::INT64)
                fail("input " + std::to_string(i) + " ('" + name + "') must be int64, got " +
                     dtype_name(in(i).dtype));
            fail("input " + std::to_string(i) + " ('" + name + "') must be a constant "
                 "(an initializer or Constant node), shapes computed at run time are not supported");
        }
        std::vector<int64_t> out;
        for (const std::optional<int64_t>& e : *v) {
            if (!e)
                fail("input " + std::to_string(i) + " ('" + name + "') depends on a dynamic dim, fix the "
                     "input shapes with --batch or --input-shape");
            out.push_back(*e);
        }
        return out;
    }

private:
    const Node* constant_node(const std::string& name) const {
        for (const Node& n : graph_.nodes)
            if (!n.is_dead && n.op == OpType::Constant && !n.outputs.empty() && n.outputs[0] == name) return &n;
        return nullptr;
    }

    std::vector<const Tensor*> inputs_;
    const Graph& graph_;
    const std::unordered_map<std::string, Value>* values_;
};

using Rule = std::vector<TypeInfo> (*)(const Context&);

// product of dims, -1 if any is dynamic
int64_t product(const std::vector<int64_t>& dims, size_t begin, size_t end) {
    int64_t p = 1;
    for (size_t i = begin; i < end; ++i) {
        if (dims[i] < 0) return -1;
        if (__builtin_mul_overflow(p, dims[i], &p)) return -1;
    }
    return p;
}

// RULES - ELEMENTWISE

// same type and shape as input 0, numeric input
std::vector<TypeInfo> unary_numeric(const Context& c) {
    c.require_inputs(1, 1);
    c.require_numeric(0);
    return {{c.in(0).dtype, c.in(0).shape}};
}

// same type and shape as input 0, float input
std::vector<TypeInfo> unary_float(const Context& c) {
    c.require_inputs(1, 1);
    c.require_float(0);
    return {{c.in(0).dtype, c.in(0).shape}};
}

std::vector<TypeInfo> identity(const Context& c) {
    c.require_inputs(1, 1);
    return {{c.in(0).dtype, c.in(0).shape}};
}

// Clip(x, min?, max?), min and max are optional scalars of x's type
std::vector<TypeInfo> clip(const Context& c) {
    c.require_inputs(1, 3);
    c.require_numeric(0);
    for (size_t i = 1; i < c.num_inputs(); ++i) {
        if (!c.has(i)) continue;
        c.require_scalar(i);
        c.require_same_dtype(0, i);
    }
    return {{c.in(0).dtype, c.in(0).shape}};
}

// Dropout(data, ratio?, training_mode?) -> output, mask?
// inference only: output is data, mask is all true
std::vector<TypeInfo> dropout(const Context& c) {
    c.require_inputs(1, 3);
    c.require_float(0);
    c.require_scalar(1);
    c.require_scalar(2);
    return {{c.in(0).dtype, c.in(0).shape}, {DType::BOOL, c.in(0).shape}};
}

// Cast(x) -> x converted to the type in attribute "to"
std::vector<TypeInfo> cast(const Context& c) {
    c.require_inputs(1, 1);
    if (!c.node.has_attr("to")) c.fail("missing attribute 'to'");
    // DType values mirror onnx's data type numbers
    int64_t to = c.node.attr_int("to");
    DType dt = static_cast<DType>(to);
    if (to <= 0 || to > 255 || dtype_size(dt) == 0)
        c.fail("unsupported target type " + std::to_string(to));
    return {{dt, c.in(0).shape}};
}

// one int64 result of Add, Sub, Mul or Div, if it is certain: nullopt on
// overflow, division by zero, or a division whose rounding direction matters
// (onnxruntime truncates, numpy floors; they agree when the operands have the
// same sign or the division is exact)
std::optional<int64_t> int_arith(OpType op, int64_t a, int64_t b) {
    int64_t r;
    switch (op) {
        case OpType::Add: if (__builtin_add_overflow(a, b, &r)) return std::nullopt; return r;
        case OpType::Sub: if (__builtin_sub_overflow(a, b, &r)) return std::nullopt; return r;
        case OpType::Mul: if (__builtin_mul_overflow(a, b, &r)) return std::nullopt; return r;
        case OpType::Div:
            if (b == 0 || (a == INT64_MIN && b == -1)) return std::nullopt;
            if ((a < 0) != (b < 0) && a % b != 0) return std::nullopt;
            return a / b;
        default: return std::nullopt;
    }
}

// Add, Sub, Mul, Div: same types, broadcast shapes. int64 values are computed
// when both inputs have them: exporters do shape arithmetic this way, like
// PyTorch's chunk, which splits at (C + 1) / 2
std::vector<TypeInfo> binary_arith(const Context& c) {
    c.require_inputs(2, 2);
    c.require_numeric(0);
    c.require_same_dtype(0, 1);
    std::vector<int64_t> out;
    try {
        out = broadcast_shapes(c.in(0).shape, c.in(1).shape);
    } catch (const ShapeError& e) {
        c.fail(e.what());
    }
    TypeInfo result{c.in(0).dtype, out, std::nullopt};
    std::optional<Value> a = c.value_of(0), b = c.value_of(1);
    const int64_t total = product(out, 0, out.size());
    if (!a || !b || total < 0 || total > kMaxValueElements) return {result};
    // output index -> input index: dims are aligned from the right, and a
    // dim of 1 in an input repeats its one element
    auto at = [&](const std::vector<int64_t>& s, int64_t k) {
        int64_t index = 0, scale = 1;
        for (size_t d = out.size(); d-- > 0;) {
            const int64_t pos = k % out[d];
            k /= out[d];
            const size_t from_end = out.size() - 1 - d;
            if (from_end >= s.size()) continue;
            const int64_t dim = s[s.size() - 1 - from_end];
            if (dim != 1) index += pos * scale;
            scale *= dim;
        }
        return static_cast<size_t>(index);
    };
    Value v;
    for (int64_t k = 0; k < total; ++k) {
        const std::optional<int64_t>& x = (*a)[at(c.in(0).shape, k)];
        const std::optional<int64_t>& y = (*b)[at(c.in(1).shape, k)];
        v.push_back(x && y ? int_arith(c.node.op, *x, *y) : std::nullopt);
    }
    result.value = v;
    return {result};
}

// Pow(base, exponent): result has the base's type, the exponent may differ
std::vector<TypeInfo> pow_rule(const Context& c) {
    c.require_inputs(2, 2);
    c.require_numeric(0);
    c.require_numeric(1);
    try {
        return {{c.in(0).dtype, broadcast_shapes(c.in(0).shape, c.in(1).shape)}};
    } catch (const ShapeError& e) {
        c.fail(e.what());
    }
}

// Constant() -> the tensor in its value attribute, or one of the
// value_float / value_floats / value_int / value_ints shorthands
std::vector<TypeInfo> constant(const Context& c) {
    c.require_inputs(0, 0);
    const Node& n = c.node;
    if (n.has_attr("value")) {
        const Tensor& t = n.get_attr("value").as_tensor();
        return {{t.dtype, t.shape}};
    }
    if (n.has_attr("value_float")) return {{DType::FLOAT32, {}}};
    if (n.has_attr("value_int"))   return {{DType::INT64, {}}};
    if (n.has_attr("value_floats"))
        return {{DType::FLOAT32, {static_cast<int64_t>(n.attr_floats("value_floats").size())}}};
    if (n.has_attr("value_ints"))
        return {{DType::INT64, {static_cast<int64_t>(n.attr_ints("value_ints").size())}}};
    c.fail("no supported value attribute (value, value_float(s), value_int(s))");
}

// RULES - LINEAR ALGEBRA

std::string shape_of(const std::vector<int64_t>& s) {
    return Tensor("", s, DType::UNDEFINED).shape_str();
}

// element types onnx allows for MatMul and Gemm (that null supports)
void require_matmul_type(const Context& c, size_t i) {
    DType dt = c.in(i).dtype;
    if (dt != DType::FLOAT32 && dt != DType::FLOAT64 && dt != DType::INT32 && dt != DType::INT64)
        c.fail("input " + std::to_string(i) + " must be float32, float64, int32 or int64, got " +
               dtype_name(dt));
}

// MatMul(A, B), numpy matmul semantics
//   [.., M, K] x [.., K, N] -> [broadcast(..), M, N]
//   a 1-D A is treated as [1, K] and a 1-D B as [K, 1], the added dim is
//   removed from the result, so [K] x [K] -> []
std::vector<TypeInfo> matmul(const Context& c) {
    c.require_inputs(2, 2);
    require_matmul_type(c, 0);
    c.require_same_dtype(0, 1);

    std::vector<int64_t> a = c.in(0).shape, b = c.in(1).shape;
    if (a.empty() || b.empty())
        c.fail("inputs must be at least 1-D, got " + shape_of(a) + " and " + shape_of(b));

    const bool a_vec = a.size() == 1, b_vec = b.size() == 1;
    if (a_vec) a.insert(a.begin(), 1);
    if (b_vec) b.push_back(1);

    int64_t ka = a[a.size() - 1], kb = b[b.size() - 2];
    if (ka >= 0 && kb >= 0 && ka != kb)
        c.fail("inner dimensions differ: " + shape_of(c.in(0).shape) + " x " +
               shape_of(c.in(1).shape) + " (" + std::to_string(ka) + " vs " +
               std::to_string(kb) + ")");

    // leading (batch) dims broadcast like elementwise ops
    std::vector<int64_t> out;
    try {
        out = broadcast_shapes({a.begin(), a.end() - 2}, {b.begin(), b.end() - 2});
    } catch (const ShapeError& e) {
        c.fail(std::string("batch dimensions: ") + e.what());
    }
    if (!a_vec) out.push_back(a[a.size() - 2]);
    if (!b_vec) out.push_back(b[b.size() - 1]);
    return {{c.in(0).dtype, out}};
}

// Gemm(A, B, C?) = alpha * A' * B' + beta * C, with A' = transpose(A) if
// transA and B' = transpose(B) if transB
//   A' is [M, K], B' is [K, N], the result is [M, N]
//   C is optional and must broadcast to [M, N] without changing it
std::vector<TypeInfo> gemm(const Context& c) {
    c.require_inputs(2, 3);
    require_matmul_type(c, 0);
    c.require_same_dtype(0, 1);
    if (c.has(2)) c.require_same_dtype(0, 2);

    const auto& a = c.in(0).shape;
    const auto& b = c.in(1).shape;
    if (a.size() != 2) c.fail("A must be 2-D, got " + shape_of(a));
    if (b.size() != 2) c.fail("B must be 2-D, got " + shape_of(b));

    const bool trans_a = c.node.attr_int("transA", 0) != 0;
    const bool trans_b = c.node.attr_int("transB", 0) != 0;
    int64_t m  = trans_a ? a[1] : a[0];
    int64_t ka = trans_a ? a[0] : a[1];
    int64_t kb = trans_b ? b[1] : b[0];
    int64_t n  = trans_b ? b[0] : b[1];
    if (ka >= 0 && kb >= 0 && ka != kb)
        c.fail("inner dimensions differ: A " + shape_of(a) + (trans_a ? " (transposed)" : "") +
               ", B " + shape_of(b) + (trans_b ? " (transposed)" : "") + " (" +
               std::to_string(ka) + " vs " + std::to_string(kb) + ")");

    // C broadcasts one way only: into [M, N]
    if (c.has(2)) {
        const auto& cs = c.in(2).shape;
        const std::vector<int64_t> target = {m, n};
        bool ok = cs.size() <= 2;
        for (size_t i = 0; ok && i < cs.size(); ++i) {
            int64_t cd = cs[i], td = target[target.size() - cs.size() + i];
            ok = cd == 1 || cd < 0 || td < 0 || cd == td;
        }
        if (!ok)
            c.fail("C with shape " + shape_of(cs) + " cannot broadcast to the result " +
                   shape_of(target));
    }
    return {{c.in(0).dtype, {m, n}}};
}

// RULES - SHAPE MANIPULATION

// normalise an axis attribute in [-rank, rank + extra) to [0, rank + extra)
int64_t normalize_axis(const Context& c, int64_t axis, int64_t rank, int64_t extra) {
    int64_t lo = -rank, hi = rank + extra - 1;
    if (axis < lo || axis > hi)
        c.fail("axis " + std::to_string(axis) + " is out of range [" + std::to_string(lo) +
               ", " + std::to_string(hi) + "] for rank " + std::to_string(rank));
    return axis < 0 ? axis + rank : axis;
}

// Reshape(data, shape), the target shape must be known at compile time
//   0 copies the input dim at the same position (unless allowzero=1, then
//   it means an empty dim), -1 is inferred from the element count. an entry
//   computed from a dynamic dim (Shape of the input, with the batch not yet
//   fixed) gives a dynamic output dim
std::vector<TypeInfo> reshape(const Context& c) {
    c.require_inputs(2, 2);
    const std::vector<int64_t>& in = c.in(0).shape;
    std::optional<Value> value = c.value_of(1);
    if (!value) c.const_ints(1);       // fails, saying why
    if (c.in(1).shape.size() != 1) c.fail("shape input must be 1-D, got " + c.in(1).shape_str());
    const bool allowzero = c.node.attr_int("allowzero", 0) != 0;
    std::vector<int64_t> target;
    for (const std::optional<int64_t>& e : *value) target.push_back(e ? *e : 1);   // 1: a placeholder

    std::vector<int64_t> out(target.size());
    std::vector<bool> copied(in.size(), false);   // input dims carried over by a 0
    int infer_at = -1;
    for (size_t i = 0; i < target.size(); ++i) {
        int64_t t = target[i];
        if (!(*value)[i]) {
            out[i] = -1;                    // known once the dynamic dims are fixed
        } else if (t == -1) {
            if (infer_at >= 0) c.fail("shape " + shape_of(target) + " has more than one -1");
            infer_at = static_cast<int>(i);
            out[i] = -1;
        } else if (t == 0 && !allowzero) {
            if (i >= in.size())
                c.fail("shape " + shape_of(target) + " copies dim " + std::to_string(i) +
                       " with 0, but the input " + shape_of(in) + " has rank " +
                       std::to_string(in.size()));
            out[i] = in[i];
            copied[i] = true;
        } else if (t < 0) {
            c.fail("shape " + shape_of(target) + " has invalid dim " + std::to_string(t));
        } else {
            out[i] = t;
        }
    }
    if (allowzero && infer_at >= 0 &&
        std::find(target.begin(), target.end(), 0) != target.end())
        c.fail("shape " + shape_of(target) + " has both 0 and -1 with allowzero=1");

    // element counts, ignoring dims carried over unchanged by 0
    std::vector<int64_t> in_rest, out_rest;
    for (size_t i = 0; i < in.size(); ++i)
        if (!copied[i]) in_rest.push_back(in[i]);
    for (size_t i = 0; i < out.size(); ++i)
        if (!(i < copied.size() && copied[i]) && static_cast<int>(i) != infer_at) out_rest.push_back(out[i]);
    int64_t in_count = product(in_rest, 0, in_rest.size());
    int64_t out_count = product(out_rest, 0, out_rest.size());

    if (infer_at >= 0) {
        if (in_count >= 0 && out_count > 0) {
            if (in_count % out_count != 0)
                c.fail("cannot reshape " + shape_of(in) + " to " + shape_of(target) + ": " +
                       std::to_string(in_count) + " elements are not divisible by " +
                       std::to_string(out_count));
            out[static_cast<size_t>(infer_at)] = in_count / out_count;
        } else if (out_count == 0 && in_count > 0) {
            c.fail("cannot reshape " + shape_of(in) + " to " + shape_of(target) +
                   ": -1 next to a zero sized dim");
        }
        // otherwise the inferred dim stays dynamic
    } else if (in_count >= 0 && out_count >= 0 && in_count != out_count) {
        c.fail("cannot reshape " + shape_of(in) + " to " + shape_of(target) + ": " +
               std::to_string(in_count) + " vs " + std::to_string(out_count) + " elements");
    }
    return {{c.in(0).dtype, out}};
}

// Flatten(x), axis in [-rank, rank], default 1
//   [d0 .. d(axis-1)] and [d(axis) .. d(rank-1)] each collapse into one dim
std::vector<TypeInfo> flatten(const Context& c) {
    c.require_inputs(1, 1);
    const std::vector<int64_t>& in = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(in.size());
    const size_t axis = static_cast<size_t>(normalize_axis(c, c.node.attr_int("axis", 1), rank, 1));
    return {{c.in(0).dtype, {product(in, 0, axis), product(in, axis, in.size())}}};
}

// Softmax(x), opset 13+: normalises along one axis (default -1), same shape
std::vector<TypeInfo> softmax(const Context& c) {
    c.require_inputs(1, 1);
    c.require_float(0);
    const int64_t rank = static_cast<int64_t>(c.in(0).shape.size());
    if (rank == 0) c.fail("input must have rank >= 1, got a scalar");
    normalize_axis(c, c.node.attr_int("axis", -1), rank, 0);
    return {{c.in(0).dtype, c.in(0).shape}};
}

// RULES - SHAPES AS VALUES
//
// Shape turns a tensor's dims into an int64 tensor, and exporters build
// Reshape targets from it with Gather, Unsqueeze, Squeeze and Concat. Each of
// these also computes its output's value when its inputs' values are known,
// so the whole chain is a constant by the time it reaches the Reshape.

// Shape(data) -> int64 [end - start]: the dims from start (default 0) to end
// (default rank), each clamped to [0, rank] after adding rank to a negative
std::vector<TypeInfo> shape_rule(const Context& c) {
    c.require_inputs(1, 1);
    const std::vector<int64_t>& x = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    auto clamp = [&](int64_t v) { return std::clamp<int64_t>(v < 0 ? v + rank : v, 0, rank); };
    const int64_t start = clamp(c.node.attr_int("start", 0)), end = clamp(c.node.attr_int("end", rank));
    Value v;
    for (int64_t d = start; d < end; ++d) {
        int64_t dim = x[static_cast<size_t>(d)];
        v.push_back(dim >= 0 ? std::optional<int64_t>(dim) : std::nullopt);
    }
    return {{DType::INT64, {static_cast<int64_t>(v.size())}, v}};
}

// Gather(data, indices) along axis (default 0): data's dims before the axis,
// then the indices' dims, then data's dims after the axis
std::vector<TypeInfo> gather(const Context& c) {
    c.require_inputs(2, 2);
    if (c.in(1).dtype != DType::INT64 && c.in(1).dtype != DType::INT32)
        c.fail("indices must be int32 or int64, got " + dtype_name(c.in(1).dtype));
    const std::vector<int64_t>& x = c.in(0).shape;
    const std::vector<int64_t>& idx = c.in(1).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    if (rank == 0) c.fail("data must have rank >= 1, got a scalar");
    const size_t axis = static_cast<size_t>(normalize_axis(c, c.node.attr_int("axis", 0), rank, 0));
    std::vector<int64_t> out(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(axis));
    out.insert(out.end(), idx.begin(), idx.end());
    out.insert(out.end(), x.begin() + static_cast<std::ptrdiff_t>(axis) + 1, x.end());
    TypeInfo result{c.in(0).dtype, out, std::nullopt};

    // constant indices are range-checked now, and with data's value, folded
    std::optional<Value> indices = c.value_of(1);
    const int64_t n = x[axis];
    if (!indices || n < 0) return {result};
    std::vector<int64_t> pos;
    for (const std::optional<int64_t>& e : *indices) {
        if (!e) return {result};
        if (*e < -n || *e >= n)
            c.fail("index " + std::to_string(*e) + " is out of range for dim " + std::to_string(axis) +
                   " of size " + std::to_string(n));
        pos.push_back(*e < 0 ? *e + n : *e);
    }
    std::optional<Value> data = c.value_of(0);
    const int64_t outer = product(x, 0, axis), inner = product(x, axis + 1, x.size());
    if (!data || outer < 0 || inner < 0) return {result};
    Value v;
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t p : pos)
            for (int64_t i = 0; i < inner; ++i) v.push_back((*data)[static_cast<size_t>((o * n + p) * inner + i)]);
    result.value = v;
    return {result};
}

// the axes input of Unsqueeze and Squeeze (opset 13+), normalised for a rank
std::vector<bool> axes_input(const Context& c, int64_t rank) {
    if (c.in(1).shape.size() != 1) c.fail("axes must be 1-D, got " + c.in(1).shape_str());
    std::vector<int64_t> axes = c.const_ints(1);
    std::vector<bool> marked(static_cast<size_t>(rank), false);
    for (int64_t a : axes) {
        size_t d = static_cast<size_t>(normalize_axis(c, a, rank, 0));
        if (marked[d]) c.fail("axes " + shape_of(axes) + " name dim " + std::to_string(d) + " twice");
        marked[d] = true;
    }
    return marked;
}

// Unsqueeze(data, axes): a 1 inserted at each axis, the axes counting dims of
// the output, whose rank is the input's plus the number of axes
std::vector<TypeInfo> unsqueeze(const Context& c) {
    c.require_inputs(2, 2);
    const std::vector<int64_t>& x = c.in(0).shape;
    if (c.in(1).num_elements() < 0) c.fail("axes must have a static shape, got " + c.in(1).shape_str());
    const int64_t rank = static_cast<int64_t>(x.size()) + c.in(1).num_elements();
    std::vector<bool> inserted = axes_input(c, rank);
    std::vector<int64_t> out;
    size_t next = 0;
    for (bool one : inserted) out.push_back(one ? 1 : x[next++]);
    return {{c.in(0).dtype, out, c.value_of(0)}};
}

// Squeeze(data, axes?): the given dims removed, each of which must be 1, or
// without axes every dim of size 1
std::vector<TypeInfo> squeeze(const Context& c) {
    c.require_inputs(1, 2);
    const std::vector<int64_t>& x = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    std::vector<bool> removed(x.size(), false);
    if (c.has(1)) {
        removed = axes_input(c, rank);
        for (size_t d = 0; d < x.size(); ++d)
            if (removed[d] && x[d] >= 0 && x[d] != 1)     // a dynamic dim must turn out to be 1
                c.fail("dim " + std::to_string(d) + " has size " + std::to_string(x[d]) +
                       ", only dims of size 1 can be squeezed");
    } else {
        for (size_t d = 0; d < x.size(); ++d) {
            if (x[d] < 0)
                c.fail("dim " + std::to_string(d) + " is dynamic, so whether Squeeze without axes removes it "
                       "is unknown; fix the input shapes with --batch or --input-shape");
            removed[d] = x[d] == 1;
        }
    }
    std::vector<int64_t> out;
    for (size_t d = 0; d < x.size(); ++d)
        if (!removed[d]) out.push_back(x[d]);
    return {{c.in(0).dtype, out, c.value_of(0)}};
}

// Concat(inputs...) along axis: the same type and rank, the same dims except
// along the axis, where they add up
std::vector<TypeInfo> concat(const Context& c) {
    if (c.num_inputs() == 0) c.fail("expects at least 1 input");
    if (!c.node.has_attr("axis")) c.fail("axis is required");
    std::vector<int64_t> out = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(out.size());
    if (rank == 0) c.fail("inputs must have rank >= 1, got a scalar");
    const size_t axis = static_cast<size_t>(normalize_axis(c, c.node.attr_int("axis"), rank, 0));
    for (size_t i = 1; i < c.num_inputs(); ++i) {
        c.require_same_dtype(0, i);
        const std::vector<int64_t>& s = c.in(i).shape;
        if (s.size() != out.size())
            c.fail("input " + std::to_string(i) + " has rank " + std::to_string(s.size()) +
                   ", input 0 has rank " + std::to_string(rank));
        for (size_t d = 0; d < out.size(); ++d) {
            if (d == axis) {
                out[d] = out[d] < 0 || s[d] < 0 ? -1 : out[d] + s[d];
            } else if (out[d] < 0) {
                out[d] = s[d];
            } else if (s[d] >= 0 && s[d] != out[d]) {
                c.fail("input " + std::to_string(i) + " has size " + std::to_string(s[d]) + " in dim " +
                       std::to_string(d) + ", the others " + std::to_string(out[d]));
            }
        }
    }

    // the value, if every input has one: for each slice before the axis, each
    // input's block in turn
    TypeInfo result{c.in(0).dtype, out, std::nullopt};
    const int64_t outer = product(out, 0, axis), inner = product(out, axis + 1, out.size());
    if (outer < 0 || inner < 0) return {result};
    std::vector<Value> values;
    for (size_t i = 0; i < c.num_inputs(); ++i) {
        std::optional<Value> v = c.value_of(i);
        if (!v || c.in(i).shape[axis] < 0) return {result};
        values.push_back(*v);
    }
    Value v;
    for (int64_t o = 0; o < outer; ++o)
        for (size_t i = 0; i < values.size(); ++i) {
            const int64_t block = c.in(i).shape[axis] * inner;
            for (int64_t k = 0; k < block; ++k) v.push_back(values[i][static_cast<size_t>(o * block + k)]);
        }
    result.value = v;
    return {result};
}

// RULES - SLICING
//
// Slice and each output of Split take a box of the input: per dim a first
// index, a step and a count. the shape rule, the values computed at compile
// time and the emitter all work from that one description

SliceRegion whole(const std::vector<int64_t>& x) {
    SliceRegion r;
    r.start.assign(x.size(), 0);
    r.step.assign(x.size(), 1);
    r.count = x;
    return r;
}

// an optional 1-D int64 input known at compile time. (the attribute forms of
// Slice before opset 10 and Split before opset 13 never get here: the
// frontend accepts opset 13 and later)
std::optional<std::vector<int64_t>> ints_input(const Context& c, size_t i, const char* what) {
    if (!c.has(i)) return std::nullopt;
    if (c.in(i).shape.size() != 1) c.fail(std::string(what) + " must be 1-D, got " + c.in(i).shape_str());
    return c.const_ints(i);
}

// Slice(data, starts, ends, axes?, steps?): for each named axis, starts and
// ends (negative ones count from the end) are clamped to the dim, then every
// step-th index from start up to, not including, end is taken; a negative step
// walks backwards. axes default to 0, 1, ..., steps to 1
SliceRegion slice_region(const Context& c) {
    c.require_inputs(3, 5);
    const std::vector<int64_t>& x = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    if (rank == 0) c.fail("data must have rank >= 1, got a scalar");
    auto required = [&](size_t i, const char* name) {
        std::optional<std::vector<int64_t>> v = ints_input(c, i, name);
        if (!v) c.fail(std::string(name) + " is required");
        return *v;
    };
    std::vector<int64_t> starts = required(1, "starts"), ends = required(2, "ends");
    std::vector<int64_t> axes, steps(starts.size(), 1);
    if (auto a = ints_input(c, 3, "axes")) axes = *a;
    else for (size_t i = 0; i < starts.size(); ++i) axes.push_back(static_cast<int64_t>(i));
    if (auto s = ints_input(c, 4, "steps")) steps = *s;
    if (ends.size() != starts.size() || axes.size() != starts.size() || steps.size() != starts.size())
        c.fail("starts " + shape_of(starts) + ", ends " + shape_of(ends) + ", axes " + shape_of(axes) +
               " and steps " + shape_of(steps) + " must have the same length");

    SliceRegion r = whole(x);
    std::vector<bool> seen(x.size(), false);
    for (size_t i = 0; i < axes.size(); ++i) {
        const size_t d = static_cast<size_t>(normalize_axis(c, axes[i], rank, 0));
        if (seen[d]) c.fail("axes " + shape_of(axes) + " name dim " + std::to_string(d) + " twice");
        seen[d] = true;
        const int64_t step = steps[i], dim = x[d];
        if (step == 0) c.fail("steps " + shape_of(steps) + " has a 0");
        r.step[d] = step;
        if (dim < 0) {                      // known once the dynamic dims are fixed
            r.start[d] = starts[i];
            r.count[d] = -1;
            continue;
        }
        if (dim == 0) {
            r.start[d] = r.count[d] = 0;
            continue;
        }
        // negative positions count from the end; adding dim to one cannot overflow
        int64_t start = starts[i] < 0 ? starts[i] + dim : starts[i];
        int64_t end = ends[i] < 0 ? ends[i] + dim : ends[i];
        if (step > 0) {
            start = std::clamp<int64_t>(start, 0, dim);
            end = std::clamp<int64_t>(end, 0, dim);
            r.count[d] = end > start ? (end - start - 1) / step + 1 : 0;
        } else {
            start = std::clamp<int64_t>(start, 0, dim - 1);
            end = std::clamp<int64_t>(end, -1, dim - 1);
            const int64_t back = step == INT64_MIN ? INT64_MAX : -step;
            r.count[d] = start > end ? (start - end - 1) / back + 1 : 0;
        }
        r.start[d] = r.count[d] ? start : 0;
    }
    return r;
}

// Split(input, split?) along axis (default 0): consecutive pieces, one per
// output, of the sizes in split. without it, equal pieces; with opset 18's
// num_outputs the last piece may be smaller
std::vector<SliceRegion> split_regions(const Context& c) {
    c.require_inputs(1, 2);
    const std::vector<int64_t>& x = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    if (rank == 0) c.fail("input must have rank >= 1, got a scalar");
    const size_t axis = static_cast<size_t>(normalize_axis(c, c.node.attr_int("axis", 0), rank, 0));
    const int64_t dim = x[axis], outputs = static_cast<int64_t>(c.node.outputs.size());
    if (outputs == 0) c.fail("has no outputs");
    std::vector<int64_t> sizes;
    if (auto s = ints_input(c, 1, "split")) {
        if (c.node.has_attr("num_outputs")) c.fail("split and num_outputs are both given");
        sizes = *s;
        if (static_cast<int64_t>(sizes.size()) != outputs)
            c.fail("split " + shape_of(sizes) + " has " + std::to_string(sizes.size()) + " sizes for " +
                   std::to_string(outputs) + " outputs");
        int64_t sum = 0;
        for (int64_t v : sizes) {
            if (v < 0) c.fail("split " + shape_of(sizes) + " has a negative size");
            if (__builtin_add_overflow(sum, v, &sum)) c.fail("split " + shape_of(sizes) + " overflows");
        }
        if (dim >= 0 && sum != dim)
            c.fail("split " + shape_of(sizes) + " adds up to " + std::to_string(sum) + ", but dim " +
                   std::to_string(axis) + " has size " + std::to_string(dim));
    } else if (dim < 0) {
        sizes.assign(static_cast<size_t>(outputs), -1);
    } else if (c.node.has_attr("num_outputs")) {
        if (c.node.attr_int("num_outputs") != outputs)
            c.fail("num_outputs is " + std::to_string(c.node.attr_int("num_outputs")) + ", but the node has " +
                   std::to_string(outputs) + " outputs");
        const int64_t chunk = (dim + outputs - 1) / outputs, last = dim - chunk * (outputs - 1);
        if (last < 0)
            c.fail("dim " + std::to_string(axis) + " of size " + std::to_string(dim) + " cannot be split into " +
                   std::to_string(outputs) + " pieces of " + std::to_string(chunk));
        sizes.assign(static_cast<size_t>(outputs), chunk);
        sizes.back() = last;
    } else {
        if (dim % outputs != 0)
            c.fail("dim " + std::to_string(axis) + " of size " + std::to_string(dim) + " cannot be split into " +
                   std::to_string(outputs) + " equal pieces");
        sizes.assign(static_cast<size_t>(outputs), dim / outputs);
    }

    std::vector<SliceRegion> regions;
    int64_t at = 0;
    for (int64_t size : sizes) {
        SliceRegion r = whole(x);
        r.start[axis] = size < 0 ? 0 : at;
        r.count[axis] = size;
        at += size < 0 ? 0 : size;
        regions.push_back(r);
    }
    return regions;
}

// the value of a region of input 0, if input 0 has one and every size is known
std::optional<Value> region_value(const Context& c, const SliceRegion& r) {
    const std::vector<int64_t>& x = c.in(0).shape;
    std::optional<Value> data = c.value_of(0);
    const int64_t total = product(r.count, 0, r.count.size());
    if (!data || total < 0 || product(x, 0, x.size()) < 0) return std::nullopt;
    std::vector<int64_t> stride(x.size(), 1), idx(x.size(), 0);
    for (size_t d = x.size(); d-- > 1;) stride[d - 1] = stride[d] * x[d];
    Value v;
    for (int64_t k = 0; k < total; ++k) {
        int64_t at = 0;
        for (size_t d = 0; d < x.size(); ++d) at += (r.start[d] + idx[d] * r.step[d]) * stride[d];
        v.push_back((*data)[static_cast<size_t>(at)]);
        for (size_t d = x.size(); d-- > 0;) {
            if (++idx[d] < r.count[d]) break;
            idx[d] = 0;
        }
    }
    return v;
}

std::vector<TypeInfo> slice(const Context& c) {
    SliceRegion r = slice_region(c);
    return {{c.in(0).dtype, r.count, region_value(c, r)}};
}

std::vector<TypeInfo> split(const Context& c) {
    std::vector<TypeInfo> out;
    for (const SliceRegion& r : split_regions(c)) out.push_back({c.in(0).dtype, r.count, region_value(c, r)});
    return out;
}

// RULES - CONVOLUTION

// Conv(x, w, b?), 2-D only: x [N, C, H, W], w [M, C / group, kH, kW], b [M]
std::vector<TypeInfo> conv(const Context& c) {
    c.require_inputs(2, 3);
    c.require_float(0);
    c.require_same_dtype(0, 1);
    const auto& x = c.in(0).shape;
    const auto& w = c.in(1).shape;
    if (x.size() != 4) c.fail("only 2-D convolution is supported, input must be [N, C, H, W], got " + shape_of(x));
    for (size_t i = 1; i < 4; ++i)
        if (x[i] < 0) c.fail("input " + shape_of(x) + " must have static C, H and W");
    ConvGeometry g = conv_geometry(c.node, x, w);
    if (c.has(2)) {
        c.require_same_dtype(0, 2);
        if (c.in(2).shape != std::vector<int64_t>{g.m})
            c.fail("bias must be [" + std::to_string(g.m) + "], got " + c.in(2).shape_str());
    }
    return {{c.in(0).dtype, {x[0], g.m, g.out_h, g.out_w}}};
}

// RULES - POOLING

// MaxPool(x) -> y, 2-D only: x [N, C, H, W], y [N, C, outH, outW]
std::vector<TypeInfo> max_pool(const Context& c) {
    c.require_inputs(1, 1);
    c.require_float(0);
    if (c.node.outputs.size() > 1 && !c.node.outputs[1].empty())
        c.fail("the Indices output is not supported");
    const auto& x = c.in(0).shape;
    if (x.size() != 4) c.fail("only 2-D pooling is supported, input must be [N, C, H, W], got " + shape_of(x));
    for (size_t i = 1; i < 4; ++i)
        if (x[i] < 0) c.fail("input " + shape_of(x) + " must have static C, H and W");
    PoolGeometry g = pool_geometry(c.node, x);
    return {{c.in(0).dtype, {x[0], x[1], g.out_h, g.out_w}}};
}

// RULES - REDUCTION

// the dims a reduction averages away, see reduced_axes in the header
//   ReduceMean takes axes as an attribute up to opset 17 and as an optional
//   constant input from opset 18. no axes (or empty ones) means every dim,
//   unless noop_with_empty_axes=1, which means none
std::vector<bool> axes_to_reduce(const Context& c) {
    const int64_t rank = static_cast<int64_t>(c.in(0).shape.size());
    std::vector<bool> reduce(static_cast<size_t>(rank), false);
    if (c.node.op == OpType::GlobalAveragePool) {
        for (size_t i = 2; i < reduce.size(); ++i) reduce[i] = true;
        return reduce;
    }
    std::vector<int64_t> axes;
    if (c.node.has_attr("axes")) {
        if (c.has(1))
            c.fail("axes given both as an attribute (opset 17 and earlier) and as an input (opset 18 and later)");
        axes = c.node.attr_ints("axes");
    } else if (c.has(1)) {
        if (c.in(1).shape.size() != 1) c.fail("axes must be 1-D, got " + c.in(1).shape_str());
        axes = c.const_ints(1);
    }
    if (axes.empty()) {
        if (c.node.attr_int("noop_with_empty_axes", 0) == 0) reduce.assign(reduce.size(), true);
        return reduce;
    }
    for (int64_t a : axes) {
        size_t d = static_cast<size_t>(normalize_axis(c, a, rank, 0));
        if (reduce[d]) c.fail("axes " + shape_of(axes) + " name dim " + std::to_string(d) + " twice");
        reduce[d] = true;
    }
    return reduce;
}

// ReduceMean(data, axes?) -> reduced, keepdims (default 1) leaves each
// reduced dim in place as a 1, otherwise it is removed
std::vector<TypeInfo> reduce_mean(const Context& c) {
    c.require_inputs(1, 2);
    c.require_numeric(0);
    const std::vector<int64_t>& x = c.in(0).shape;
    std::vector<bool> reduce = axes_to_reduce(c);
    const bool keepdims = c.node.attr_int("keepdims", 1) != 0;
    std::vector<int64_t> out;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!reduce[i]) out.push_back(x[i]);
        else if (keepdims) out.push_back(1);
    }
    return {{c.in(0).dtype, out}};
}

// GlobalAveragePool(x) -> y: x [N, C, D1, ...], y [N, C, 1, ...], the mean of
// each channel
std::vector<TypeInfo> global_average_pool(const Context& c) {
    c.require_inputs(1, 1);
    c.require_float(0);
    std::vector<int64_t> out = c.in(0).shape;
    if (out.size() < 3) c.fail("input must be [N, C, D1, ...], got " + c.in(0).shape_str());
    for (size_t i = 2; i < out.size(); ++i) out[i] = 1;
    return {{c.in(0).dtype, out}};
}

// RULES - NORMALIZATION AND LAYOUT

// LayerNormalization(X, Scale, B?) -> Y, opset 17: X normalised over its dims
// from axis (default -1) on, then scaled and shifted; Scale and B must
// broadcast to those normalized dims. Y has X's shape
std::vector<TypeInfo> layer_norm(const Context& c) {
    c.require_inputs(2, 3);
    c.require_float(0);
    for (size_t i = 1; i < c.node.outputs.size(); ++i)
        if (!c.node.outputs[i].empty()) c.fail("the Mean and InvStdDev outputs are not supported");
    const std::vector<int64_t>& x = c.in(0).shape;
    const int64_t rank = static_cast<int64_t>(x.size());
    if (rank == 0) c.fail("input must have rank >= 1, got a scalar");
    const size_t axis = static_cast<size_t>(normalize_axis(c, c.node.attr_int("axis", -1), rank, 0));
    const std::vector<int64_t> normalized(x.begin() + static_cast<std::ptrdiff_t>(axis), x.end());
    for (size_t i = 1; i < 3; ++i) {
        if (!c.has(i)) continue;
        c.require_same_dtype(0, i);
        const std::vector<int64_t>& s = c.in(i).shape;
        bool fits = s.size() <= normalized.size();
        for (size_t k = 1; fits && k <= s.size(); ++k) {
            int64_t d = s[s.size() - k], n = normalized[normalized.size() - k];
            fits = d == 1 || d < 0 || n < 0 || d == n;
        }
        if (!fits)
            c.fail(std::string(i == 1 ? "scale " : "bias ") + c.in(i).shape_str() +
                   " does not broadcast to the normalized dims " + shape_of(normalized));
    }
    return {{c.in(0).dtype, x}};
}

// BatchNormalization(X, scale, B, input_mean, input_var), inference only: Y has
// X's shape, and each of the four parameters is one value per channel, dim 1
// of X. the training outputs (running_mean and running_var) and
// training_mode = 1 are refused
std::vector<TypeInfo> batch_norm(const Context& c) {
    c.require_inputs(5, 5);
    c.require_float(0);
    for (size_t i = 1; i < c.node.outputs.size(); ++i)
        if (!c.node.outputs[i].empty()) c.fail("the training outputs (running mean and variance) are not supported");
    if (c.node.attr_int("training_mode", 0) != 0) c.fail("training_mode = 1 is not supported, only inference");
    const std::vector<int64_t>& x = c.in(0).shape;
    if (x.size() < 2) c.fail("input must have rank >= 2 (N, C, ...), got " + c.in(0).shape_str());
    static const char* names[] = {"", "scale", "B", "input_mean", "input_var"};
    for (size_t i = 1; i < 5; ++i) {
        if (!c.has(i)) c.fail(std::string(names[i]) + " is required");
        c.require_same_dtype(0, i);
        const std::vector<int64_t>& s = c.in(i).shape;
        if (s.size() != 1 || (s[0] >= 0 && x[1] >= 0 && s[0] != x[1]))
            c.fail(std::string(names[i]) + " " + c.in(i).shape_str() + " must have one value per channel, [" +
                   std::to_string(x[1]) + "]");
    }
    return {{c.in(0).dtype, x}};
}

// Transpose(data), perm (default: the dims reversed): output dim i is input
// dim perm[i]
std::vector<TypeInfo> transpose(const Context& c) {
    c.require_inputs(1, 1);
    const std::vector<int64_t>& x = c.in(0).shape;
    std::vector<int64_t> perm = transpose_perm(c.node, x.size());
    std::vector<bool> seen(x.size(), false);
    bool valid = perm.size() == x.size();
    for (size_t i = 0; valid && i < perm.size(); ++i) {
        valid = perm[i] >= 0 && perm[i] < static_cast<int64_t>(x.size()) && !seen[static_cast<size_t>(perm[i])];
        if (valid) seen[static_cast<size_t>(perm[i])] = true;
    }
    if (!valid)
        c.fail("perm " + shape_of(perm) + " is not a permutation of the " + std::to_string(x.size()) +
               " dims of " + c.in(0).shape_str());
    std::vector<int64_t> out;
    for (int64_t p : perm) out.push_back(x[static_cast<size_t>(p)]);
    return {{c.in(0).dtype, out}};
}

const std::unordered_map<OpType, Rule>& rules() {
    static const std::unordered_map<OpType, Rule> table = {
        {OpType::Relu,     unary_numeric},
        {OpType::Abs,      unary_numeric},
        {OpType::Neg,      unary_numeric},
        {OpType::Sigmoid,  unary_float},
        {OpType::Tanh,     unary_float},
        {OpType::Exp,      unary_float},
        {OpType::Log,      unary_float},
        {OpType::Sqrt,     unary_float},
        {OpType::Identity, identity},
        {OpType::Clip,     clip},
        {OpType::Dropout,  dropout},
        {OpType::Cast,     cast},
        {OpType::Add,      binary_arith},
        {OpType::Sub,      binary_arith},
        {OpType::Mul,      binary_arith},
        {OpType::Div,      binary_arith},
        {OpType::Pow,      pow_rule},
        {OpType::Constant, constant},
        {OpType::MatMul,   matmul},
        {OpType::Gemm,     gemm},
        {OpType::Reshape,  reshape},
        {OpType::Flatten,  flatten},
        {OpType::Softmax,  softmax},
        {OpType::Shape,     shape_rule},
        {OpType::Gather,    gather},
        {OpType::Unsqueeze, unsqueeze},
        {OpType::Squeeze,   squeeze},
        {OpType::Concat,    concat},
        {OpType::Slice,     slice},
        {OpType::Split,     split},
        {OpType::LayerNormalization, layer_norm},
        {OpType::BatchNormalization, batch_norm},
        {OpType::Transpose,          transpose},
        {OpType::Conv,     conv},
        {OpType::MaxPool,  max_pool},
        {OpType::GlobalAveragePool, global_average_pool},
        {OpType::ReduceMean,        reduce_mean},
    };
    return table;
}

// combine an inferred type with what the model declared for the same value
// fails on any contradiction, fills inferred -1 dims from known declared ones
TypeInfo merge(const Context& c, const std::string& name, TypeInfo inferred, const Tensor& declared) {
    auto conflict = [&](const std::string& what) {
        c.fail("output '" + name + "' is declared as " + dtype_name(declared.dtype) +
               declared.shape_str() + " but inferred as " + dtype_name(inferred.dtype) +
               Tensor("", inferred.shape, inferred.dtype).shape_str() + " (" + what + ")");
    };
    if (declared.dtype != DType::UNDEFINED && declared.dtype != inferred.dtype) conflict("type");
    if (declared.shape.size() != inferred.shape.size()) conflict("rank");
    for (size_t i = 0; i < inferred.shape.size(); ++i) {
        int64_t d = declared.shape[i], f = inferred.shape[i];
        if (d >= 0 && f >= 0 && d != f) conflict("dim " + std::to_string(i));
        if (f < 0) inferred.shape[i] = d;
    }
    return inferred;
}

} // namespace

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank);
    for (size_t i = 0; i < rank; ++i) {
        // align from the right, missing leading dims act as 1
        int64_t x = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        int64_t y = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (x == 1)                      out[i] = y;
        else if (y == 1)                 out[i] = x;
        else if (x == y)                 out[i] = x;
        else if (x < 0)                  out[i] = y;   // dynamic must match the known size
        else if (y < 0)                  out[i] = x;
        else {
            throw ShapeError("cannot broadcast " + Tensor("", a, DType::UNDEFINED).shape_str() +
                             " with " + Tensor("", b, DType::UNDEFINED).shape_str() +
                             " (dim " + std::to_string(i) + ": " + std::to_string(x) +
                             " vs " + std::to_string(y) + ")");
        }
    }
    return out;
}

namespace {

// strides, dilations, padding and output size of a 2-D sliding window of
// kh x kw over an h x w image, shared by Conv and the pooling operators
struct Window {
    int64_t stride_h, stride_w, dil_h, dil_w;
    int64_t pad_top, pad_left, pad_bottom, pad_right;
    int64_t out_h, out_w;
};

Window sliding_window(const Node& node, int64_t h, int64_t w, int64_t kh, int64_t kw, bool pooling) {
    auto fail = [&](const std::string& msg) -> ShapeError { return ShapeError(where(node) + ": " + msg); };
    Window g{};

    // attributes, with onnx's defaults
    auto pair = [&](const char* key) {
        std::vector<int64_t> v = node.attr_ints(key, {1, 1});
        if (v.size() != 2) throw fail(std::string(key) + " must have 2 values for a 2-D window");
        for (int64_t d : v)
            if (d < 1) throw fail(std::string(key) + " must be at least 1");
        return v;
    };
    auto strides = pair("strides"), dilations = pair("dilations");
    g.stride_h = strides[0]; g.stride_w = strides[1];
    g.dil_h = dilations[0]; g.dil_w = dilations[1];
    const bool ceil_mode = pooling && node.attr_int("ceil_mode", 0) != 0;

    // padding: explicit pads [top, left, bottom, right], or computed by auto_pad
    const int64_t eff_h = (kh - 1) * g.dil_h + 1, eff_w = (kw - 1) * g.dil_w + 1;
    const std::string auto_pad = node.attr_str("auto_pad", "NOTSET");
    if (auto_pad == "NOTSET") {
        std::vector<int64_t> pads = node.attr_ints("pads", {0, 0, 0, 0});
        if (pads.size() != 4) throw fail("pads must have 4 values [top, left, bottom, right]");
        for (int64_t p : pads)
            if (p < 0) throw fail("pads must not be negative");
        // a pooling window must overlap the image (onnx runtime and pytorch check this too)
        if (pooling && (pads[0] >= kh || pads[2] >= kh || pads[1] >= kw || pads[3] >= kw))
            throw fail("pads must be smaller than the kernel");
        g.pad_top = pads[0]; g.pad_left = pads[1]; g.pad_bottom = pads[2]; g.pad_right = pads[3];
    } else if (auto_pad == "VALID") {
        g.pad_top = g.pad_left = g.pad_bottom = g.pad_right = 0;
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        // output = ceil(input / stride), with the padding split evenly and
        // any odd pixel at the end (UPPER) or the start (LOWER)
        auto same = [&](int64_t in, int64_t stride, int64_t eff, int64_t& begin, int64_t& end) {
            const int64_t out = (in + stride - 1) / stride;
            const int64_t total = std::max<int64_t>(0, (out - 1) * stride + eff - in);
            begin = auto_pad == "SAME_UPPER" ? total / 2 : total - total / 2;
            end = total - begin;
        };
        same(h, g.stride_h, eff_h, g.pad_top, g.pad_bottom);
        same(w, g.stride_w, eff_w, g.pad_left, g.pad_right);
    } else {
        throw fail("unknown auto_pad '" + auto_pad + "'");
    }

    const int64_t span_h = h + g.pad_top + g.pad_bottom, span_w = w + g.pad_left + g.pad_right;
    if (span_h < eff_h || span_w < eff_w)
        throw fail("kernel " + std::to_string(eff_h) + "x" + std::to_string(eff_w) +
                   " (with dilation) is larger than the padded input " + std::to_string(span_h) + "x" +
                   std::to_string(span_w));
    // ceil_mode rounds up, so a last, partial window counts; but not one that
    // would start in the end padding and see no pixels at all
    auto count = [&](int64_t in, int64_t span, int64_t eff, int64_t stride, int64_t pad_begin) {
        if (!ceil_mode || auto_pad != "NOTSET") return (span - eff) / stride + 1;
        int64_t out = (span - eff + stride - 1) / stride + 1;
        if ((out - 1) * stride >= in + pad_begin) --out;
        return out;
    };
    g.out_h = count(h, span_h, eff_h, g.stride_h, g.pad_top);
    g.out_w = count(w, span_w, eff_w, g.stride_w, g.pad_left);
    return g;
}

} // namespace

ConvGeometry conv_geometry(const Node& node, const std::vector<int64_t>& x,
                           const std::vector<int64_t>& weight) {
    auto fail = [&](const std::string& msg) -> ShapeError { return ShapeError(where(node) + ": " + msg); };
    auto shape = [](const std::vector<int64_t>& s) { return Tensor("", s, DType::UNDEFINED).shape_str(); };
    if (x.size() != 4) throw fail("input must be [N, C, H, W], got " + shape(x));
    if (weight.size() != 4) throw fail("weight must be [M, C / group, kH, kW], got " + shape(weight));
    for (int64_t d : weight)
        if (d < 1) throw fail("weight must have a static, non-empty shape, got " + shape(weight));

    ConvGeometry g{};
    g.n = x[0]; g.c = x[1]; g.h = x[2]; g.w = x[3];
    g.m = weight[0]; g.kh = weight[2]; g.kw = weight[3];
    g.group = node.attr_int("group", 1);
    if (node.has_attr("kernel_shape") && node.attr_ints("kernel_shape") != std::vector<int64_t>{g.kh, g.kw})
        throw fail("kernel_shape does not match the weight " + shape(weight));

    // channels: each of the group's filters sees C / group input channels
    if (g.group < 1) throw fail("group must be at least 1");
    if (g.c % g.group != 0 || g.c / g.group != weight[1])
        throw fail("input has " + std::to_string(g.c) + " channels, but the weight " + shape(weight) +
                   " with group " + std::to_string(g.group) + " expects " + std::to_string(weight[1] * g.group));
    if (g.m % g.group != 0)
        throw fail(std::to_string(g.m) + " output channels are not divisible by group " + std::to_string(g.group));

    Window win = sliding_window(node, g.h, g.w, g.kh, g.kw, false);
    g.stride_h = win.stride_h; g.stride_w = win.stride_w; g.dil_h = win.dil_h; g.dil_w = win.dil_w;
    g.pad_top = win.pad_top; g.pad_left = win.pad_left; g.pad_bottom = win.pad_bottom; g.pad_right = win.pad_right;
    g.out_h = win.out_h; g.out_w = win.out_w;
    return g;
}

PoolGeometry pool_geometry(const Node& node, const std::vector<int64_t>& x) {
    auto fail = [&](const std::string& msg) -> ShapeError { return ShapeError(where(node) + ": " + msg); };
    if (x.size() != 4)
        throw fail("only 2-D pooling is supported, input must be [N, C, H, W], got " +
                   Tensor("", x, DType::UNDEFINED).shape_str());
    if (!node.has_attr("kernel_shape")) throw fail("kernel_shape is required");
    std::vector<int64_t> k = node.attr_ints("kernel_shape");
    if (k.size() != 2) throw fail("kernel_shape must have 2 values for 2-D pooling");
    if (k[0] < 1 || k[1] < 1) throw fail("kernel_shape must be at least 1");
    if (node.attr_int("storage_order", 0) != 0) throw fail("storage_order 1 (column-major indices) is not supported");

    PoolGeometry g{};
    g.n = x[0]; g.c = x[1]; g.h = x[2]; g.w = x[3];
    g.kh = k[0]; g.kw = k[1];
    Window win = sliding_window(node, g.h, g.w, g.kh, g.kw, true);
    g.stride_h = win.stride_h; g.stride_w = win.stride_w; g.dil_h = win.dil_h; g.dil_w = win.dil_w;
    g.pad_top = win.pad_top; g.pad_left = win.pad_left; g.pad_bottom = win.pad_bottom; g.pad_right = win.pad_right;
    g.out_h = win.out_h; g.out_w = win.out_w;
    return g;
}

std::vector<int64_t> transpose_perm(const Node& node, size_t rank) {
    if (node.has_attr("perm")) return node.attr_ints("perm");
    std::vector<int64_t> perm;
    for (size_t i = rank; i-- > 0;) perm.push_back(static_cast<int64_t>(i));
    return perm;
}

std::vector<bool> reduced_axes(const Node& node, const Graph& g) {
    return axes_to_reduce(Context(node, g));
}

std::vector<SliceRegion> slice_regions(const Node& node, const Graph& g) {
    Context c(node, g);
    if (node.op == OpType::Split) return split_regions(c);
    return {slice_region(c)};
}

bool has_shape_rule(OpType op) {
    return rules().count(op) > 0;
}

void infer_shapes(Graph& g) {
    // every value computed in this pass; the fully known ones are also kept
    // with their tensor, for later passes and the emitter
    std::unordered_map<std::string, Value> values;
    for (const Node* n : g.sorted_nodes()) {
        auto rule = rules().find(n->op);
        if (rule == rules().end())
            throw ShapeError(where(*n) + ": no shape inference rule for this operator yet");

        Context c(*n, g, &values);
        std::vector<TypeInfo> results = rule->second(c);
        if (n->outputs.size() > results.size())
            c.fail("has " + std::to_string(n->outputs.size()) + " outputs, expected at most " +
                   std::to_string(results.size()));

        for (size_t i = 0; i < n->outputs.size(); ++i) {
            const std::string& name = n->outputs[i];
            if (name.empty()) continue;   // omitted optional output
            TypeInfo t = results[i];
            if (const Tensor* declared = g.get_tensor_info(name)) t = merge(c, name, t, *declared);
            Tensor out(name, t.shape, t.dtype);
            const int64_t count = product(t.shape, 0, t.shape.size());
            if (t.value && t.dtype == DType::INT64 && count >= 0 && count <= kMaxValueElements &&
                static_cast<int64_t>(t.value->size()) == count) {
                values[name] = *t.value;
                if (std::all_of(t.value->begin(), t.value->end(), [](auto& e) { return e.has_value(); })) {
                    std::vector<int64_t> known;
                    for (const std::optional<int64_t>& e : *t.value) known.push_back(*e);
                    out.data.resize(known.size() * sizeof(int64_t));
                    if (!known.empty()) std::memcpy(out.data.data(), known.data(), out.data.size());
                }
            }
            g.add_value_info(out);
        }
    }
}

std::optional<std::vector<int64_t>> computed_value(const Graph& g, const std::string& name) {
    const Tensor* t = g.get_tensor_info(name);
    if (!t || t->dtype != DType::INT64 || t->data.empty() || g.is_initializer(name)) return std::nullopt;
    return std::vector<int64_t>(t->int64_ptr(), t->int64_ptr() + t->data.size() / sizeof(int64_t));
}

} // namespace null
