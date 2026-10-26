#include "analysis/shape_inference.h"

#include <algorithm>
#include <unordered_map>

namespace null {

namespace {

// result of a shape rule for one output
struct TypeInfo {
    DType dtype = DType::UNDEFINED;
    std::vector<int64_t> shape;
};

std::string where(const Node& n) {
    return "node '" + n.name + "' (" + optype_to_str(n.op) + ")";
}

// everything a rule needs about one node
class Context {
public:
    Context(const Node& n, const Graph& g) : node(n) {
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

private:
    std::vector<const Tensor*> inputs_;
};

using Rule = std::vector<TypeInfo> (*)(const Context&);

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

// Add, Sub, Mul, Div: same types, broadcast shapes
std::vector<TypeInfo> binary_arith(const Context& c) {
    c.require_inputs(2, 2);
    c.require_numeric(0);
    c.require_same_dtype(0, 1);
    try {
        return {{c.in(0).dtype, broadcast_shapes(c.in(0).shape, c.in(1).shape)}};
    } catch (const ShapeError& e) {
        c.fail(e.what());
    }
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

bool has_shape_rule(OpType op) {
    return rules().count(op) > 0;
}

void infer_shapes(Graph& g) {
    for (const Node* n : g.sorted_nodes()) {
        auto rule = rules().find(n->op);
        if (rule == rules().end())
            throw ShapeError(where(*n) + ": no shape inference rule for this operator yet");

        Context c(*n, g);
        std::vector<TypeInfo> results = rule->second(c);
        if (n->outputs.size() > results.size())
            c.fail("has " + std::to_string(n->outputs.size()) + " outputs, expected at most " +
                   std::to_string(results.size()));

        for (size_t i = 0; i < n->outputs.size(); ++i) {
            const std::string& name = n->outputs[i];
            if (name.empty()) continue;   // omitted optional output
            TypeInfo t = results[i];
            if (const Tensor* declared = g.get_tensor_info(name)) t = merge(c, name, t, *declared);
            g.add_value_info(Tensor(name, t.shape, t.dtype));
        }
    }
}

} // namespace null
