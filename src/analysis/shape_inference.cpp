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
