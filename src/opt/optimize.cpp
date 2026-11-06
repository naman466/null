#include "opt/optimize.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_set>

#include "analysis/shape_inference.h"
#include "null_runtime.h"   // the kernels generated code runs, so folded values are the same bits

namespace null {

namespace {

bool is_graph_output(const Graph& g, const std::string& name) {
    return std::find(g.outputs.begin(), g.outputs.end(), name) != g.outputs.end();
}

// read by a live node, or a graph output
bool is_used(const Graph& g, const std::string& name) {
    if (is_graph_output(g, name)) return true;
    for (const Node& n : g.nodes)
        if (!n.is_dead && std::find(n.inputs.begin(), n.inputs.end(), name) != n.inputs.end()) return true;
    return false;
}

// the live node that writes a tensor, if any
Node* producer_of(Graph& g, const std::string& name) {
    for (Node& n : g.nodes)
        if (!n.is_dead && std::find(n.outputs.begin(), n.outputs.end(), name) != n.outputs.end()) return &n;
    return nullptr;
}

// every live node that reads `from` reads `to` instead
void rename_uses(Graph& g, const std::string& from, const std::string& to) {
    for (Node& n : g.nodes)
        if (!n.is_dead)
            for (std::string& in : n.inputs)
                if (in == from) in = to;
}

// in the order they run: constants first, then identities (which can stand
// between a Gemm and its Relu), then fusion, then whatever is left without a reader
const std::vector<Pass>& passes() {
    static const std::vector<Pass> list = {
        {"fold-constants", fold_constants},
        {"remove-identities", remove_identities},
        {"fuse-relu", fuse_relu},
        {"dead-code", eliminate_dead_nodes},
    };
    return list;
}

// CONSTANT FOLDING

Tensor make_constant(const std::string& name, const std::vector<int64_t>& shape, DType dtype,
                     const void* bytes, size_t size) {
    Tensor t(name, shape, dtype);
    t.data.resize(size);
    if (size) std::memcpy(t.data.data(), bytes, size);
    t.is_constant = true;
    return t;
}

std::vector<float> floats_of(const Tensor& t) {
    std::vector<float> v(static_cast<size_t>(t.num_elements()));
    if (!v.empty()) std::memcpy(v.data(), t.data.data(), v.size() * sizeof(float));
    return v;
}

// a Constant node's value as a tensor named after its output, if it has one
// of the common forms
std::optional<Tensor> constant_value(const Node& n, const std::string& name) {
    if (n.has_attr("value")) {
        Tensor t = n.get_attr("value").as_tensor();
        t.name = name;
        t.is_constant = true;
        return t;
    }
    if (n.has_attr("value_ints")) {
        std::vector<int64_t> v = n.attr_ints("value_ints");
        return make_constant(name, {static_cast<int64_t>(v.size())}, DType::INT64, v.data(), v.size() * 8);
    }
    if (n.has_attr("value_int")) {
        int64_t v = n.attr_int("value_int");
        return make_constant(name, {}, DType::INT64, &v, 8);
    }
    if (n.has_attr("value_floats")) {
        std::vector<float> v = n.attr_floats("value_floats");
        return make_constant(name, {static_cast<int64_t>(v.size())}, DType::FLOAT32, v.data(), v.size() * 4);
    }
    if (n.has_attr("value_float")) {
        float v = n.attr_float("value_float");
        return make_constant(name, {}, DType::FLOAT32, &v, 4);
    }
    return std::nullopt;
}

// a float node whose inputs are all constants, computed with the runtime's
// own kernels. only operations whose result is exact or a single rounding per
// element are folded, so the compiler computes the very bits the generated
// code would; anything else is left for run time
std::optional<std::vector<float>> evaluate(const Graph& g, const Node& n, const std::vector<int64_t>& out) {
    const Tensor& x = g.initializers.at(n.inputs[0]);
    std::vector<float> a = floats_of(x);
    std::vector<float> y(static_cast<size_t>(Tensor("", out, DType::FLOAT32).num_elements()));
    switch (n.op) {
        case OpType::Relu:
            null_relu_f32(a.data(), y.data(), static_cast<int64_t>(y.size()));
            return y;
        // exact (abs and neg only touch the sign bit) or correctly rounded (sqrt,
        // by IEEE 754); exp and log are neither, and wait for run time
        case OpType::Abs:
            null_abs_f32(a.data(), y.data(), static_cast<int64_t>(y.size()));
            return y;
        case OpType::Neg:
            null_neg_f32(a.data(), y.data(), static_cast<int64_t>(y.size()));
            return y;
        case OpType::Sqrt:
            null_sqrt_f32(a.data(), y.data(), static_cast<int64_t>(y.size()));
            return y;
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div: {
            const Tensor& w = g.initializers.at(n.inputs[1]);
            std::vector<float> b = floats_of(w);
            const int rank = static_cast<int>(out.size());
            if (rank > NULL_MAX_RANK) return std::nullopt;
            int64_t sa[NULL_MAX_RANK] = {0}, sb[NULL_MAX_RANK] = {0};
            null_broadcast_strides(rank, out.data(), static_cast<int>(x.shape.size()), x.shape.data(), sa);
            null_broadcast_strides(rank, out.data(), static_cast<int>(w.shape.size()), w.shape.data(), sb);
            auto kernel = n.op == OpType::Add ? null_add_bcast_f32 : n.op == OpType::Sub ? null_sub_bcast_f32
                        : n.op == OpType::Mul ? null_mul_bcast_f32 : null_div_bcast_f32;
            kernel(a.data(), b.data(), y.data(), rank, out.data(), sa, sb);
            return y;
        }
        case OpType::Transpose: {
            const int rank = static_cast<int>(x.shape.size());
            if (rank > NULL_MAX_RANK) return std::nullopt;
            std::vector<int64_t> perm = transpose_perm(n, x.shape.size());
            std::vector<int64_t> stride(x.shape.size(), 1), xs;
            for (size_t d = x.shape.size(); d-- > 1;) stride[d - 1] = stride[d] * x.shape[d];
            for (int64_t p : perm) xs.push_back(stride[static_cast<size_t>(p)]);
            null_transpose_f32(a.data(), y.data(), rank, out.data(), xs.data());
            return y;
        }
        case OpType::Reshape:
        case OpType::Flatten:
        case OpType::Squeeze:
        case OpType::Unsqueeze:
        case OpType::Identity:
            return a;      // the same elements in the same order
        default:
            return std::nullopt;
    }
}

} // namespace

size_t fold_constants(Graph& g) {
    size_t changes = 0;
    // in execution order, so a value folded here can feed the next fold
    for (size_t idx : g.topological_sort()) {
        Node& n = g.nodes[idx];
        if (n.is_dead || n.outputs.size() != 1 || n.outputs[0].empty()) continue;
        const std::string out = n.outputs[0];
        // the generated code must compute its outputs, so they are never folded
        if (is_graph_output(g, out)) continue;
        const Tensor* info = g.get_tensor_info(out);
        if (!info || info->num_elements() < 0) continue;

        std::optional<Tensor> folded;
        if (n.op == OpType::Constant) {
            folded = constant_value(n, out);
        } else if (std::optional<std::vector<int64_t>> v = computed_value(g, out)) {
            folded = make_constant(out, info->shape, DType::INT64, v->data(), v->size() * 8);
        } else if (info->dtype == DType::FLOAT32 && !n.inputs.empty()) {
            // every input a constant the caller cannot replace, the first one
            // float, and the result no larger than what it replaces
            int64_t inputs_size = 0;
            bool constant = true;
            for (const std::string& in : n.inputs) {
                if (in.empty() || !g.is_initializer(in) || g.is_graph_input(in)) constant = false;
                else inputs_size += g.initializers.at(in).num_elements();
            }
            if (constant && g.initializers.at(n.inputs[0]).dtype == DType::FLOAT32 &&
                info->num_elements() <= inputs_size)
                if (std::optional<std::vector<float>> y = evaluate(g, n, info->shape))
                    folded = make_constant(out, info->shape, DType::FLOAT32, y->data(), y->size() * 4);
        }
        if (!folded) continue;
        n.is_dead = true;
        g.add_initializer(*folded);
        ++changes;
    }
    return changes;
}

size_t remove_identities(Graph& g) {
    size_t changes = 0;
    for (Node& n : g.nodes) {
        if (n.is_dead || (n.op != OpType::Identity && n.op != OpType::Dropout)) continue;
        if (n.inputs.empty() || n.outputs.empty() || n.inputs[0].empty() || n.outputs[0].empty()) continue;
        if (n.op == OpType::Dropout) {
            if (n.inputs.size() > 2 && !n.inputs[2].empty()) continue;                       // training_mode
            if (n.outputs.size() > 1 && !n.outputs[1].empty() && is_used(g, n.outputs[1])) continue;  // the mask
        }
        const std::string in = n.inputs[0], out = n.outputs[0];
        if (!is_graph_output(g, out)) {
            n.is_dead = true;
            rename_uses(g, out, in);
            g.value_info.erase(out);
        } else {
            // the output is part of the graph's interface, so it keeps its
            // name: the node that produces the input writes it instead. an
            // input that is a graph input, an initializer or itself a graph
            // output has no such node, and the identity stays (it costs nothing)
            Node* p = producer_of(g, in);
            if (!p || p == &n || is_graph_output(g, in)) continue;
            n.is_dead = true;
            for (std::string& o : p->outputs)
                if (o == in) o = out;
            rename_uses(g, in, out);
            g.value_info.erase(in);
        }
        if (n.outputs.size() > 1 && !n.outputs[1].empty()) g.value_info.erase(n.outputs[1]);
        ++changes;
    }
    return changes;
}

size_t fuse_relu(Graph& g) {
    size_t changes = 0;
    for (Node& relu : g.nodes) {
        if (relu.is_dead || relu.op != OpType::Relu || relu.inputs.empty() || relu.inputs[0].empty()) continue;
        const std::string mid = relu.inputs[0];
        Node* p = producer_of(g, mid);
        if (!p || (p->op != OpType::Gemm && p->op != OpType::Conv) || p->has_attr(kFusedRelu)) continue;
        // the value before the Relu must not be needed by anyone else
        if (is_graph_output(g, mid)) continue;
        size_t readers = 0;
        for (const Node& n : g.nodes)
            if (!n.is_dead && std::find(n.inputs.begin(), n.inputs.end(), mid) != n.inputs.end()) ++readers;
        if (readers != 1) continue;
        const Tensor* t = g.get_tensor_info(mid);
        if (!t || t->dtype != DType::FLOAT32) continue;
        p->add_attr(kFusedRelu, int64_t{1});
        for (std::string& o : p->outputs)
            if (o == mid) o = relu.outputs.at(0);
        relu.is_dead = true;
        g.value_info.erase(mid);
        ++changes;
    }
    return changes;
}

size_t eliminate_dead_nodes(Graph& g) {
    // walk backwards from the outputs: a node is needed if one of its outputs is
    std::unordered_set<std::string> needed(g.outputs.begin(), g.outputs.end());
    size_t removed = 0;
    std::vector<size_t> order = g.topological_sort();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node& n = g.nodes[*it];
        if (n.is_dead) continue;
        bool live = false;
        for (const std::string& o : n.outputs)
            if (!o.empty() && needed.count(o)) live = true;
        if (!live) {
            n.is_dead = true;
            for (const std::string& o : n.outputs)
                if (!o.empty()) g.value_info.erase(o);
            ++removed;
            continue;
        }
        for (const std::string& in : n.inputs)
            if (!in.empty()) needed.insert(in);
    }
    for (auto it = g.initializers.begin(); it != g.initializers.end();) {
        if (!needed.count(it->first) && !g.is_graph_input(it->first)) {
            g.value_info.erase(it->first);
            it = g.initializers.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

std::vector<PassResult> run_passes(Graph& g, const std::vector<Pass>& list) {
    std::vector<PassResult> results;
    for (const Pass& p : list) results.push_back({p.name, 0});
    for (int round = 0; round < kMaxPassRounds; ++round) {
        size_t changed = 0;
        for (size_t i = 0; i < list.size(); ++i) {
            size_t c = list[i].run(g);
            results[i].changes += c;
            changed += c;
        }
        if (changed == 0) break;
    }
    return results;
}

std::vector<PassResult> optimize(Graph& g) {
    return run_passes(g, passes());
}

} // namespace null
