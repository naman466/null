#include "analysis/specialize.h"

namespace null {

namespace {

bool is_dynamic(const std::vector<int64_t>& shape) {
    for (int64_t d : shape)
        if (d < 0) return true;
    return false;
}

std::string shape_of(const std::vector<int64_t>& s) {
    return Tensor("", s, DType::UNDEFINED).shape_str();
}

} // namespace

void set_input_shape(Graph& g, const std::string& input, const std::vector<int64_t>& shape) {
    if (!g.is_graph_input(input)) {
        std::string inputs;
        for (auto& name : g.inputs) inputs += (inputs.empty() ? "" : ", ") + name;
        throw ShapeError("'" + input + "' is not a graph input (inputs: " + inputs + ")");
    }
    const Tensor* current = g.get_tensor_info(input);
    if (!current) throw ShapeError("input '" + input + "' has no metadata");

    auto fail = [&](const std::string& msg) {
        throw ShapeError("cannot set input '" + input + "' " + shape_of(current->shape) +
                         " to " + shape_of(shape) + ": " + msg);
    };
    if (shape.size() != current->shape.size())
        fail("rank " + std::to_string(shape.size()) + " instead of " +
             std::to_string(current->shape.size()));
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) fail("dim " + std::to_string(i) + " must be >= 0");
        if (current->shape[i] >= 0 && current->shape[i] != shape[i])
            fail("dim " + std::to_string(i) + " is fixed at " + std::to_string(current->shape[i]));
    }
    g.add_value_info(Tensor(input, shape, current->dtype));
}

int set_batch_size(Graph& g, int64_t n) {
    if (n < 1) throw ShapeError("batch size must be at least 1, got " + std::to_string(n));
    int changed = 0;
    for (const std::string& name : g.inputs) {
        const Tensor* t = g.get_tensor_info(name);
        if (!t || t->shape.empty() || t->shape[0] >= 0) continue;
        std::vector<int64_t> shape = t->shape;
        shape[0] = n;
        g.add_value_info(Tensor(name, shape, t->dtype));
        ++changed;
    }
    return changed;
}

std::vector<std::string> dynamic_tensors(const Graph& g) {
    std::vector<std::string> out;
    auto check = [&](const std::string& name) {
        if (name.empty()) return;
        const Tensor* t = g.get_tensor_info(name);
        if (!t || is_dynamic(t->shape)) out.push_back(name);
    };
    for (const std::string& name : g.inputs) check(name);
    for (const Node* n : g.sorted_nodes())
        for (const std::string& name : n->outputs) check(name);
    return out;
}

} // namespace null
