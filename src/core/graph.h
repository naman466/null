#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "node.h"
#include "tensor.h"

namespace null {

// computational graph, core structure
class Graph {
public:
    std::string name;

    // nodes in original ordered, maybe reordered by toposort
    std::vector<Node> nodes;

    // value info, tensor metadata
    // maps tensor name -> Tensor (shape/dtype info, no data)
    std::unordered_map<std::string, Tensor> value_info;

    // initializers, constant tensors (model weights, biases)
    // maps tensor name -> Tensor (with data)
    std::unordered_map<std::string, Tensor> initializers;

    // graph input / output tensor names (in order)
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    Graph() = default;
    explicit Graph(std::string n) : name(std::move(n)) {}

    // add node and return ref
    Node& add_node(Node n) {
        nodes.push_back(std::move(n));
        return nodes.back();
    }

    // add metadata info
    void add_value_info(Tensor t) {
        std::string n = t.name;
        value_info[n] = std::move(t);
    }

    // add an initializer (constant tensor), also adds to value info
    void add_initializer(Tensor t) {
        t.is_constant = true;
        std::string n = t.name;
        // add to value info
        value_info[n] = t;
        initializers[n] = std::move(t);
    }

    bool is_initializer(const std::string& name) const {
        return initializers.count(name) > 0;
    }

    bool is_graph_input(const std::string& name) const {
        for (auto& s : inputs)
            if (s == name) return true;
        return false;
    }

    // look at tensor info by name
    const Tensor* get_tensor_info(const std::string& name) const {
        auto it = value_info.find(name);
        if (it != value_info.end()) return &it->second;
        return nullptr;
    }
};

} // namespace null