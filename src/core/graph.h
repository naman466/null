#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <unordered_set>   
#include <algorithm>       
#include <stdexcept>    
#include <cstddef>         
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

    // topo sort of the nodes using Kahn's algo
    // returns sorted node indices, stable sort
    std::vector<size_t> topological_sort() const {
        size_t N = nodes.size();
        // map from output tensor name -> node index
        std::unordered_map<std::string, size_t> producer;
        for (size_t i = 0; i < N; ++i) {
            for (auto& out : nodes[i].outputs) {
                producer[out] = i;
            }
        }

        // build adjacency: node i -> set of nodes that consume i's outputs
        std::vector<std::unordered_set<size_t>> successors(N);
        std::vector<int> in_degree(N, 0);

        for (size_t i = 0; i < N; ++i) {
            for (auto& inp : nodes[i].inputs) {
                auto it = producer.find(inp);
                if (it != producer.end()) {
                    size_t src = it->second;
                    if (successors[src].insert(i).second) {
                        in_degree[i]++;
                    }
                }
            }
        }

        // Kahn's algorithm uses queue
        // maintain original order by using sorted ready list
        std::vector<size_t> ready;
        for (size_t i = 0; i < N; ++i) {
            if (in_degree[i] == 0) ready.push_back(i);
        }

        std::vector<size_t> order;
        order.reserve(N);

        while (!ready.empty()) {
            // pick smallest index for stability 
            size_t idx = ready.front();
            ready.erase(ready.begin());
            order.push_back(idx);

            // collect successors and add newly addd ones
            std::vector<size_t> newly_ready;
            for (size_t succ : successors[idx]) {
                if (--in_degree[succ] == 0) {
                    newly_ready.push_back(succ);
                }
            }
            std::sort(newly_ready.begin(), newly_ready.end());
            // insert in sorted position to maintain stability
            for (size_t nr : newly_ready) {
                auto it = std::lower_bound(ready.begin(), ready.end(), nr);
                ready.insert(it, nr);
            }
        }

        if (order.size() != N) {
            throw std::runtime_error("Graph has a cycle; topological sort failed");
        }
        return order;
    }
};

} // namespace null