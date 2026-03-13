#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <unordered_set>   
#include <algorithm>       
#include <stdexcept>    
#include <cstddef>      
#include <sstream>   
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

    // return nodes in topological order and skip dead ones
    std::vector<const Node*> sorted_nodes() const {
        auto order = topological_sort();
        std::vector<const Node*> result;
        for (size_t idx : order) {
            if (!nodes[idx].is_dead) {
                result.push_back(&nodes[idx]);
            }
        }
        return result;
    }

    // validate graph and check if all inputs are defined
    std::vector<std::string> validate() const {
        std::vector<std::string> errors;

        // build list of all define d names
        std::unordered_set<std::string> defined;
        for (auto& s : inputs) defined.insert(s);
        for (auto& [name, _] : initializers) defined.insert(name);

        // traverse in topo order and check
        std::vector<size_t> order;
        try {
            order = topological_sort();
        } catch (std::exception& e) {
            errors.push_back(std::string("Cycle detected: ") + e.what());
            return errors;
        }

        for (size_t idx : order) {
            const Node& n = nodes[idx];
            if (n.is_dead) continue;
            for (auto& inp : n.inputs) {
                if (!inp.empty() && defined.count(inp) == 0) {
                    errors.push_back("Node '" + n.name + "' uses undefined tensor '" + inp + "'");
                }
            }
            for (auto& out : n.outputs) {
                if (!out.empty()) defined.insert(out);
            }
        }

        for (auto& out : outputs) {
            if (defined.count(out) == 0) {
                errors.push_back("Graph output '" + out + "' is not defined");
            }
        }
        return errors;
    }

    // count live nodes
    size_t live_node_count() const {
        size_t count = 0;
        for (auto& n : nodes) {
            if (!n.is_dead) count++;
        }
        return count;
    }

    std::string summary() const {
        std::ostringstream ss;
        ss << "Graph: " << name << "\n";
        ss << "  Inputs: ";
        for (auto& s : inputs) ss << s << " ";
        ss << "\n  Outputs: ";
        for (auto& s : outputs) ss << s << " ";
        ss << "\n  Nodes: " << nodes.size()
           << " (live: " << live_node_count() << ")\n";
        ss << "  Initializers: " << initializers.size() << "\n";
        return ss.str();
    }
    
};

} // namespace null