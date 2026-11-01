#include "opt/optimize.h"

#include <algorithm>
#include <unordered_set>

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

// in the order they run; identities first, so the nodes they bypass can die
const std::vector<Pass>& passes() {
    static const std::vector<Pass> list = {
        {"remove-identities", remove_identities},
        {"dead-code", eliminate_dead_nodes},
    };
    return list;
}

} // namespace

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
