#pragma once
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"
#include "node.h"
#include "tensor.h"

namespace null {

// human readable text form of a graph, for the cli and for debugging passes

// [1, ?, 28], dynamic dims print as ?
inline std::string dims_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += shape[i] < 0 ? "?" : std::to_string(shape[i]);
    }
    return s + "]";
}

// float32[1, ?, 28]
inline std::string type_str(const Tensor& t) {
    return dtype_name(t.dtype) + dims_str(t.shape);
}

// 1234567 -> 1,234,567
inline std::string group_digits(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

// "1 node", "2 nodes"
inline std::string count_str(uint64_t n, const std::string& noun) {
    return group_digits(n) + " " + noun + (n == 1 ? "" : "s");
}

inline std::string bytes_str(uint64_t bytes) {
    char buf[32];
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) std::snprintf(buf, sizeof buf, "%.1f KB", bytes / 1024.0);
    else std::snprintf(buf, sizeof buf, "%.1f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

// shortest round-tripping form, 0.5 rather than 0.500000
inline std::string float_str(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

inline std::string attr_str(const Attribute& a) {
    auto join = [](const auto& values, auto fmt) {
        std::string s = "[";
        for (size_t i = 0; i < values.size(); ++i) s += (i ? ", " : "") + fmt(values[i]);
        return s + "]";
    };
    if (a.has_int())    return std::to_string(a.as_int());
    if (a.has_float())  return float_str(a.as_float());
    if (a.has_string()) return "\"" + a.as_string() + "\"";
    if (a.has_ints())   return join(a.as_ints(), [](int64_t v) { return std::to_string(v); });
    if (a.has_floats()) return join(a.as_floats(), [](float v) { return float_str(v); });
    if (a.has_tensor()) return "tensor " + type_str(a.as_tensor());
    return "?";
}

// name  type, aligned to the longest name
inline void dump_values(std::ostringstream& out, const Graph& g, const std::vector<std::string>& names) {
    size_t width = 0;
    for (auto& n : names) width = std::max(width, n.size());
    for (auto& n : names) {
        out << "  " << n << std::string(width - n.size() + 2, ' ');
        const Tensor* t = g.get_tensor_info(n);
        out << (t ? type_str(*t) : "(unknown)") << "\n";
    }
}

inline std::string dump_graph(const Graph& g) {
    std::ostringstream out;

    // weights summary
    uint64_t params = 0, weight_bytes = 0;
    for (auto& [_, t] : g.initializers) {
        params += static_cast<uint64_t>(t.num_elements());
        weight_bytes += t.data.size();
    }
    out << "graph '" << g.name << "': " << count_str(g.live_node_count(), "node") << ", "
        << count_str(g.initializers.size(), "initializer") << ", " << count_str(params, "parameter")
        << " (" << bytes_str(weight_bytes) << ")\n";

    out << "\ninputs\n";
    dump_values(out, g, g.inputs);
    out << "\noutputs\n";
    dump_values(out, g, g.outputs);

    if (!g.initializers.empty()) {
        // sorted, the map's own order is unspecified
        std::vector<std::string> names;
        for (auto& [name, _] : g.initializers) names.push_back(name);
        std::sort(names.begin(), names.end());
        size_t width = 0, type_width = 0;
        for (auto& n : names) {
            width = std::max(width, n.size());
            type_width = std::max(type_width, type_str(g.initializers.at(n)).size());
        }
        out << "\ninitializers\n";
        for (auto& n : names) {
            const Tensor& t = g.initializers.at(n);
            std::string type = type_str(t);
            out << "  " << n << std::string(width - n.size() + 2, ' ')
                << type << std::string(type_width - type.size() + 2, ' ')
                << bytes_str(t.data.size()) << "\n";
        }
    }

    // nodes in execution order, outputs annotated with their type when known
    out << "\nnodes\n";
    std::vector<const Node*> sorted = g.sorted_nodes();
    size_t index_width = std::to_string(sorted.empty() ? 0 : sorted.size() - 1).size();
    for (size_t i = 0; i < sorted.size(); ++i) {
        const Node& n = *sorted[i];
        std::string idx = std::to_string(i);
        out << "  " << std::string(index_width - idx.size(), ' ') << idx << "  "
            << n.name << ": " << optype_to_str(n.op) << "(";
        for (size_t k = 0; k < n.inputs.size(); ++k)
            out << (k ? ", " : "") << (n.inputs[k].empty() ? "_" : n.inputs[k]);
        out << ") -> ";
        for (size_t k = 0; k < n.outputs.size(); ++k) {
            const std::string& o = n.outputs[k];
            out << (k ? ", " : "") << (o.empty() ? "_" : o);
            if (const Tensor* t = o.empty() ? nullptr : g.get_tensor_info(o)) out << ": " << type_str(*t);
        }
        if (!n.attrs.empty()) {
            std::vector<std::string> keys;
            for (auto& [k, _] : n.attrs) keys.push_back(k);
            std::sort(keys.begin(), keys.end());
            out << "  {";
            for (size_t k = 0; k < keys.size(); ++k)
                out << (k ? ", " : "") << keys[k] << "=" << attr_str(n.attrs.at(keys[k]));
            out << "}";
        }
        out << "\n";
    }
    return out.str();
}

} // namespace null
