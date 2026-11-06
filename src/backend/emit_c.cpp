#include "backend/emit_c.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "analysis/shape_inference.h"
#include "core/dump.h"
#include "null_runtime.h"   // the same broadcast_strides the generated code relies on

namespace null {

namespace {

std::string where(const Node& n) {
    return "node '" + n.name + "' (" + optype_to_str(n.op) + ")";
}

// text from the model (graph, node and tensor names) made safe to write
// inside a C block comment: printable ASCII only, and no "*/" that would end
// the comment early (and let a name inject code), no "/*" (-Wcomment) and no
// "??" (a trigraph like ??/ can splice lines). anything else becomes '?'
std::string comment_text(const std::string& s) {
    std::string out;
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        const char safe = c >= 0x20 && c < 0x7f ? ch : '?';
        if (!out.empty()) {
            const char prev = out.back();
            if ((prev == '*' && safe == '/') || (prev == '/' && safe == '*') || (prev == '?' && safe == '?'))
                out += ' ';
        }
        out += safe;
    }
    return out;
}

std::string int_list(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

// the runtime part whose kernel an operator's code calls, or nullptr when
// it calls none (aliases and constants)
const char* runtime_part(OpType op) {
    switch (op) {
        case OpType::Relu:
        case OpType::Sigmoid:
        case OpType::Tanh:
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Abs:
        case OpType::Neg:
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Clip:    return "elementwise";
        case OpType::Gemm:
        case OpType::MatMul:  return "matmul";
        case OpType::Softmax: return "softmax";
        case OpType::Conv:    return "conv";
        case OpType::MaxPool: return "pool";
        case OpType::GlobalAveragePool:
        case OpType::ReduceMean: return "reduce";
        case OpType::LayerNormalization: return "norm";
        default:              return nullptr;    // Transpose and Gather add "layout" only when they call it
    }
}

class Emitter {
public:
    Emitter(const Graph& g, const EmitOptions& o) : g_(g), opt_(o) {}

    std::string run() {
        check_static();
        declare_inputs();
        std::vector<const Node*> order = g_.sorted_nodes();
        // a node no output depends on gets no code, even unoptimised: its
        // weights would be arrays nothing reads, which -Werror rejects
        std::set<std::string> needed(g_.outputs.begin(), g_.outputs.end());
        std::set<const Node*> wanted;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            bool used = false;
            for (const std::string& out : (*it)->outputs) used = used || (!out.empty() && needed.count(out));
            if (!used) continue;
            wanted.insert(*it);
            for (const std::string& in : (*it)->inputs) needed.insert(in);
        }
        for (const Node* n : order) {
            if (wanted.count(n)) {
                ++step_;
                emit_node(*n);
            } else {
                body_ << "    /* " << comment_text(n->name) << ": " << optype_to_str(n->op)
                      << ", no output depends on it */\n";
            }
        }
        return assemble();
    }

private:
    const Graph& g_;
    const EmitOptions& opt_;
    std::map<std::string, std::string> expr_;   // tensor name -> C expression
    std::set<std::string> idents_;              // every C identifier used
    std::set<std::string> parts_ = {"core"};    // runtime parts the code calls
    std::ostringstream decls_, body_;
    std::vector<std::string> params_;
    uint64_t buffer_bytes_ = 0, weight_bytes_ = 0;
    int temp_ = 0;

    // an intermediate buffer, and the steps (emitted nodes, counted from 1)
    // during which it must keep its value: written at `first`, last read at
    // `last`. memory planning gives it `offset`, in floats, in one arena
    struct Buffer {
        std::string id, tensor;
        int64_t count;
        int first, last;
        int64_t offset = 0;
    };
    std::vector<Buffer> buffers_;
    std::map<std::string, size_t> root_;        // tensor name -> the buffer its data lives in
    int step_ = 0;
    int64_t arena_ = 0;                         // floats, after planning
    static constexpr int64_t kAlign = 16;       // buffers start on 64-byte boundaries
    static constexpr int kToTheEnd = 1 << 30;   // the last step of a graph output: after every node

    [[noreturn]] void fail(const std::string& msg) const { throw EmitError(msg); }

    const Tensor& info(const std::string& name) const {
        const Tensor* t = g_.get_tensor_info(name);
        if (!t) fail("tensor '" + name + "' has no shape, run shape inference first");
        return *t;
    }

    // a fresh C identifier for a tensor name: non [A-Za-z0-9] become _,
    // and a numeric suffix keeps it unique
    std::string ident(const std::string& prefix, const std::string& name) {
        std::string base = prefix;
        for (char c : name) base += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        std::string id = base;
        for (int k = 2; idents_.count(id); ++k) id = base + "_" + std::to_string(k);
        idents_.insert(id);
        return id;
    }

    std::string temp(const std::string& what) { return ident("k" + std::to_string(temp_++) + "_", what); }

    // every tensor a compute op touches must have a fully known shape
    void check_static() const {
        auto check = [&](const std::string& name) {
            const Tensor& t = info(name);
            for (int64_t d : t.shape)
                if (d < 0) fail("tensor '" + name + "' has dynamic shape " + type_str(t) +
                                ", fix the input shapes first");
        };
        for (auto& name : g_.inputs) check(name);
        for (const Node* n : g_.sorted_nodes())
            for (auto& out : n->outputs)
                if (!out.empty()) check(out);
    }

    void declare_inputs() {
        for (auto& name : g_.inputs) {
            const Tensor& t = info(name);
            if (t.dtype != DType::FLOAT32)
                fail("input '" + name + "' is " + dtype_name(t.dtype) + ", only float32 inputs are supported");
            std::string id = ident("in_", name);
            params_.push_back("const float* " + id);
            expr_[name] = id;
        }
    }

    // the C expression holding a float tensor, emitting a weight array on first use
    std::string read(const Node& n, size_t i) {
        if (i >= n.inputs.size() || n.inputs[i].empty())
            fail(where(n) + ": missing input " + std::to_string(i));
        const std::string& name = n.inputs[i];
        if (auto it = expr_.find(name); it != expr_.end()) {
            // reading a buffer, directly or through an alias, keeps it alive until now
            if (auto r = root_.find(name); r != root_.end())
                buffers_[r->second].last = std::max(buffers_[r->second].last, step_);
            return it->second;
        }
        if (auto it = g_.initializers.find(name); it != g_.initializers.end())
            return emit_constant(n, name, it->second);
        // a Constant node's value, as an unoptimised graph has it: emitted on
        // first use like an initializer, so a value only read at compile time
        // (a Clip bound) leaves no array behind for -Werror to reject
        for (const Node& c : g_.nodes)
            if (!c.is_dead && c.op == OpType::Constant && !c.outputs.empty() && c.outputs[0] == name &&
                c.has_attr("value"))
                return emit_constant(n, name, c.get_attr("value").as_tensor());
        fail(where(n) + ": input '" + name + "' has no value");
    }

    std::string emit_constant(const Node& n, const std::string& name, const Tensor& t) {
        if (t.dtype != DType::FLOAT32)
            fail(where(n) + ": input '" + name + "' is a " + dtype_name(t.dtype) +
                 " constant, only float32 data can be used here");
        const int64_t count = t.num_elements();
        std::string id = ident("w_", name);
        decls_ << "/* " << comment_text(name) << ": " << type_str(t) << " */\n"
               << "static const float " << id << "[" << (count ? count : 1) << "] = {";
        const float* p = t.float_ptr();
        for (int64_t i = 0; i < count; ++i)
            decls_ << (i % 6 ? " " : "\n    ") << c_float_literal(p[i]) << ",";
        if (!count) decls_ << "0";
        decls_ << "\n};\n\n";
        weight_bytes_ += static_cast<uint64_t>(count) * 4;
        expr_[name] = id;
        return id;
    }

    // a static buffer for a node output
    std::string write(const Node& n, size_t i) {
        const std::string& name = n.outputs.at(i);
        const Tensor& t = info(name);
        if (t.dtype != DType::FLOAT32)
            fail(where(n) + ": output '" + name + "' is " + dtype_name(t.dtype) + ", only float32 is supported");
        const int64_t count = t.num_elements();
        std::string id = ident("t_", name);
        buffers_.push_back({id, name, count ? count : 1, step_, step_});
        root_[name] = buffers_.size() - 1;
        buffer_bytes_ += static_cast<uint64_t>(((count ? count : 1) + kAlign - 1) / kAlign * kAlign) * 4;   // as --no-plan lays it out
        expr_[name] = id;
        return id;
    }

    // `out` is `in` seen differently (a new shape, or a pointer into it): it
    // is the same buffer, which stays alive while either is read
    void alias(const std::string& out, const std::string& in, const std::string& expr) {
        expr_[out] = expr;
        if (auto r = root_.find(in); r != root_.end()) root_[out] = r->second;
    }

    // memory planning. two buffers alive at the same step (their [first, last]
    // ranges meet) need disjoint memory; any others may share it. largest
    // first, each goes at the lowest aligned offset clear of every placed
    // buffer it meets in time. a step's inputs and outputs always meet, so no
    // kernel ever writes over what it reads. unplanned, every buffer gets its
    // own range, which is what separate static arrays amounted to
    void plan_memory() {
        auto aligned = [](int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; };
        std::vector<size_t> order(buffers_.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return buffers_[a].count > buffers_[b].count; });
        std::vector<size_t> placed;
        for (size_t i : order) {
            Buffer& b = buffers_[i];
            // the ranges taken by placed buffers alive at the same time, by offset
            std::vector<std::pair<int64_t, int64_t>> taken;
            for (size_t j : placed) {
                const Buffer& o = buffers_[j];
                if (!opt_.plan_memory || (o.first <= b.last && b.first <= o.last))
                    taken.push_back({o.offset, o.offset + aligned(o.count)});
            }
            std::sort(taken.begin(), taken.end());
            int64_t at = 0;
            for (const auto& [lo, hi] : taken) {
                if (at + b.count <= lo) break;       // fits in the gap before this one
                at = std::max(at, hi);
            }
            b.offset = at;
            arena_ = std::max(arena_, at + aligned(b.count));
            placed.push_back(i);
        }
    }

    std::string when(int step) const { return step >= kToTheEnd ? "the end" : "step " + std::to_string(step); }

    // a static const int64 array for shapes and strides
    std::string int_array(const std::string& what, const std::vector<int64_t>& v) {
        std::string id = temp(what);
        decls_ << "static const int64_t " << id << "[" << (v.empty() ? 1 : v.size()) << "] = {"
               << (v.empty() ? "0" : int_list(v)) << "};\n";
        return id;
    }

    std::vector<int64_t> strides(const std::vector<int64_t>& out, const std::vector<int64_t>& in) {
        std::vector<int64_t> s(out.size() ? out.size() : 1, 0);
        null_broadcast_strides(static_cast<int>(out.size()), out.data(),
                               static_cast<int>(in.size()), in.data(), s.data());
        s.resize(out.size());
        return s;
    }

    void emit_node(const Node& n) {
        body_ << "    /* " << comment_text(n.name) << ": " << optype_to_str(n.op) << " */\n";
        if (const char* part = runtime_part(n.op)) parts_.insert(part);
        switch (n.op) {
            case OpType::Relu:    emit_unary(n, "relu"); break;
            case OpType::Sigmoid: emit_unary(n, "sigmoid"); break;
            case OpType::Tanh:    emit_unary(n, "tanh"); break;
            case OpType::Exp:     emit_unary(n, "exp"); break;
            case OpType::Log:     emit_unary(n, "log"); break;
            case OpType::Sqrt:    emit_unary(n, "sqrt"); break;
            case OpType::Abs:     emit_unary(n, "abs"); break;
            case OpType::Neg:     emit_unary(n, "neg"); break;
            case OpType::Clip: emit_clip(n); break;
            case OpType::Add: emit_binary(n, "add"); break;
            case OpType::Sub: emit_binary(n, "sub"); break;
            case OpType::Mul: emit_binary(n, "mul"); break;
            case OpType::Div: emit_binary(n, "div"); break;
            case OpType::Gemm:    emit_gemm(n); break;
            case OpType::MatMul:  emit_matmul(n); break;
            case OpType::Softmax: emit_softmax(n); break;
            case OpType::Conv:    emit_conv(n); break;
            case OpType::MaxPool: emit_maxpool(n); break;
            case OpType::GlobalAveragePool:
            case OpType::ReduceMean: emit_reduce_mean(n); break;
            case OpType::LayerNormalization: emit_layernorm(n); break;
            case OpType::Transpose: emit_transpose(n); break;
            case OpType::Reshape:
            case OpType::Flatten:
            case OpType::Identity:
            case OpType::Dropout:
                emit_alias(n);
                break;
            case OpType::Constant:
                emit_constant_node(n);
                break;
            case OpType::Shape:
            case OpType::Gather:
            case OpType::Concat:
            case OpType::Unsqueeze:
            case OpType::Squeeze:
                if (emit_computed(n)) break;
                if (n.op == OpType::Unsqueeze || n.op == OpType::Squeeze) {
                    emit_alias(n);
                    break;
                }
                if (n.op == OpType::Gather) {
                    emit_gather(n);
                    break;
                }
                fail(where(n) + ": no code generation for this operator yet (only for shape values "
                     "known at compile time)");
            default:
                fail(where(n) + ": no code generation for this operator yet");
        }
    }

    // an int64 output whose value shape inference computed, like a Reshape
    // target built from Shape: it exists only at compile time, so no code
    bool emit_computed(const Node& n) {
        const std::string& out = n.outputs.at(0);
        if (info(out).dtype != DType::INT64) return false;
        std::optional<std::vector<int64_t>> v = computed_value(g_, out);
        if (!v && info(out).num_elements() != 0) return false;
        std::string shown = v ? int_list(std::vector<int64_t>(v->begin(), v->begin() + std::min<size_t>(v->size(), 8))) : "";
        if (v && v->size() > 8) shown += ", ...";
        body_ << "    /* " << comment_text(out) << " = [" << shown << "], computed at compile time */\n";
        return true;
    }

    // same bytes, new shape: the output is another name for the input
    void emit_alias(const Node& n) {
        alias(n.outputs.at(0), n.inputs.at(0), read(n, 0));
        body_ << "    /* " << comment_text(n.outputs[0]) << " is " << expr_[n.outputs[0]] << ", no data moves */\n";
        if (n.op == OpType::Dropout && n.outputs.size() > 1 && !n.outputs[1].empty())
            for (const Node& other : g_.nodes)
                for (auto& in : other.inputs)
                    if (in == n.outputs[1]) fail(where(n) + ": the dropout mask output is not supported");
    }

    // float constants become weights; others (shape tensors) are only used at compile time
    void emit_constant_node(const Node& n) {
        if (!n.has_attr("value")) fail(where(n) + ": only Constant with a 'value' tensor is supported");
        body_ << "    /* " << comment_text(n.outputs[0]) << " is a compile-time constant */\n";
    }

    void emit_binary(const Node& n, const char* op) {
        std::string a = read(n, 0), b = read(n, 1), y = write(n, 0);
        const auto& sa = info(n.inputs[0]).shape;
        const auto& sb = info(n.inputs[1]).shape;
        const auto& so = info(n.outputs[0]).shape;
        if (sa == so && sb == so) {
            body_ << "    null_" << op << "_f32(" << a << ", " << b << ", " << y << ", "
                  << info(n.outputs[0]).num_elements() << ");\n";
            return;
        }
        std::string shape = int_array(n.name + "_shape", so);
        std::string st_a = int_array(n.name + "_a_strides", strides(so, sa));
        std::string st_b = int_array(n.name + "_b_strides", strides(so, sb));
        body_ << "    null_" << op << "_bcast_f32(" << a << ", " << b << ", " << y << ", " << so.size()
              << ", " << shape << ", " << st_a << ", " << st_b << ");\n";
    }

    void emit_gemm(const Node& n) {
        std::string a = read(n, 0), b = read(n, 1);
        const bool has_c = n.inputs.size() > 2 && !n.inputs[2].empty();
        std::string c = has_c ? read(n, 2) : "NULL";
        std::string y = write(n, 0);
        const auto& sa = info(n.inputs[0]).shape;
        const auto& so = info(n.outputs[0]).shape;
        const int64_t trans_a = n.attr_int("transA", 0) != 0, trans_b = n.attr_int("transB", 0) != 0;
        const int64_t m = so[0], nn = so[1], k = trans_a ? sa[0] : sa[1];
        int64_t c_rs = 0, c_cs = 0;
        if (has_c) {
            auto s = strides(so, info(n.inputs[2]).shape);
            c_rs = s[0];
            c_cs = s[1];
        }
        body_ << "    null_gemm_f32(" << a << ", " << b << ", " << c << ", " << y << ", " << m << ", " << nn
              << ", " << k << ", " << trans_a << ", " << trans_b << ", "
              << c_float_literal(n.attr_float("alpha", 1.0f)) << ", " << c_float_literal(n.attr_float("beta", 1.0f))
              << ", " << c_rs << ", " << c_cs << ", " << n.attr_int(kFusedRelu, 0) << ");\n";
    }

    void emit_matmul(const Node& n) {
        std::string a = read(n, 0), b = read(n, 1), y = write(n, 0);
        const auto& sa = info(n.inputs[0]).shape;
        const auto& sb = info(n.inputs[1]).shape;
        const bool a_vec = sa.size() == 1, b_vec = sb.size() == 1;
        const int64_t m = a_vec ? 1 : sa[sa.size() - 2], k = sa.back(), nn = b_vec ? 1 : sb.back();
        std::vector<int64_t> ba(sa.begin(), sa.end() - (a_vec ? 1 : 2));
        std::vector<int64_t> bb(sb.begin(), sb.end() - (b_vec ? 1 : 2));
        std::vector<int64_t> batch = broadcast_shapes(ba, bb);
        std::vector<int64_t> st_a = strides(batch, ba), st_b = strides(batch, bb);
        for (auto& s : st_a) s *= m * k;
        for (auto& s : st_b) s *= k * nn;
        std::string shape = int_array(n.name + "_batch", batch);
        std::string sta = int_array(n.name + "_a_strides", st_a);
        std::string stb = int_array(n.name + "_b_strides", st_b);
        body_ << "    null_matmul_f32(" << a << ", " << b << ", " << y << ", " << m << ", " << nn << ", " << k
              << ", " << batch.size() << ", " << shape << ", " << sta << ", " << stb << ");\n";
    }

    void emit_conv(const Node& n) {
        std::string x = read(n, 0), w = read(n, 1);
        const bool has_b = n.inputs.size() > 2 && !n.inputs[2].empty();
        std::string b = has_b ? read(n, 2) : "NULL";
        std::string y = write(n, 0);
        // the same geometry the shape rule used, so padding always agrees
        ConvGeometry g = conv_geometry(n, info(n.inputs[0]).shape, info(n.inputs[1]).shape);
        std::string params = temp(n.name + "_conv");
        decls_ << "static const null_conv2d_params " << params << " = {"
               << g.n << ", " << g.c << ", " << g.h << ", " << g.w << ", "
               << g.m << ", " << g.kh << ", " << g.kw << ", " << g.group << ", "
               << g.stride_h << ", " << g.stride_w << ", " << g.dil_h << ", " << g.dil_w << ", "
               << g.pad_top << ", " << g.pad_left << ", " << g.out_h << ", " << g.out_w << ", "
               << n.attr_int(kFusedRelu, 0) << "};\n";
        body_ << "    null_conv2d_f32(" << x << ", " << w << ", " << b << ", " << y << ", &" << params << ");\n";
    }

    void emit_maxpool(const Node& n) {
        std::string x = read(n, 0), y = write(n, 0);
        PoolGeometry g = pool_geometry(n, info(n.inputs[0]).shape);
        std::string params = temp(n.name + "_pool");
        decls_ << "static const null_maxpool2d_params " << params << " = {"
               << g.n << ", " << g.c << ", " << g.h << ", " << g.w << ", " << g.kh << ", " << g.kw << ", "
               << g.stride_h << ", " << g.stride_w << ", " << g.dil_h << ", " << g.dil_w << ", "
               << g.pad_top << ", " << g.pad_left << ", " << g.out_h << ", " << g.out_w << "};\n";
        body_ << "    null_maxpool2d_f32(" << x << ", " << y << ", &" << params << ");\n";
    }

    // size-1 dims are dropped and neighbours that are both kept or both
    // reduced merged, so a GlobalAveragePool is [N*C kept, H*W reduced]
    void emit_reduce_mean(const Node& n) {
        std::string x = read(n, 0), y = write(n, 0);
        const auto& s = info(n.inputs[0]).shape;
        std::vector<bool> reduce = reduced_axes(n, g_);
        std::vector<int64_t> dims, flags;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == 1) continue;
            if (!flags.empty() && flags.back() == reduce[i]) dims.back() *= s[i];
            else {
                dims.push_back(s[i]);
                flags.push_back(reduce[i]);
            }
        }
        if (dims.size() > NULL_MAX_RANK)
            fail(where(n) + ": the reduction alternates kept and reduced dims more than " +
                 std::to_string(NULL_MAX_RANK) + " times");
        std::string params = temp(n.name + "_reduce");
        decls_ << "static const null_reduce_params " << params << " = {" << dims.size() << ", {"
               << (dims.empty() ? "0" : int_list(dims)) << "}, {" << (flags.empty() ? "0" : int_list(flags))
               << "}};\n";
        body_ << "    null_reduce_mean_f32(" << x << ", " << y << ", &" << params << ");\n";
    }

    // a float scalar known at compile time: an initializer, or a Constant node
    // (which the optimiser turns into an initializer, but --no-opt does not)
    std::optional<float> constant_float(const std::string& name) const {
        const Tensor* t = nullptr;
        if (auto it = g_.initializers.find(name); it != g_.initializers.end()) t = &it->second;
        for (const Node& c : g_.nodes)
            if (!t && !c.is_dead && c.op == OpType::Constant && !c.outputs.empty() && c.outputs[0] == name) {
                if (c.has_attr("value_float")) return c.attr_float("value_float");
                if (c.has_attr("value")) t = &c.get_attr("value").as_tensor();
            }
        if (!t || t->dtype != DType::FLOAT32 || t->num_elements() != 1 || t->data.size() != sizeof(float))
            return std::nullopt;
        float v;
        std::memcpy(&v, t->data.data(), sizeof v);
        return v;
    }

    // y = f(x) elementwise, by null_<kernel>_f32(x, y, n)
    void emit_unary(const Node& n, const char* kernel) {
        std::string x = read(n, 0), y = write(n, 0);
        body_ << "    null_" << kernel << "_f32(" << x << ", " << y << ", " << info(n.outputs[0]).num_elements() << ");\n";
    }

    // Clip(x, min?, max?) with bounds known at compile time. a missing bound
    // is the most negative or most positive finite float: the onnx spec says
    // they default to numeric_limits lowest() and max(), so even a Clip with
    // no bounds turns infinities into finite numbers (onnxruntime agrees; the
    // onnx reference evaluator keeps them)
    void emit_clip(const Node& n) {
        float bound[2] = {-FLT_MAX, FLT_MAX};
        for (size_t i = 1; i < 3; ++i) {
            if (i >= n.inputs.size() || n.inputs[i].empty()) continue;
            std::optional<float> v = constant_float(n.inputs[i]);
            const char* which = i == 1 ? "min" : "max";
            if (!v) fail(where(n) + ": the " + which + " bound must be a float constant known at compile time");
            if (std::isnan(*v)) fail(where(n) + ": a NaN " + which + " bound is not supported");
            bound[i - 1] = *v;
        }
        std::string x = read(n, 0), y = write(n, 0);
        body_ << "    null_clip_f32(" << x << ", " << y << ", " << info(n.outputs[0]).num_elements() << ", "
              << c_float_literal(bound[0]) << ", " << c_float_literal(bound[1]) << ");\n";
    }

    void emit_layernorm(const Node& n) {
        std::string x = read(n, 0), scale = read(n, 1);
        const bool has_b = n.inputs.size() > 2 && !n.inputs[2].empty();
        std::string b = has_b ? read(n, 2) : "NULL";
        std::string y = write(n, 0);
        const auto& s = info(n.inputs[0]).shape;
        int64_t axis = n.attr_int("axis", -1);
        if (axis < 0) axis += static_cast<int64_t>(s.size());
        int64_t outer = 1, count = 1;
        for (size_t i = 0; i < s.size(); ++i) (static_cast<int64_t>(i) < axis ? outer : count) *= s[i];
        // the kernel takes one scale and one bias value per normalized element
        if (info(n.inputs[1]).num_elements() != count || (has_b && info(n.inputs[2]).num_elements() != count))
            fail(where(n) + ": a scale or bias that broadcasts within the normalized dims is not supported yet");
        body_ << "    null_layernorm_f32(" << x << ", " << scale << ", " << b << ", " << y << ", " << outer << ", "
              << count << ", " << c_float_literal(n.attr_float("epsilon", 1e-5f)) << ");\n";
    }

    // dims of size 1 are dropped first: they cannot change the order of the
    // bytes. if the others keep their order the output is the same memory and
    // nothing moves, as for most transposes in a batch of one
    void emit_transpose(const Node& n) {
        const auto& s = info(n.inputs[0]).shape;
        std::vector<int64_t> perm = transpose_perm(n, s.size());
        std::vector<int64_t> stride(s.size(), 1), out_shape, x_strides;
        for (size_t d = s.size(); d-- > 1;) stride[d - 1] = stride[d] * s[d];
        for (int64_t p : perm) {
            if (s[static_cast<size_t>(p)] == 1) continue;
            out_shape.push_back(s[static_cast<size_t>(p)]);
            x_strides.push_back(stride[static_cast<size_t>(p)]);
        }
        if (std::is_sorted(x_strides.begin(), x_strides.end(), std::greater<int64_t>())) {
            emit_alias(n);
            return;
        }
        if (out_shape.size() > NULL_MAX_RANK)
            fail(where(n) + ": more than " + std::to_string(NULL_MAX_RANK) + " dims of size above 1 are permuted");
        parts_.insert("layout");
        std::string x = read(n, 0), y = write(n, 0);
        std::string shape = int_array(n.name + "_shape", out_shape);
        std::string strides_id = int_array(n.name + "_x_strides", x_strides);
        body_ << "    null_transpose_f32(" << x << ", " << y << ", " << out_shape.size() << ", " << shape << ", "
              << strides_id << ");\n";
    }

    // the int64 values of a tensor known at compile time: an initializer, a
    // Constant node or a value shape inference computed
    std::optional<std::vector<int64_t>> constant_ints(const std::string& name) const {
        const Tensor* t = nullptr;
        if (auto it = g_.initializers.find(name); it != g_.initializers.end()) t = &it->second;
        for (const Node& c : g_.nodes)
            if (!t && !c.is_dead && c.op == OpType::Constant && !c.outputs.empty() && c.outputs[0] == name) {
                if (c.has_attr("value_ints")) return c.attr_ints("value_ints");
                if (c.has_attr("value_int")) return std::vector<int64_t>{c.attr_int("value_int")};
                if (c.has_attr("value")) t = &c.get_attr("value").as_tensor();
            }
        if (t) {
            if (t->dtype != DType::INT64) return std::nullopt;
            const int64_t count = t->num_elements();
            if (count <= 0) return std::vector<int64_t>{};
            return std::vector<int64_t>(t->int64_ptr(), t->int64_ptr() + count);
        }
        return computed_value(g_, name);
    }

    // float Gather with indices known at compile time. a single index with
    // nothing before the axis picks one contiguous block: a pointer into the
    // input, no data moves
    void emit_gather(const Node& n) {
        std::optional<std::vector<int64_t>> indices = constant_ints(n.inputs.at(1));
        if (!indices) fail(where(n) + ": the indices must be int64 and known at compile time");
        const auto& s = info(n.inputs[0]).shape;
        int64_t axis = n.attr_int("axis", 0);
        if (axis < 0) axis += static_cast<int64_t>(s.size());
        const int64_t count = s[static_cast<size_t>(axis)];
        int64_t outer = 1, inner = 1;
        for (size_t i = 0; i < s.size(); ++i) {
            if (static_cast<int64_t>(i) < axis) outer *= s[i];
            if (static_cast<int64_t>(i) > axis) inner *= s[i];
        }
        for (int64_t& k : *indices) if (k < 0) k += count;    // the shape rule checked the range
        std::string x = read(n, 0);
        if (outer == 1 && indices->size() == 1) {
            const int64_t offset = (*indices)[0] * inner;
            alias(n.outputs.at(0), n.inputs.at(0), offset ? "(" + x + " + " + std::to_string(offset) + ")" : x);
            body_ << "    /* " << comment_text(n.outputs[0]) << " is " << expr_[n.outputs[0]] << ", no data moves */\n";
            return;
        }
        parts_.insert("layout");
        std::string y = write(n, 0);
        std::string idx = int_array(n.name + "_indices", *indices);
        body_ << "    null_gather_f32(" << x << ", " << y << ", " << outer << ", " << count << ", " << inner << ", "
              << idx << ", " << indices->size() << ");\n";
    }

    void emit_softmax(const Node& n) {
        std::string x = read(n, 0), y = write(n, 0);
        const auto& s = info(n.inputs[0]).shape;
        const int64_t rank = static_cast<int64_t>(s.size());
        int64_t axis = n.attr_int("axis", -1);
        if (axis < 0) axis += rank;
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; ++i) outer *= s[static_cast<size_t>(i)];
        for (int64_t i = axis + 1; i < rank; ++i) inner *= s[static_cast<size_t>(i)];
        body_ << "    null_softmax_f32(" << x << ", " << y << ", " << outer << ", " << s[static_cast<size_t>(axis)]
              << ", " << inner << ");\n";
    }

    std::string assemble() {
        // outputs: copy the result into the caller's buffer
        std::ostringstream tail;
        for (auto& name : g_.outputs) {
            const Tensor& t = info(name);
            if (t.dtype != DType::FLOAT32)
                fail("output '" + name + "' is " + dtype_name(t.dtype) + ", only float32 outputs are supported");
            if (!expr_.count(name)) fail("output '" + name + "' is never computed");
            if (auto r = root_.find(name); r != root_.end()) buffers_[r->second].last = kToTheEnd;
            std::string id = ident("out_", name);
            params_.push_back("float* " + id);
            tail << "    for (i = 0; i < " << t.num_elements() << "; ++i) " << id << "[i] = "
                 << expr_[name] << "[i];\n";
        }

        std::ostringstream out;
        out << "/*\n * generated by null from graph '" << comment_text(g_.name) << "', do not edit\n *\n";
        for (auto& name : g_.inputs) out << " * input  " << comment_text(name) << ": " << type_str(info(name)) << "\n";
        for (auto& name : g_.outputs) out << " * output " << comment_text(name) << ": " << type_str(info(name)) << "\n";
        plan_memory();
        out << " *\n * weights " << bytes_str(weight_bytes_) << ", intermediate buffers "
            << bytes_str(static_cast<uint64_t>(arena_) * 4) << " in one static arena";
        if (opt_.plan_memory)
            out << " (" << bytes_str(buffer_bytes_) << " if each buffer had its own memory)";
        out << ", so model_run is not reentrant\n"
            << " * compile with -ffp-contract=off for bit-reproducible results\n */\n\n";
        if (opt_.embed_runtime) {
            // only the parts this model calls, in their dependency order
            for (const RuntimePart& part : runtime_parts())
                if (parts_.count(part.name))
                    out << "/* ---- runtime/null/" << part.name << ".h, embedded ---- */\n" << part.text << "\n";
            out << "/* ---- end of runtime ---- */\n\n";
        } else {
            out << "#include \"null_runtime.h\"\n\n";
        }
        out << decls_.str();
        if (!buffers_.empty()) {
            out << "\n/* intermediate buffers: one arena, shared by buffers never alive at the same time */\n"
                << "static float null_arena[" << arena_ << "];\n";
            for (const Buffer& b : buffers_)
                out << "static float* const " << b.id << " = null_arena + " << b.offset << ";   /* "
                    << comment_text(b.tensor) << ": " << type_str(info(b.tensor)) << ", " << when(b.first)
                    << " to " << when(b.last) << " */\n";
        }
        out << "\nvoid " << opt_.function_name << "(";
        for (size_t i = 0; i < params_.size(); ++i) out << (i ? ", " : "") << params_[i];
        out << ") {\n    int64_t i;\n" << body_.str() << "    /* outputs */\n" << tail.str() << "}\n";

        // the same entry point in array form, with the sizes, so a generic
        // driver or a language binding can call any model without knowing
        // its signature
        const std::string& f = opt_.function_name;
        const size_t n_in = g_.inputs.size(), n_out = g_.outputs.size();
        auto sizes = [&](const std::vector<std::string>& names) {
            std::string s;
            for (size_t i = 0; i < names.size(); ++i)
                s += (i ? ", " : "") + std::to_string(info(names[i]).num_elements());
            return s;
        };
        out << "\n/* array form, for generic drivers and bindings */\n"
            << "const int " << f << "_num_inputs = " << n_in << ";\n"
            << "const int " << f << "_num_outputs = " << n_out << ";\n"
            << "const int64_t " << f << "_input_sizes[" << (n_in ? n_in : 1) << "] = {"
            << (n_in ? sizes(g_.inputs) : "0") << "};\n"
            << "const int64_t " << f << "_output_sizes[" << n_out << "] = {" << sizes(g_.outputs) << "};\n"
            << "void " << f << "_v(const float* const* inputs, float* const* outputs) {\n"
            << "    (void)inputs;   /* a model may have no inputs */\n    " << f << "(";
        for (size_t i = 0; i < n_in; ++i) out << (i ? ", " : "") << "inputs[" << i << "]";
        for (size_t i = 0; i < n_out; ++i) out << (n_in + i ? ", " : "") << "outputs[" << i << "]";
        out << ");\n}\n";
        return out.str();
    }
};

} // namespace

// exact C literal for a float: hex float built from the bits, so every value
// round-trips and no printf %a support is needed
std::string c_float_literal(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const bool neg = bits >> 31;
    const uint32_t exp = (bits >> 23) & 0xFF, mant = bits & 0x7FFFFF;
    if (exp == 0xFF) return mant ? "NAN" : (neg ? "-INFINITY" : "INFINITY");
    char buf[40];
    if (exp == 0 && mant == 0) {
        std::snprintf(buf, sizeof buf, "%s0x0p+0f", neg ? "-" : "");
    } else if (exp == 0) {   // subnormal: 0.mant * 2^-126
        std::snprintf(buf, sizeof buf, "%s0x0.%06lxp-126f", neg ? "-" : "",
                      static_cast<unsigned long>(mant << 1));
    } else {                 // normal: 1.mant * 2^(exp - 127)
        std::snprintf(buf, sizeof buf, "%s0x1.%06lxp%+df", neg ? "-" : "",
                      static_cast<unsigned long>(mant << 1), static_cast<int>(exp) - 127);
    }
    return buf;
}

std::string emit_c(const Graph& g, const EmitOptions& options) {
    return Emitter(g, options).run();
}

} // namespace null
