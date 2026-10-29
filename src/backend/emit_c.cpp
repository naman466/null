#include "backend/emit_c.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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

std::string int_list(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

class Emitter {
public:
    Emitter(const Graph& g, const EmitOptions& o) : g_(g), opt_(o) {}

    std::string run() {
        check_static();
        declare_inputs();
        for (const Node* n : g_.sorted_nodes()) emit_node(*n);
        return assemble();
    }

private:
    const Graph& g_;
    const EmitOptions& opt_;
    std::map<std::string, std::string> expr_;   // tensor name -> C expression
    std::set<std::string> idents_;              // every C identifier used
    std::ostringstream decls_, body_;
    std::vector<std::string> params_;
    uint64_t buffer_bytes_ = 0, weight_bytes_ = 0;
    int temp_ = 0;

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
        if (auto it = expr_.find(name); it != expr_.end()) return it->second;
        if (auto it = g_.initializers.find(name); it != g_.initializers.end())
            return emit_constant(n, name, it->second);
        fail(where(n) + ": input '" + name + "' has no value");
    }

    std::string emit_constant(const Node& n, const std::string& name, const Tensor& t) {
        if (t.dtype != DType::FLOAT32)
            fail(where(n) + ": input '" + name + "' is a " + dtype_name(t.dtype) +
                 " constant, only float32 data can be used here");
        const int64_t count = t.num_elements();
        std::string id = ident("w_", name);
        decls_ << "/* " << name << ": " << type_str(t) << " */\n"
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
        decls_ << "static float " << id << "[" << (count ? count : 1) << "];   /* "
               << name << ": " << type_str(t) << " */\n";
        buffer_bytes_ += static_cast<uint64_t>(count) * 4;
        expr_[name] = id;
        return id;
    }

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
        body_ << "    /* " << n.name << ": " << optype_to_str(n.op) << " */\n";
        switch (n.op) {
            case OpType::Relu: {
                std::string x = read(n, 0), y = write(n, 0);
                body_ << "    null_relu_f32(" << x << ", " << y << ", " << info(n.outputs[0]).num_elements() << ");\n";
                break;
            }
            case OpType::Add: emit_binary(n, "add"); break;
            case OpType::Sub: emit_binary(n, "sub"); break;
            case OpType::Mul: emit_binary(n, "mul"); break;
            case OpType::Div: emit_binary(n, "div"); break;
            case OpType::Gemm:    emit_gemm(n); break;
            case OpType::MatMul:  emit_matmul(n); break;
            case OpType::Softmax: emit_softmax(n); break;
            case OpType::Reshape:
            case OpType::Flatten:
            case OpType::Identity:
            case OpType::Dropout:
                emit_alias(n);
                break;
            case OpType::Constant:
                emit_constant_node(n);
                break;
            default:
                fail(where(n) + ": no code generation for this operator yet");
        }
    }

    // same bytes, new shape: the output is another name for the input
    void emit_alias(const Node& n) {
        expr_[n.outputs.at(0)] = read(n, 0);
        body_ << "    /* " << n.outputs[0] << " is " << expr_[n.outputs[0]] << ", no data moves */\n";
        if (n.op == OpType::Dropout && n.outputs.size() > 1 && !n.outputs[1].empty())
            for (const Node& other : g_.nodes)
                for (auto& in : other.inputs)
                    if (in == n.outputs[1]) fail(where(n) + ": the dropout mask output is not supported");
    }

    // float constants become weights; others (shape tensors) are only used at compile time
    void emit_constant_node(const Node& n) {
        if (!n.has_attr("value")) fail(where(n) + ": only Constant with a 'value' tensor is supported");
        const Tensor& t = n.get_attr("value").as_tensor();
        if (t.dtype == DType::FLOAT32) emit_constant(n, n.outputs.at(0), t);
        body_ << "    /* " << n.outputs[0] << " is a compile-time constant */\n";
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
              << ", " << c_rs << ", " << c_cs << ");\n";
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
            std::string id = ident("out_", name);
            params_.push_back("float* " + id);
            tail << "    for (i = 0; i < " << t.num_elements() << "; ++i) " << id << "[i] = "
                 << expr_[name] << "[i];\n";
        }

        std::ostringstream out;
        out << "/*\n * generated by null from graph '" << g_.name << "', do not edit\n *\n";
        for (auto& name : g_.inputs) out << " * input  " << name << ": " << type_str(info(name)) << "\n";
        for (auto& name : g_.outputs) out << " * output " << name << ": " << type_str(info(name)) << "\n";
        out << " *\n * weights " << bytes_str(weight_bytes_) << ", intermediate buffers "
            << bytes_str(buffer_bytes_) << " (static, so model_run is not reentrant)\n"
            << " * compile with -ffp-contract=off for bit-reproducible results\n */\n\n";
        if (opt_.embed_runtime) {
            out << "/* ---- runtime/null_runtime.h, embedded ---- */\n" << runtime_source()
                << "/* ---- end of runtime ---- */\n\n";
        } else {
            out << "#include \"null_runtime.h\"\n\n";
        }
        out << decls_.str() << "\nvoid " << opt_.function_name << "(";
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
