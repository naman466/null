#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "backend/emit_c.h"
#include "core/graph.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef NULL_RUNTIME_DIR
#error "NULL_RUNTIME_DIR must point at the runtime/ folder"
#endif

using namespace null;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

static std::string emit_error(const Graph& g) {
    try {
        emit_c(g);
    } catch (const EmitError& e) {
        return e.what();
    }
    throw std::runtime_error("expected EmitError");
}

static void add_weights(Graph& g, const std::string& name, const std::vector<int64_t>& shape, float value) {
    Tensor t(name, shape, DType::FLOAT32);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    t.set_float_data(std::vector<float>(static_cast<size_t>(n), value));
    t.shape = shape;
    g.add_initializer(t);
}

// x[1, 4] -> Flatten -> Gemm(w[3, 4]^T, b[3]) -> Relu -> Softmax -> y[1, 3]
static Graph small_mlp() {
    Graph g("small_mlp");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {1, 4}, DType::FLOAT32));
    add_weights(g, "w", {3, 4}, 0.5f);
    add_weights(g, "b", {3}, 1.0f);
    g.add_node(Node("flat", OpType::Flatten, {"x"}, {"xf"}));
    Node fc("fc", OpType::Gemm, {"xf", "w", "b"}, {"h"});
    fc.add_attr("transB", int64_t{1});
    g.add_node(fc);
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"r"}));
    g.add_node(Node("softmax", OpType::Softmax, {"r"}, {"y"}));
    infer_shapes(g);
    return g;
}

// TESTS - FLOAT LITERALS

TEST(emit_float_literals) {
    CHECK_EQ(c_float_literal(1.5f), "0x1.800000p+0f");
    CHECK_EQ(c_float_literal(-2.0f), "-0x1.000000p+1f");
    CHECK_EQ(c_float_literal(0.0f), "0x0p+0f");
    CHECK_EQ(c_float_literal(-0.0f), "-0x0p+0f");
    CHECK_EQ(c_float_literal(1e-45f), "0x0.000002p-126f");            // smallest subnormal
    CHECK_EQ(c_float_literal(3.4028235e38f), "0x1.fffffep+127f");     // largest float
    CHECK_EQ(c_float_literal(std::numeric_limits<float>::infinity()), "INFINITY");
    CHECK_EQ(c_float_literal(-std::numeric_limits<float>::infinity()), "-INFINITY");
    CHECK_EQ(c_float_literal(std::nanf("")), "NAN");
}

TEST(emit_float_literals_round_trip) {
    // every finite float, written as a literal and parsed back, keeps all its bits
    std::mt19937 rng(7);
    for (int i = 0; i < 100000; ++i) {
        uint32_t bits = static_cast<uint32_t>(rng());
        float v;
        std::memcpy(&v, &bits, 4);
        if (!std::isfinite(v)) continue;
        std::string lit = c_float_literal(v);
        float back = std::strtof(lit.c_str(), nullptr);   // strtof reads hex floats, stops at the 'f'
        uint32_t back_bits;
        std::memcpy(&back_bits, &back, 4);
        if (back_bits != bits) throw std::runtime_error("literal " + lit + " does not round-trip");
    }
}

// TESTS - RUNTIME EMBEDDING

// a file's contents, without the \r a text-mode checkout may add (the
// embedded copies never have them)
static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text;
    for (char c : ss.str()) if (c != '\r') text += c;
    return text;
}

TEST(emit_runtime_parts_are_the_files) {
    // the copies cmake embeds must be byte-identical to the files (semicolons
    // and quotes survive the trip through cmake)
    const auto& parts = runtime_parts();
    CHECK(!parts.empty());
    CHECK_EQ(std::string(parts[0].name), std::string("core"));
    for (const RuntimePart& part : parts) {
        std::string file = read_text(std::string(NULL_RUNTIME_DIR) + "/null/" + part.name + ".h");
        CHECK(!file.empty());
        CHECK_EQ(std::string(part.text), file);
    }

    // and in the umbrella header's order, which is their dependency order
    std::string umbrella = read_text(std::string(NULL_RUNTIME_DIR) + "/null_runtime.h");
    std::vector<std::string> included;
    for (size_t at = umbrella.find("#include \"null/"); at != std::string::npos;
         at = umbrella.find("#include \"null/", at + 1)) {
        size_t start = at + std::strlen("#include \"null/");
        included.push_back(umbrella.substr(start, umbrella.find(".h\"", start) - start));
    }
    std::vector<std::string> names;
    for (const RuntimePart& part : parts) names.push_back(part.name);
    CHECK_EQ(included, names);
}

TEST(emit_embeds_or_includes_runtime) {
    Graph g = small_mlp();
    std::string embedded = emit_c(g);
    CHECK(contains(embedded, "static inline void null_gemm_f32("));
    CHECK(!contains(embedded, "#include \"null_runtime.h\""));

    EmitOptions opt;
    opt.embed_runtime = false;
    std::string included = emit_c(g, opt);
    CHECK(contains(included, "#include \"null_runtime.h\""));
    CHECK(!contains(included, "static inline void null_gemm_f32("));
}

TEST(emit_embeds_only_the_parts_used) {
    // the mlp calls gemm, relu and softmax: core first, then those parts in order
    std::string mlp = emit_c(small_mlp());
    size_t core = mlp.find("/* ---- runtime/null/core.h, embedded ---- */");
    size_t elementwise = mlp.find("/* ---- runtime/null/elementwise.h, embedded ---- */");
    size_t matmul = mlp.find("/* ---- runtime/null/matmul.h, embedded ---- */");
    size_t softmax = mlp.find("/* ---- runtime/null/softmax.h, embedded ---- */");
    CHECK(core != std::string::npos && elementwise != std::string::npos);
    CHECK(matmul != std::string::npos && softmax != std::string::npos);
    CHECK(core < elementwise && elementwise < matmul && matmul < softmax);
    CHECK(!contains(mlp, "runtime/null/conv.h"));
    CHECK(!contains(mlp, "null_conv2d_params"));
    CHECK(!contains(mlp, "runtime/null/pool.h"));
    CHECK(!contains(mlp, "runtime/null/reduce.h"));

    // a lone relu needs only the core and the elementwise part
    Graph g("relu");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    g.add_node(Node("relu", OpType::Relu, {"x"}, {"y"}));
    infer_shapes(g);
    std::string relu = emit_c(g);
    CHECK(contains(relu, "runtime/null/elementwise.h, embedded"));
    CHECK(!contains(relu, "runtime/null/matmul.h"));
    CHECK(!contains(relu, "runtime/null/softmax.h"));
    CHECK(!contains(relu, "null_softmax_f32"));
}

TEST(emit_small_mlp) {
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(small_mlp(), opt);
    CHECK(contains(c, "void model_run(const float* in_x, float* out_y) {"));
    CHECK(contains(c, "static const float w_w[12] = {"));
    CHECK(contains(c, "static const float w_b[3] = {"));
    // Flatten moves no data: the gemm reads the input directly
    CHECK(contains(c, "/* xf is in_x, no data moves */"));
    CHECK(contains(c, "null_gemm_f32(in_x, w_w, w_b, t_h, 1, 3, 4, 0, 1, 0x1.000000p+0f, 0x1.000000p+0f, 0, 1);"));
    CHECK(contains(c, "null_relu_f32(t_h, t_r, 3);"));
    CHECK(contains(c, "null_softmax_f32(t_r, t_y, 1, 3, 1);"));
    CHECK(contains(c, "for (i = 0; i < 3; ++i) out_y[i] = t_y[i];"));
    // weights are exact hex literals
    CHECK(contains(c, "0x1.000000p-1f,"));
    CHECK(!contains(c, "t_xf"));   // no buffer for the alias
}

TEST(emit_broadcasting_add) {
    Graph g("bias");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3}, DType::FLOAT32));
    add_weights(g, "b", {3}, 0.25f);
    g.add_node(Node("add", OpType::Add, {"x", "b"}, {"y"}));
    g.add_node(Node("same", OpType::Mul, {"y", "y"}, {"z"}));
    g.outputs = {"z"};
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    // different shapes: strides computed at compile time, 0 along the broadcast dim
    CHECK(contains(c, "null_add_bcast_f32(in_x, w_b, t_y, 2,"));
    CHECK(contains(c, "_shape[2] = {2, 3};"));
    CHECK(contains(c, "_a_strides[2] = {3, 1};"));
    CHECK(contains(c, "_b_strides[2] = {0, 1};"));
    // same shapes: the flat kernel
    CHECK(contains(c, "null_mul_f32(t_y, t_y, t_z, 6);"));
}

TEST(emit_conv) {
    // x[1, 2, 5, 5] -> Conv(w[4, 1, 3, 3], b[4], group 2, stride 2, pads 1 0 1 0) -> Relu -> y
    Graph g("conv");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {1, 2, 5, 5}, DType::FLOAT32));
    add_weights(g, "w", {4, 1, 3, 3}, 0.5f);
    add_weights(g, "b", {4}, 1.0f);
    Node conv("conv", OpType::Conv, {"x", "w", "b"}, {"h"});
    conv.add_attr("group", int64_t{2});
    conv.add_attr("strides", std::vector<int64_t>{2, 2});
    conv.add_attr("pads", std::vector<int64_t>{1, 0, 1, 0});
    g.add_node(conv);
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"y"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    // n c h w, m kh kw group, strides, dilations, pad top left, out h w
    CHECK(contains(c, "_conv = {1, 2, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 0, 3, 2};"));
    CHECK(contains(c, "null_conv2d_f32(in_x, w_w, w_b, t_h, &"));
    CHECK(contains(c, "null_relu_f32(t_h, t_y, 24);"));

    // no bias: the kernel gets NULL
    Graph nb("conv_nb");
    nb.inputs = {"x"};
    nb.outputs = {"y"};
    nb.add_value_info(Tensor("x", {1, 1, 3, 3}, DType::FLOAT32));
    add_weights(nb, "w", {1, 1, 2, 2}, 1.0f);
    nb.add_node(Node("conv", OpType::Conv, {"x", "w"}, {"y"}));
    infer_shapes(nb);
    CHECK(contains(emit_c(nb, opt), "null_conv2d_f32(in_x, w_w, NULL, t_y, &"));
}

TEST(emit_maxpool) {
    // x[1, 3, 6, 6] -> MaxPool(3x3, stride 2, pads 1, ceil_mode) -> y[1, 3, 4, 4]
    Graph g("pool");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {1, 3, 6, 6}, DType::FLOAT32));
    Node pool("pool", OpType::MaxPool, {"x"}, {"y"});
    pool.add_attr("kernel_shape", std::vector<int64_t>{3, 3});
    pool.add_attr("strides", std::vector<int64_t>{2, 2});
    pool.add_attr("pads", std::vector<int64_t>{1, 1, 1, 1});
    pool.add_attr("ceil_mode", int64_t{1});
    g.add_node(pool);
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    // n c h w, kh kw, strides, dilations, pad top left, out h w
    CHECK(contains(c, "_pool = {1, 3, 6, 6, 3, 3, 2, 2, 1, 1, 1, 1, 4, 4};"));
    CHECK(contains(c, "null_maxpool2d_f32(in_x, t_y, &"));
}

// x -> op -> y, with optional constant axes for ReduceMean
static std::string emit_reduce(OpType op, const std::vector<int64_t>& x, const std::vector<int64_t>& axes, int64_t keepdims = 1) {
    Graph g("reduce");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", x, DType::FLOAT32));
    Node n("mean", op, {"x"}, {"y"});
    if (op == OpType::ReduceMean) {
        Tensor t("axes", {static_cast<int64_t>(axes.size())}, DType::INT64);
        t.set_int64_data(axes);
        t.shape = {static_cast<int64_t>(axes.size())};
        g.add_initializer(t);
        n.inputs.push_back("axes");
        n.add_attr("keepdims", keepdims);
    }
    g.add_node(n);
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    return emit_c(g, opt);
}

TEST(emit_reduce_mean) {
    // the batch of 1 is dropped, C is kept, H and W merge into one reduced dim
    std::string gap = emit_reduce(OpType::GlobalAveragePool, {1, 16, 8, 8}, {});
    CHECK(contains(gap, "static const null_reduce_params k0_mean_reduce = {2, {16, 64}, {0, 1}};"));
    CHECK(contains(gap, "null_reduce_mean_f32(in_x, t_y, &k0_mean_reduce);"));
    CHECK(contains(gap, "float32[1, 16, 1, 1]"));

    // the same reduction from pytorch's dynamo exporter, keepdims or not
    std::string rm = emit_reduce(OpType::ReduceMean, {1, 16, 8, 8}, {-1, -2}, 0);
    CHECK(contains(rm, "= {2, {16, 64}, {0, 1}};"));
    CHECK(contains(rm, "float32[1, 16]"));

    // alternating dims stay apart; size-1 dims vanish whether reduced or not
    CHECK(contains(emit_reduce(OpType::ReduceMean, {2, 1, 3, 4, 5}, {0, 1, 3}), "= {4, {2, 3, 4, 5}, {1, 0, 1, 0}};"));
    // everything reduced is one dim; nothing reduced is a copy
    CHECK(contains(emit_reduce(OpType::ReduceMean, {2, 3, 4}, {0, 1, 2}), "= {1, {24}, {1}};"));
    CHECK(contains(emit_reduce(OpType::ReduceMean, {1, 1}, {0}), "= {0, {0}, {0}};"));
}

TEST(emit_batched_matmul) {
    Graph g("bmm");
    g.inputs = {"a"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("a", {4, 2, 3}, DType::FLOAT32));
    add_weights(g, "w", {3, 5}, 1.0f);
    g.add_node(Node("mm", OpType::MatMul, {"a", "w"}, {"y"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    // batch [4] read from a with stride M*K = 6, the shared weights with stride 0
    CHECK(contains(c, "null_matmul_f32(in_a, w_w, t_y, 2, 5, 3, 1,"));
    CHECK(contains(c, "_a_strides[1] = {6};"));
    CHECK(contains(c, "_b_strides[1] = {0};"));
}

TEST(emit_unique_identifiers) {
    // names that sanitise to the same C identifier get distinct ones
    Graph g("names");
    g.inputs = {"a/b"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("a/b", {2}, DType::FLOAT32));
    g.add_node(Node("r1", OpType::Relu, {"a/b"}, {"a.b"}));
    g.add_node(Node("r2", OpType::Relu, {"a.b"}, {"a-b"}));
    g.add_node(Node("r3", OpType::Relu, {"a-b"}, {"y"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "const float* in_a_b"));
    CHECK(contains(c, "static float t_a_b[2];"));
    CHECK(contains(c, "static float t_a_b_2[2];"));
    CHECK(contains(c, "null_relu_f32(t_a_b, t_a_b_2, 2);"));
}

TEST(emit_reshape_to_output_and_constants) {
    // a reshaped input as the graph output is copied out; an int64 shape
    // constant is used at compile time and never emitted
    Graph g("reshape");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3}, DType::FLOAT32));
    Tensor shape("shape", {1}, DType::INT64);
    shape.set_int64_data({6});
    shape.shape = {1};
    g.add_initializer(shape);
    g.add_node(Node("reshape", OpType::Reshape, {"x", "shape"}, {"y"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "for (i = 0; i < 6; ++i) out_y[i] = in_x[i];"));
    CHECK(!contains(c, "w_shape"));
}

// TESTS - ERRORS

TEST(emit_errors) {
    Graph dynamic("dyn");
    dynamic.inputs = {"x"};
    dynamic.outputs = {"y"};
    dynamic.add_value_info(Tensor("x", {-1, 4}, DType::FLOAT32));
    dynamic.add_node(Node("r", OpType::Relu, {"x"}, {"y"}));
    infer_shapes(dynamic);
    CHECK(contains(emit_error(dynamic), "tensor 'x' has dynamic shape float32[?, 4]"));

    Graph unsupported("clip");
    unsupported.inputs = {"x"};
    unsupported.outputs = {"y"};
    unsupported.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    unsupported.add_node(Node("c", OpType::Clip, {"x"}, {"y"}));
    infer_shapes(unsupported);
    CHECK(contains(emit_error(unsupported), "node 'c' (Clip): no code generation for this operator yet"));

    Graph ints("ints");
    ints.inputs = {"x"};
    ints.outputs = {"y"};
    ints.add_value_info(Tensor("x", {4}, DType::INT64));
    ints.add_node(Node("r", OpType::Relu, {"x"}, {"y"}));
    infer_shapes(ints);
    CHECK(contains(emit_error(ints), "input 'x' is int64, only float32 inputs are supported"));

    Graph uninferred("none");
    uninferred.inputs = {"x"};
    uninferred.outputs = {"y"};
    uninferred.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    uninferred.add_node(Node("r", OpType::Relu, {"x"}, {"y"}));
    CHECK(contains(emit_error(uninferred), "has no shape, run shape inference first"));
}

TEST(emit_array_entry_point) {
    // a generic driver can call any model through model_run_v and its sizes
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(small_mlp(), opt);
    CHECK(contains(c, "const int model_run_num_inputs = 1;"));
    CHECK(contains(c, "const int model_run_num_outputs = 1;"));
    CHECK(contains(c, "const int64_t model_run_input_sizes[1] = {4};"));
    CHECK(contains(c, "const int64_t model_run_output_sizes[1] = {3};"));
    CHECK(contains(c, "void model_run_v(const float* const* inputs, float* const* outputs) {"));
    CHECK(contains(c, "model_run(inputs[0], outputs[0]);"));

    // the names follow the chosen function name
    opt.function_name = "digits";
    std::string named = emit_c(small_mlp(), opt);
    CHECK(contains(named, "void digits(const float* in_x, float* out_y) {"));
    CHECK(contains(named, "void digits_v("));
    CHECK(contains(named, "const int digits_num_inputs = 1;"));
}
