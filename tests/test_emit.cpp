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
    CHECK(contains(c, "null_gemm_f32(in_x, w_w, w_b, t_h, 1, 3, 4, 0, 1, 0x1.000000p+0f, 0x1.000000p+0f, 0, 1, 0);"));
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
    // n c h w, m kh kw group, strides, dilations, pad top left, out h w, fused relu
    CHECK(contains(c, "_conv = {1, 2, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 0, 3, 2, 0};"));
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

TEST(emit_shape_chain_is_compile_time_only) {
    // x[1, 12] -> Shape -> Gather(0) -> Unsqueeze(0) -> Concat with [-1] -> Reshape -> Relu,
    // then Unsqueeze on the float result
    Graph g("view");
    g.inputs = {"x"};
    g.outputs = {"z"};
    g.add_value_info(Tensor("x", {1, 12}, DType::FLOAT32));
    auto ints = [&](const std::string& name, const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
        Tensor t(name, shape, DType::INT64);
        t.set_int64_data(v);
        t.shape = shape;
        g.add_initializer(t);
    };
    ints("zero", {0}, {});
    ints("axes", {0}, {1});
    ints("rest", {-1}, {1});
    g.add_node(Node("shape", OpType::Shape, {"x"}, {"s"}));
    g.add_node(Node("gather", OpType::Gather, {"s", "zero"}, {"batch"}));
    g.add_node(Node("unsq", OpType::Unsqueeze, {"batch", "axes"}, {"batch1"}));
    Node cat("concat", OpType::Concat, {"batch1", "rest"}, {"target"});
    cat.add_attr("axis", int64_t{0});
    g.add_node(cat);
    g.add_node(Node("reshape", OpType::Reshape, {"x", "target"}, {"flat"}));
    g.add_node(Node("relu", OpType::Relu, {"flat"}, {"y"}));
    g.add_node(Node("lift", OpType::Unsqueeze, {"y", "axes"}, {"z"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "/* s = [1, 12], computed at compile time */"));
    CHECK(contains(c, "/* batch = [1], computed at compile time */"));
    CHECK(contains(c, "/* target = [1, -1], computed at compile time */"));
    CHECK(contains(c, "/* flat is in_x, no data moves */"));
    CHECK(contains(c, "null_relu_f32(in_x, t_y, 12);"));
    CHECK(contains(c, "/* z is t_y, no data moves */"));            // a float Unsqueeze is an alias
    CHECK(contains(c, " * output z: float32[1, 1, 12]"));
}

// x [shape] -> one node -> y, with an optional int64 initializer
static std::string emit_one(const std::vector<int64_t>& shape, Node n, const std::vector<int64_t>& ints = {},
                            const std::vector<int64_t>& ints_shape = {}) {
    Graph g("one");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", shape, DType::FLOAT32));
    if (n.inputs.size() > 1) {
        Tensor t(n.inputs[1], ints_shape, DType::INT64);
        t.set_int64_data(ints);
        t.shape = ints_shape;
        g.add_initializer(t);
    }
    g.add_node(n);
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    return emit_c(g, opt);
}

TEST(emit_gather) {
    // one index, nothing before the axis: a pointer into x
    std::string row = emit_one({4, 3}, Node("g", OpType::Gather, {"x", "idx"}, {"y"}), {-1}, {});
    CHECK(contains(row, "/* y is (in_x + 9), no data moves */"));
    CHECK(!contains(row, "null_gather_f32("));
    std::string first = emit_one({4, 3}, Node("g", OpType::Gather, {"x", "idx"}, {"y"}), {0}, {1});
    CHECK(contains(first, "/* y is in_x, no data moves */"));

    // several indices, or an axis with dims before it: the kernel, indices made non-negative
    Node cols("g", OpType::Gather, {"x", "idx"}, {"y"});
    cols.add_attr("axis", int64_t{1});
    std::string c = emit_one({4, 3}, cols, {2, -3}, {2});
    CHECK(contains(c, "static const int64_t k0_g_indices[2] = {2, 0};"));
    CHECK(contains(c, "null_gather_f32(in_x, t_y, 4, 3, 1, k0_g_indices, 2);"));
}

TEST(emit_transpose) {
    // [1, 8, 32] with perm [1, 0, 2]: only a size-1 dim moves, so no data does
    Node swap("t", OpType::Transpose, {"x"}, {"y"});
    swap.add_attr("perm", std::vector<int64_t>{1, 0, 2});
    CHECK(contains(emit_one({1, 8, 32}, swap), "/* y is in_x, no data moves */"));

    // a real transpose: size-1 dims dropped, the strides of the input dims in output order
    Node t("t", OpType::Transpose, {"x"}, {"y"});
    t.add_attr("perm", std::vector<int64_t>{2, 1, 0, 3});
    std::string c = emit_one({2, 1, 3, 4}, t);
    CHECK(contains(c, "static const int64_t k0_t_shape[3] = {3, 2, 4};"));
    CHECK(contains(c, "static const int64_t k1_t_x_strides[3] = {4, 12, 1};"));
    CHECK(contains(c, "null_transpose_f32(in_x, t_y, 3, k0_t_shape, k1_t_x_strides);"));
    // no perm: the dims reversed
    std::string rev = emit_one({2, 3}, Node("t", OpType::Transpose, {"x"}, {"y"}));
    CHECK(contains(rev, " * output y: float32[3, 2]"));
    CHECK(contains(rev, "null_transpose_f32(in_x, t_y, 2,"));
}

TEST(emit_layernorm) {
    Graph g("ln");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3, 4}, DType::FLOAT32));
    Tensor scale("scale", {4}, DType::FLOAT32), bias("bias", {4}, DType::FLOAT32);
    scale.set_float_data({1, 1, 1, 1});
    bias.set_float_data({0, 0, 0, 0});
    g.add_initializer(scale);
    g.add_initializer(bias);
    Node n("ln", OpType::LayerNormalization, {"x", "scale", "bias"}, {"y"});
    n.add_attr("epsilon", 1e-3f);
    g.add_node(n);
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    CHECK(contains(emit_c(g, opt), "null_layernorm_f32(in_x, w_scale, w_bias, t_y, 6, 4, 0x1.0624dep-10f);"));

    // a scale of [1] would broadcast over the row, which the kernel does not do yet
    Graph narrow("ln");
    narrow.inputs = {"x"};
    narrow.outputs = {"y"};
    narrow.add_value_info(Tensor("x", {2, 4}, DType::FLOAT32));
    Tensor one("scale", {1}, DType::FLOAT32);
    one.set_float_data({2});
    narrow.add_initializer(one);
    narrow.add_node(Node("ln", OpType::LayerNormalization, {"x", "scale"}, {"y"}));
    infer_shapes(narrow);
    CHECK(contains(emit_error(narrow), "a scale or bias that broadcasts within the normalized dims is not supported yet"));
}

// splits generated C into its comments and the code between them, and checks
// that each comment is closed by the first */ after it and holds only
// printable ASCII (and line breaks) with no /* and no ?? inside
static bool comments_are_clean(const std::string& c, std::string& code) {
    size_t at = 0;
    while (true) {
        size_t open = c.find("/*", at);
        code += c.substr(at, open == std::string::npos ? std::string::npos : open - at);
        if (open == std::string::npos) return true;
        size_t close = c.find("*/", open + 2);
        if (close == std::string::npos) return false;
        std::string body = c.substr(open + 2, close - open - 2);
        if (body.find("/*") != std::string::npos || body.find("??") != std::string::npos) return false;
        for (char ch : body) {
            const unsigned char u = static_cast<unsigned char>(ch);
            if (ch != '\n' && (u < 0x20 || u >= 0x7f)) return false;
        }
        at = close + 2;
    }
}

TEST(emit_names_cannot_escape_comments) {
    // every name from the model lands in some comment: graph, input, output,
    // weight, buffer, node and alias. each tries to end its comment and inject
    // code, or to break the build another way
    Graph g("g*/int*from_graph;/*");
    const std::string x = "x*/int*from_input;/*", w = "w/**/?\?/", h = "h*/\nint*from_buffer;\t/*";
    const std::string y = "y\xc3\xa9*/int*from_output;/*", flat = "f*//*int*from_alias;";
    g.inputs = {x};
    g.outputs = {y};
    g.add_value_info(Tensor(x, {4}, DType::FLOAT32));
    Tensor wt(w, {4}, DType::FLOAT32);
    wt.set_float_data({1, 2, 3, 4});
    g.add_initializer(wt);
    g.add_node(Node("*/int*from_node;/*", OpType::Add, {x, w}, {h}));
    g.add_node(Node("?\?/", OpType::Flatten, {h}, {flat}));
    g.add_node(Node("r*", OpType::Relu, {flat}, {y}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt), code;
    CHECK(comments_are_clean(c, code));
    CHECK(!contains(code, "*from_"));                    // identifiers keep the letters, never the code
    CHECK(contains(c, "graph 'g* /int*from_graph;/ *'"));
    CHECK(contains(c, "/* * /int*from_node;/ *: Add */"));
    CHECK(contains(c, "/* w/ ** /? ?/: float32[4] */"));
    CHECK(contains(c, "h* /?int*from_buffer;?/ *"));    // line break and tab become '?'
    CHECK(contains(c, " * output y? ?* /int*from_output;/ *: float32[4, 1]"));   // so do non-ASCII bytes
    CHECK(contains(c, "/* f* // *int*from_alias; is "));
}

TEST(emit_skips_what_no_output_needs) {
    // unoptimised graphs can hold a Constant nothing reads, or an Identity of a
    // weight that goes nowhere: their weights would be arrays no code reads,
    // which -Werror rejects, so no code is written for them at all
    Graph g("unused");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    Tensor w("w", {4}, DType::FLOAT32);
    w.set_float_data({1, 2, 3, 4});
    g.add_initializer(w);
    Node k("k", OpType::Constant, {}, {"c"});
    Tensor v("v", {2}, DType::FLOAT32);
    v.set_float_data({5, 6});
    k.add_attr("value", v);
    g.add_node(k);
    g.add_node(Node("copy", OpType::Identity, {"w"}, {"w2"}));
    g.add_node(Node("relu", OpType::Relu, {"x"}, {"y"}));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "/* k: Constant, no output depends on it */"));
    CHECK(contains(c, "/* copy: Identity, no output depends on it */"));
    CHECK(!contains(c, "static const float"));
    CHECK(contains(c, "null_relu_f32(in_x, t_y, 4);"));
}

// x[4] -> Clip(x, min, max) -> y; a bound is an initializer when given a
// value, "" when omitted
static Graph clip_graph(const std::string& min, const std::string& max) {
    Graph g("clip");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    for (const auto& [name, value] : {std::pair<std::string, float>{"lo", 0.0f}, {"hi", 6.0f}}) {
        Tensor t(name, {}, DType::FLOAT32);
        t.set_float_data({value});
        t.shape = {};
        g.add_initializer(t);
    }
    g.add_node(Node("clip", OpType::Clip, {"x", min, max}, {"y"}));
    infer_shapes(g);
    return g;
}

TEST(emit_clip) {
    EmitOptions opt;
    opt.embed_runtime = false;
    CHECK(contains(emit_c(clip_graph("lo", "hi"), opt), "null_clip_f32(in_x, t_y, 4, 0x0p+0f, 0x1.800000p+2f);"));
    // an omitted bound is the largest finite float on that side, as onnx defines it
    CHECK(contains(emit_c(clip_graph("", "hi"), opt), "null_clip_f32(in_x, t_y, 4, -0x1.fffffep+127f, 0x1.800000p+2f);"));
    CHECK(contains(emit_c(clip_graph("lo", ""), opt), "null_clip_f32(in_x, t_y, 4, 0x0p+0f, 0x1.fffffep+127f);"));

    // a bound from a Constant node, as an unoptimised graph has it
    Graph k("clip");
    k.inputs = {"x"};
    k.outputs = {"y"};
    k.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    Node six("six", OpType::Constant, {}, {"hi"});
    Tensor v("v", {}, DType::FLOAT32);
    v.set_float_data({6.0f});
    v.shape = {};
    six.add_attr("value", v);
    k.add_node(six);
    k.add_node(Node("clip", OpType::Clip, {"x", "", "hi"}, {"y"}));
    infer_shapes(k);
    CHECK(contains(emit_c(k, opt), "null_clip_f32(in_x, t_y, 4, -0x1.fffffep+127f, 0x1.800000p+2f);"));
    // read only at compile time, the Constant leaves no array behind: an
    // unused static const array fails -Werror (MobileNetV2 unoptimised had 70)
    CHECK(!contains(emit_c(k, opt), "static const float"));
    // while a Constant a kernel reads becomes a weight, when it is first read
    k.outputs = {"z"};
    k.add_node(Node("add", OpType::Add, {"y", "hi"}, {"z"}));
    infer_shapes(k);
    std::string used = emit_c(k, opt);
    CHECK(contains(used, "static const float w_hi[1]"));
    CHECK(contains(used, "null_add_bcast_f32(t_y, w_hi, t_z,"));

    // a bound computed at run time, or NaN, is refused
    Graph run("clip");
    run.inputs = {"x", "m"};
    run.outputs = {"y"};
    run.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    run.add_value_info(Tensor("m", {}, DType::FLOAT32));
    run.add_node(Node("clip", OpType::Clip, {"x", "m"}, {"y"}));
    infer_shapes(run);
    CHECK(contains(emit_error(run), "node 'clip' (Clip): the min bound must be a float constant known at compile time"));
    Graph nan = clip_graph("lo", "hi");
    float q = NAN;
    std::memcpy(nan.initializers.at("hi").data.data(), &q, 4);
    CHECK(contains(emit_error(nan), "a NaN max bound is not supported"));
}

TEST(emit_sigmoid_tanh) {
    // x[2, 3] -> Sigmoid -> Tanh -> y: one elementwise call each, from the elementwise part
    Graph g("act");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3}, DType::FLOAT32));
    g.add_node(Node("sig", OpType::Sigmoid, {"x"}, {"s"}));
    g.add_node(Node("tanh", OpType::Tanh, {"s"}, {"y"}));
    infer_shapes(g);
    std::string c = emit_c(g);
    CHECK(contains(c, "null_sigmoid_f32(in_x, t_s, 6);"));
    CHECK(contains(c, "null_tanh_f32(t_s, t_y, 6);"));
    CHECK(contains(c, "runtime/null/elementwise.h, embedded"));
    CHECK(!contains(c, "runtime/null/matmul.h"));
}

TEST(emit_exp_log_sqrt_abs_neg) {
    // x[5] -> Exp -> Log -> Sqrt -> Abs -> Neg -> y: one elementwise call each
    Graph g("unary");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {5}, DType::FLOAT32));
    g.add_node(Node("exp", OpType::Exp, {"x"}, {"a"}));
    g.add_node(Node("log", OpType::Log, {"a"}, {"b"}));
    g.add_node(Node("sqrt", OpType::Sqrt, {"b"}, {"c"}));
    g.add_node(Node("abs", OpType::Abs, {"c"}, {"d"}));
    g.add_node(Node("neg", OpType::Neg, {"d"}, {"y"}));
    infer_shapes(g);
    std::string c = emit_c(g);
    CHECK(contains(c, "null_exp_f32(in_x, t_a, 5);"));
    CHECK(contains(c, "null_log_f32(t_a, t_b, 5);"));
    CHECK(contains(c, "null_sqrt_f32(t_b, t_c, 5);"));
    CHECK(contains(c, "null_abs_f32(t_c, t_d, 5);"));
    CHECK(contains(c, "null_neg_f32(t_d, t_y, 5);"));
    CHECK(contains(c, "runtime/null/elementwise.h, embedded"));
}

TEST(emit_fused_relu) {
    // a Gemm or Conv carrying the optimiser's kFusedRelu passes 1 to its kernel
    Graph g("fused");
    g.inputs = {"x", "img"};
    g.outputs = {"y", "z"};
    g.add_value_info(Tensor("x", {1, 4}, DType::FLOAT32));
    g.add_value_info(Tensor("img", {1, 1, 3, 3}, DType::FLOAT32));
    Tensor w("w", {2, 4}, DType::FLOAT32), k("k", {1, 1, 2, 2}, DType::FLOAT32);
    w.set_float_data(std::vector<float>(8, 1));
    k.set_float_data(std::vector<float>(4, 1));
    g.add_initializer(w);
    g.add_initializer(k);
    Node gemm("gemm", OpType::Gemm, {"x", "w"}, {"y"});
    gemm.add_attr("transB", int64_t{1});
    gemm.add_attr(kFusedRelu, int64_t{1});
    Node conv("conv", OpType::Conv, {"img", "k"}, {"z"});
    conv.add_attr(kFusedRelu, int64_t{1});
    g.add_node(gemm);
    g.add_node(conv);
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "null_gemm_f32(in_x, w_w, NULL, t_y, 1, 2, 4, 0, 1, 0x1.000000p+0f, 0x1.000000p+0f, 0, 0, 1);"));
    CHECK(contains(c, "_conv = {1, 1, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 2, 2, 1};"));
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
    CHECK(contains(c, "static float* const t_a_b = null_arena + 0;"));
    CHECK(contains(c, "static float* const t_a_b_2 = null_arena + 16;"));
    CHECK(contains(c, "null_relu_f32(t_a_b, t_a_b_2, 2);"));
}

// x[n] -> Relu -> a -> Relu -> b -> ... a chain of `links` relus ending in y,
// with extra nodes appended
static Graph relu_chain(int links, int64_t n, const std::vector<Node>& extra = {}) {
    Graph g("chain");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {n}, DType::FLOAT32));
    std::string prev = "x";
    for (int i = 0; i < links; ++i) {
        std::string out = i + 1 == links ? "y" : "t" + std::to_string(i);
        g.add_node(Node("r" + std::to_string(i), OpType::Relu, {prev}, {out}));
        prev = out;
    }
    for (const Node& e : extra) g.add_node(e);
    infer_shapes(g);
    return g;
}

TEST(emit_memory_plan) {
    // a chain of five: each buffer is alive from its step to the next, so two
    // ranges of 64 floats take turns; unplanned, five ranges side by side
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(relu_chain(5, 64), opt);
    CHECK(contains(c, "static float null_arena[128];"));
    CHECK(contains(c, "t_t0 = null_arena + 0;   /* t0: float32[64], step 1 to step 2 */"));
    CHECK(contains(c, "t_t1 = null_arena + 64;   /* t1: float32[64], step 2 to step 3 */"));
    CHECK(contains(c, "t_t2 = null_arena + 0;"));
    CHECK(contains(c, "t_y = null_arena + 0;   /* y: float32[64], step 5 to the end */"));
    CHECK(contains(c, "intermediate buffers 512 B in one static arena (1.2 KB if each buffer had its own memory)"));

    opt.plan_memory = false;
    std::string flat = emit_c(relu_chain(5, 64), opt);
    CHECK(contains(flat, "static float null_arena[320];"));
    CHECK(contains(flat, "t_y = null_arena + 256;"));
    CHECK(contains(flat, "intermediate buffers 1.2 KB in one static arena, so"));

    // sizes are rounded up to 16 floats, so every buffer starts on 64 bytes
    opt.plan_memory = true;
    CHECK(contains(emit_c(relu_chain(3, 5), opt), "t_t1 = null_arena + 16;"));
}

TEST(emit_memory_plan_follows_aliases) {
    // t0 is read again at the end through a reshape alias, so it stays alive
    // the whole time and nothing may take its place; y, read only by the
    // output copy, lives to the end
    Node flat("flat", OpType::Flatten, {"t0"}, {"t0_flat"});
    Node late("late", OpType::Add, {"t2", "t0_flat"}, {"z"});
    Graph g = relu_chain(4, 64, {flat, late});
    g.outputs = {"y", "z"};
    g.add_value_info(Tensor("t0_flat", {64, 1}, DType::FLOAT32));
    infer_shapes(g);
    EmitOptions opt;
    opt.embed_runtime = false;
    std::string c = emit_c(g, opt);
    CHECK(contains(c, "/* t0: float32[64], step 1 to step 6 */"));
    CHECK(contains(c, "/* y: float32[64], step 4 to the end */"));
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

    Graph unsupported("cast");
    unsupported.inputs = {"x"};
    unsupported.outputs = {"y"};
    unsupported.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    Node cast("c", OpType::Cast, {"x"}, {"y"});
    cast.add_attr("to", int64_t{1});
    unsupported.add_node(cast);
    infer_shapes(unsupported);
    CHECK(contains(emit_error(unsupported), "node 'c' (Cast): no code generation for this operator yet"));

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
