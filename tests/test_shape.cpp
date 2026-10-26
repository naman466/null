#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "core/graph.h"

#include <string>
#include <vector>

using namespace null;

using Shape = std::vector<int64_t>;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

// graph with float32 inputs of the given shapes and one node consuming them
static Graph one_node(OpType op, const std::vector<Shape>& input_shapes,
                      DType dtype = DType::FLOAT32, size_t num_outputs = 1) {
    Graph g("g");
    std::vector<std::string> ins, outs;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        std::string name = "in" + std::to_string(i);
        ins.push_back(name);
        g.inputs.push_back(name);
        g.add_value_info(Tensor(name, input_shapes[i], dtype));
    }
    for (size_t i = 0; i < num_outputs; ++i) outs.push_back("out" + std::to_string(i));
    g.outputs = {outs[0]};
    g.add_node(Node("n", op, ins, outs));
    return g;
}

static const Tensor& info(const Graph& g, const std::string& name) {
    const Tensor* t = g.get_tensor_info(name);
    if (!t) throw std::runtime_error("no value info for '" + name + "'");
    return *t;
}

static std::string shape_error(Graph& g) {
    try {
        infer_shapes(g);
    } catch (const ShapeError& e) {
        return e.what();
    }
    throw std::runtime_error("expected ShapeError");
}

// TESTS - BROADCASTING

TEST(broadcast_same_shape) {
    CHECK_EQ(broadcast_shapes({2, 3}, {2, 3}), (Shape{2, 3}));
    CHECK_EQ(broadcast_shapes({}, {}), (Shape{}));
}

TEST(broadcast_ones_and_rank) {
    CHECK_EQ(broadcast_shapes({2, 3}, {3}), (Shape{2, 3}));            // bias add
    CHECK_EQ(broadcast_shapes({2, 1}, {1, 3}), (Shape{2, 3}));
    CHECK_EQ(broadcast_shapes({}, {4, 5}), (Shape{4, 5}));             // scalar
    CHECK_EQ(broadcast_shapes({8, 1, 6, 1}, {7, 1, 5}), (Shape{8, 7, 6, 5}));
    CHECK_EQ(broadcast_shapes({1, 16, 1, 1}, {1, 16, 32, 32}), (Shape{1, 16, 32, 32}));
    CHECK_EQ(broadcast_shapes({0, 3}, {1, 3}), (Shape{0, 3}));          // empty stays empty
}

TEST(broadcast_dynamic_dims) {
    CHECK_EQ(broadcast_shapes({-1, 3}, {3}), (Shape{-1, 3}));
    CHECK_EQ(broadcast_shapes({-1, 3}, {1, 3}), (Shape{-1, 3}));
    CHECK_EQ(broadcast_shapes({-1, 3}, {5, 3}), (Shape{5, 3}));         // dynamic must equal 5
    CHECK_EQ(broadcast_shapes({-1}, {-1}), (Shape{-1}));
    CHECK_EQ(broadcast_shapes({1}, {-1}), (Shape{-1}));
}

TEST(broadcast_incompatible_throws) {
    CHECK_THROWS(broadcast_shapes({2, 3}, {4}));
    CHECK_THROWS(broadcast_shapes({2, 3}, {3, 2}));
    CHECK_THROWS(broadcast_shapes({0}, {2}));
    try {
        broadcast_shapes({4, 3}, {5, 3});
        throw std::runtime_error("expected ShapeError");
    } catch (const ShapeError& e) {
        CHECK(contains(e.what(), "cannot broadcast [4, 3] with [5, 3] (dim 0: 4 vs 5)"));
    }
}

// TESTS - ELEMENTWISE RULES

TEST(shape_unary_ops_keep_type_and_shape) {
    for (OpType op : {OpType::Relu, OpType::Abs, OpType::Neg, OpType::Sigmoid, OpType::Tanh,
                      OpType::Exp, OpType::Log, OpType::Sqrt, OpType::Identity}) {
        Graph g = one_node(op, {{1, -1, 8}});
        infer_shapes(g);
        CHECK_EQ(info(g, "out0").shape, (Shape{1, -1, 8}));
        CHECK_EQ(info(g, "out0").dtype, DType::FLOAT32);
    }
}

TEST(shape_unary_integer_inputs) {
    // Relu, Abs and Neg accept integers, the float-only ops reject them
    for (OpType op : {OpType::Relu, OpType::Abs, OpType::Neg, OpType::Identity}) {
        Graph g = one_node(op, {{4}}, DType::INT32);
        infer_shapes(g);
        CHECK_EQ(info(g, "out0").dtype, DType::INT32);
    }
    for (OpType op : {OpType::Sigmoid, OpType::Tanh, OpType::Exp, OpType::Log, OpType::Sqrt}) {
        Graph g = one_node(op, {{4}}, DType::INT64);
        CHECK(contains(shape_error(g), "must be float32 or float64"));
    }
    Graph b = one_node(OpType::Relu, {{4}}, DType::BOOL);
    CHECK(contains(shape_error(b), "must be numeric"));
}

TEST(shape_binary_broadcast) {
    for (OpType op : {OpType::Add, OpType::Sub, OpType::Mul, OpType::Div}) {
        Graph g = one_node(op, {{-1, 16, 1, 1}, {16, 32, 32}});
        infer_shapes(g);
        CHECK_EQ(info(g, "out0").shape, (Shape{-1, 16, 32, 32}));
    }
}

TEST(shape_binary_errors) {
    Graph bad = one_node(OpType::Add, {{2, 3}, {4}});
    std::string msg = shape_error(bad);
    CHECK(contains(msg, "node 'n' (Add)"));
    CHECK(contains(msg, "cannot broadcast"));

    // mixed element types
    Graph mixed("mixed");
    mixed.inputs = {"a", "b"};
    mixed.outputs = {"y"};
    mixed.add_value_info(Tensor("a", {2}, DType::FLOAT32));
    mixed.add_value_info(Tensor("b", {2}, DType::INT64));
    mixed.add_node(Node("mul", OpType::Mul, {"a", "b"}, {"y"}));
    CHECK(contains(shape_error(mixed), "different types (float32 and int64)"));

    Graph one_input = one_node(OpType::Add, {{2}});
    CHECK(contains(shape_error(one_input), "expects 2 inputs, got 1"));
}

TEST(shape_pow_keeps_base_type) {
    Graph g("pow");
    g.inputs = {"x", "p"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3}, DType::FLOAT32));
    g.add_value_info(Tensor("p", {}, DType::INT64));   // exponent of another type is fine
    g.add_node(Node("pow", OpType::Pow, {"x", "p"}, {"y"}));
    infer_shapes(g);
    CHECK_EQ(info(g, "y").dtype, DType::FLOAT32);
    CHECK_EQ(info(g, "y").shape, (Shape{2, 3}));
}

TEST(shape_clip_optional_bounds) {
    Graph g("clip");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {3, 4}, DType::FLOAT32));
    Tensor max("max", {}, DType::FLOAT32);
    max.set_float_data({6.0f});
    max.shape = {};
    g.add_initializer(max);
    g.add_node(Node("clip", OpType::Clip, {"x", "", "max"}, {"y"}));
    infer_shapes(g);
    CHECK_EQ(info(g, "y").shape, (Shape{3, 4}));

    // bounds must be scalars of x's type
    Graph bad = one_node(OpType::Clip, {{3, 4}, {2}});
    CHECK(contains(shape_error(bad), "must be a scalar"));
}

TEST(shape_dropout_mask) {
    Graph g = one_node(OpType::Dropout, {{2, 5}}, DType::FLOAT32, 2);
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").dtype, DType::FLOAT32);
    CHECK_EQ(info(g, "out1").dtype, DType::BOOL);
    CHECK_EQ(info(g, "out1").shape, (Shape{2, 5}));

    // omitted optional mask output
    Graph omitted("drop");
    omitted.inputs = {"x"};
    omitted.outputs = {"y"};
    omitted.add_value_info(Tensor("x", {3}, DType::FLOAT32));
    omitted.add_node(Node("d", OpType::Dropout, {"x"}, {"y", ""}));
    infer_shapes(omitted);
    CHECK(omitted.get_tensor_info("") == nullptr);
}

TEST(shape_cast) {
    Graph g = one_node(OpType::Cast, {{2, 2}});
    g.nodes[0].add_attr("to", int64_t{7});   // INT64
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").dtype, DType::INT64);
    CHECK_EQ(info(g, "out0").shape, (Shape{2, 2}));

    Graph half = one_node(OpType::Cast, {{2}});
    half.nodes[0].add_attr("to", int64_t{10});   // FLOAT16
    CHECK(contains(shape_error(half), "unsupported target type 10"));

    Graph missing = one_node(OpType::Cast, {{2}});
    CHECK(contains(shape_error(missing), "missing attribute 'to'"));
}

TEST(shape_constant_forms) {
    auto constant = [](const std::string& key, AttrValue v) {
        Graph g("c");
        g.outputs = {"c"};
        Node n("const", OpType::Constant, {}, {"c"});
        n.add_attr(key, std::move(v));
        g.add_node(n);
        infer_shapes(g);
        return info(g, "c");
    };
    Tensor value("", {2, 3}, DType::INT64);
    value.set_int64_data({1, 2, 3, 4, 5, 6});
    value.shape = {2, 3};
    Tensor t = constant("value", value);
    CHECK_EQ(t.dtype, DType::INT64);
    CHECK_EQ(t.shape, (Shape{2, 3}));

    CHECK_EQ(constant("value_float", 0.5f).shape, (Shape{}));
    CHECK_EQ(constant("value_int", int64_t{3}).dtype, DType::INT64);
    CHECK_EQ(constant("value_floats", std::vector<float>{1, 2, 3}).shape, (Shape{3}));
    CHECK_EQ(constant("value_ints", std::vector<int64_t>{1, -1}).shape, (Shape{2}));
}

// TESTS - GRAPH LEVEL

TEST(shape_chain_propagates) {
    // x -> Relu -> Mul(bias) -> Sigmoid, each output gets a type
    Graph g("chain");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {-1, 4}, DType::FLOAT32));
    Tensor s("s", {4}, DType::FLOAT32);
    s.set_float_data({1, 2, 3, 4});
    g.add_initializer(s);
    // stored out of order, inference follows execution order
    g.add_node(Node("sig", OpType::Sigmoid, {"m"}, {"y"}));
    g.add_node(Node("mul", OpType::Mul, {"r", "s"}, {"m"}));
    g.add_node(Node("relu", OpType::Relu, {"x"}, {"r"}));
    infer_shapes(g);
    for (const char* name : {"r", "m", "y"}) {
        CHECK_EQ(info(g, name).shape, (Shape{-1, 4}));
        CHECK_EQ(info(g, name).dtype, DType::FLOAT32);
    }
}

TEST(shape_declared_dims_refine) {
    // the model declares y as [8, 4], inference only knows [?, 4]
    Graph g = one_node(OpType::Relu, {{-1, 4}});
    g.add_value_info(Tensor("out0", {8, 4}, DType::FLOAT32));
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").shape, (Shape{8, 4}));
}

TEST(shape_declared_conflicts_throw) {
    Graph dim = one_node(OpType::Relu, {{2, 4}});
    dim.add_value_info(Tensor("out0", {2, 5}, DType::FLOAT32));
    std::string msg = shape_error(dim);
    CHECK(contains(msg, "declared as float32[2, 5] but inferred as float32[2, 4] (dim 1)"));

    Graph rank = one_node(OpType::Relu, {{2, 4}});
    rank.add_value_info(Tensor("out0", {8}, DType::FLOAT32));
    CHECK(contains(shape_error(rank), "(rank)"));

    Graph type = one_node(OpType::Relu, {{2, 4}});
    type.add_value_info(Tensor("out0", {2, 4}, DType::INT64));
    CHECK(contains(shape_error(type), "(type)"));
}

TEST(shape_missing_input_metadata) {
    Graph g("missing");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_node(Node("relu", OpType::Relu, {"x"}, {"y"}));   // x has no value info
    CHECK(contains(shape_error(g), "input 'x' has no type or shape"));
}

TEST(shape_unsupported_op_reported) {
    Graph g = one_node(OpType::Conv, {{1, 3, 8, 8}, {4, 3, 3, 3}});
    CHECK(contains(shape_error(g), "node 'n' (Conv): no shape inference rule"));
    CHECK(!has_shape_rule(OpType::Conv));
    CHECK(has_shape_rule(OpType::Add));
}

TEST(shape_too_many_outputs) {
    Graph g = one_node(OpType::Relu, {{2}}, DType::FLOAT32, 2);
    CHECK(contains(shape_error(g), "has 2 outputs, expected at most 1"));
}

TEST(shape_dead_nodes_skipped) {
    Graph g = one_node(OpType::Relu, {{2}});
    g.add_node(Node("dead", OpType::Conv, {"in0"}, {"junk"}));   // no rule, but dead
    g.nodes.back().is_dead = true;
    infer_shapes(g);
    CHECK(g.get_tensor_info("junk") == nullptr);
}

// TESTS - MATMUL

static Shape matmul_shape(const Shape& a, const Shape& b, DType dt = DType::FLOAT32) {
    Graph g = one_node(OpType::MatMul, {a, b}, dt);
    infer_shapes(g);
    return info(g, "out0").shape;
}

TEST(shape_matmul_2d) {
    CHECK_EQ(matmul_shape({2, 3}, {3, 4}), (Shape{2, 4}));
    CHECK_EQ(matmul_shape({1, 784}, {784, 128}), (Shape{1, 128}));
    CHECK_EQ(matmul_shape({-1, 784}, {784, 10}), (Shape{-1, 10}));
}

TEST(shape_matmul_vectors) {
    // 1-D operands get a dim added and then removed again
    CHECK_EQ(matmul_shape({3}, {3, 4}), (Shape{4}));
    CHECK_EQ(matmul_shape({2, 3}, {3}), (Shape{2}));
    CHECK_EQ(matmul_shape({3}, {3}), (Shape{}));
    CHECK_EQ(matmul_shape({5, 2, 3}, {3}), (Shape{5, 2}));
    CHECK_EQ(matmul_shape({3}, {5, 3, 4}), (Shape{5, 4}));
}

TEST(shape_matmul_batched) {
    CHECK_EQ(matmul_shape({8, 2, 3}, {8, 3, 4}), (Shape{8, 2, 4}));
    CHECK_EQ(matmul_shape({8, 2, 3}, {3, 4}), (Shape{8, 2, 4}));          // shared weights
    CHECK_EQ(matmul_shape({2, 1, 5, 3}, {7, 3, 4}), (Shape{2, 7, 5, 4}));  // batch broadcast
    CHECK_EQ(matmul_shape({-1, 12, 64, 64}, {-1, 12, 64, 32}), (Shape{-1, 12, 64, 32}));
}

TEST(shape_matmul_types) {
    Graph g = one_node(OpType::MatMul, {{2, 3}, {3, 1}}, DType::INT64);
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").dtype, DType::INT64);

    Graph u8 = one_node(OpType::MatMul, {{2, 3}, {3, 1}}, DType::UINT8);
    CHECK(contains(shape_error(u8), "must be float32, float64, int32 or int64, got uint8"));
}

TEST(shape_matmul_errors) {
    Graph inner = one_node(OpType::MatMul, {{2, 3}, {4, 5}});
    CHECK(contains(shape_error(inner), "inner dimensions differ: [2, 3] x [4, 5] (3 vs 4)"));

    Graph batch = one_node(OpType::MatMul, {{2, 2, 3}, {5, 3, 4}});
    CHECK(contains(shape_error(batch), "batch dimensions: cannot broadcast"));

    Graph scalar = one_node(OpType::MatMul, {{}, {3}});
    CHECK(contains(shape_error(scalar), "must be at least 1-D"));

    // a dynamic inner dim is accepted
    CHECK_EQ(matmul_shape({2, -1}, {5, 4}), (Shape{2, 4}));
}

// TESTS - GEMM

static Graph gemm_graph(const Shape& a, const Shape& b, const Shape* c,
                        int64_t trans_a = 0, int64_t trans_b = 0) {
    std::vector<Shape> shapes = {a, b};
    if (c) shapes.push_back(*c);
    Graph g = one_node(OpType::Gemm, shapes);
    if (trans_a) g.nodes[0].add_attr("transA", trans_a);
    if (trans_b) g.nodes[0].add_attr("transB", trans_b);
    g.nodes[0].add_attr("alpha", 0.5f);   // scaling never changes the shape
    return g;
}

static Shape gemm_shape(const Shape& a, const Shape& b, const Shape* c,
                        int64_t trans_a = 0, int64_t trans_b = 0) {
    Graph g = gemm_graph(a, b, c, trans_a, trans_b);
    infer_shapes(g);
    return info(g, "out0").shape;
}

TEST(shape_gemm_transposes) {
    CHECK_EQ(gemm_shape({2, 3}, {3, 4}, nullptr), (Shape{2, 4}));
    CHECK_EQ(gemm_shape({3, 2}, {3, 4}, nullptr, 1, 0), (Shape{2, 4}));
    CHECK_EQ(gemm_shape({2, 3}, {4, 3}, nullptr, 0, 1), (Shape{2, 4}));   // pytorch Linear
    CHECK_EQ(gemm_shape({3, 2}, {4, 3}, nullptr, 1, 1), (Shape{2, 4}));
    CHECK_EQ(gemm_shape({-1, 784}, {128, 784}, nullptr, 0, 1), (Shape{-1, 128}));
}

TEST(shape_gemm_bias_shapes) {
    for (const Shape& c : {Shape{}, Shape{4}, Shape{1, 4}, Shape{2, 1}, Shape{2, 4}, Shape{1}}) {
        CHECK_EQ(gemm_shape({2, 3}, {4, 3}, &c, 0, 1), (Shape{2, 4}));
    }
    // a bias against a dynamic batch
    Shape bias = {128};
    CHECK_EQ(gemm_shape({-1, 784}, {128, 784}, &bias, 0, 1), (Shape{-1, 128}));
}

TEST(shape_gemm_errors) {
    Graph inner = gemm_graph({2, 3}, {3, 4}, nullptr, 0, 1);   // transB makes B [4, 3]
    CHECK(contains(shape_error(inner), "inner dimensions differ: A [2, 3], B [3, 4] (transposed) (3 vs 4)"));

    Shape wide = {5};
    Graph bias = gemm_graph({2, 3}, {3, 4}, &wide);
    CHECK(contains(shape_error(bias), "C with shape [5] cannot broadcast to the result [2, 4]"));

    // C may not grow the result
    Shape grows = {3, 2, 4};
    Graph rank = gemm_graph({2, 3}, {3, 4}, &grows);
    CHECK(contains(shape_error(rank), "cannot broadcast to the result"));

    Graph not2d = one_node(OpType::Gemm, {{2, 3, 4}, {4, 5}});
    CHECK(contains(shape_error(not2d), "A must be 2-D, got [2, 3, 4]"));
}

TEST(shape_mlp_end_to_end) {
    // Gemm -> Relu -> Gemm, as an exported Linear-ReLU-Linear
    Graph g("mlp");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {-1, 16}, DType::FLOAT32));
    auto weights = [&](const std::string& name, Shape shape) {
        Tensor t(name, shape, DType::FLOAT32);
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        t.set_float_data(std::vector<float>(static_cast<size_t>(n), 0.0f));
        t.shape = shape;
        g.add_initializer(t);
    };
    weights("w1", {8, 16});
    weights("b1", {8});
    weights("w2", {3, 8});
    weights("b2", {3});
    Node fc1("fc1", OpType::Gemm, {"x", "w1", "b1"}, {"h"});
    fc1.add_attr("transB", int64_t{1});
    Node fc2("fc2", OpType::Gemm, {"r", "w2", "b2"}, {"y"});
    fc2.add_attr("transB", int64_t{1});
    g.add_node(fc1);
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"r"}));
    g.add_node(fc2);
    infer_shapes(g);
    CHECK_EQ(info(g, "h").shape, (Shape{-1, 8}));
    CHECK_EQ(info(g, "r").shape, (Shape{-1, 8}));
    CHECK_EQ(info(g, "y").shape, (Shape{-1, 3}));
}
