#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "analysis/specialize.h"
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
    Graph g = one_node(OpType::AveragePool, {{1, 3, 8, 8}});
    CHECK(contains(shape_error(g), "node 'n' (AveragePool): no shape inference rule"));
    CHECK(!has_shape_rule(OpType::AveragePool));
    CHECK(has_shape_rule(OpType::MaxPool));
    CHECK(has_shape_rule(OpType::Conv));
    CHECK(has_shape_rule(OpType::Add));
}

TEST(shape_too_many_outputs) {
    Graph g = one_node(OpType::Relu, {{2}}, DType::FLOAT32, 2);
    CHECK(contains(shape_error(g), "has 2 outputs, expected at most 1"));
}

TEST(shape_dead_nodes_skipped) {
    Graph g = one_node(OpType::Relu, {{2}});
    g.add_node(Node("dead", OpType::AveragePool, {"in0"}, {"junk"}));   // no rule, but dead
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

// TESTS - RESHAPE

static Graph reshape_graph(const Shape& in, const std::vector<int64_t>& target, bool allowzero = false) {
    Graph g = one_node(OpType::Reshape, {in});
    Tensor t("target", {static_cast<int64_t>(target.size())}, DType::INT64);
    t.set_int64_data(target);
    t.shape = {static_cast<int64_t>(target.size())};
    g.add_initializer(t);
    g.nodes[0].inputs.push_back("target");
    if (allowzero) g.nodes[0].add_attr("allowzero", int64_t{1});
    return g;
}

static Shape reshape_shape(const Shape& in, const std::vector<int64_t>& target, bool allowzero = false) {
    Graph g = reshape_graph(in, target, allowzero);
    infer_shapes(g);
    return info(g, "out0").shape;
}

TEST(shape_reshape_explicit) {
    CHECK_EQ(reshape_shape({2, 3, 4}, {6, 4}), (Shape{6, 4}));
    CHECK_EQ(reshape_shape({24}, {2, 3, 4}), (Shape{2, 3, 4}));
    CHECK_EQ(reshape_shape({}, {1, 1}), (Shape{1, 1}));            // scalar
    CHECK_EQ(reshape_shape({1, 1}, {}), (Shape{}));                 // to scalar
}

TEST(shape_reshape_minus_one) {
    CHECK_EQ(reshape_shape({2, 3, 4}, {-1}), (Shape{24}));
    CHECK_EQ(reshape_shape({2, 3, 4}, {2, -1}), (Shape{2, 12}));
    CHECK_EQ(reshape_shape({1, 1, 28, 28}, {1, -1}), (Shape{1, 784}));
    // a dynamic input dim keeps the inferred dim dynamic
    CHECK_EQ(reshape_shape({-1, 1, 28, 28}, {-1, 784}), (Shape{-1, 784}));
}

TEST(shape_reshape_zero_copies) {
    CHECK_EQ(reshape_shape({2, 3, 4}, {0, -1}), (Shape{2, 12}));
    CHECK_EQ(reshape_shape({2, 3, 4}, {0, 0, 2, 2}), (Shape{2, 3, 2, 2}));
    // the copied dynamic batch cancels out, so -1 can still be solved
    CHECK_EQ(reshape_shape({-1, 16, 7, 7}, {0, -1}), (Shape{-1, 784}));
    CHECK_EQ(reshape_shape({-1, 784}, {0, 28, 28}), (Shape{-1, 28, 28}));
}

TEST(shape_reshape_allowzero) {
    // with allowzero=1 a 0 is a literal empty dim
    CHECK_EQ(reshape_shape({0, 4}, {4, 0}, true), (Shape{4, 0}));
    Graph bad = reshape_graph({2, 3}, {0, -1}, true);
    CHECK(contains(shape_error(bad), "both 0 and -1 with allowzero=1"));
}

TEST(shape_reshape_errors) {
    Graph count = reshape_graph({2, 3, 4}, {5, 5});
    CHECK(contains(shape_error(count), "cannot reshape [2, 3, 4] to [5, 5]: 24 vs 25 elements"));

    Graph divide = reshape_graph({2, 3, 4}, {5, -1});
    CHECK(contains(shape_error(divide), "24 elements are not divisible by 5"));

    Graph two = reshape_graph({2, 3}, {-1, -1});
    CHECK(contains(shape_error(two), "more than one -1"));

    Graph neg = reshape_graph({6}, {-2, 3});
    CHECK(contains(shape_error(neg), "invalid dim -2"));

    Graph copy = reshape_graph({6}, {2, 0});
    CHECK(contains(shape_error(copy), "copies dim 1 with 0, but the input [6] has rank 1"));
}

TEST(shape_reshape_needs_constant_shape) {
    // the target shape is a graph input: not known at compile time
    Graph g = one_node(OpType::Reshape, {{2, 3}});
    g.inputs.push_back("target");
    g.add_value_info(Tensor("target", {1}, DType::INT64));
    g.nodes[0].inputs.push_back("target");
    CHECK(contains(shape_error(g), "('target') must be a constant"));
}

TEST(shape_reshape_shape_from_constant_node) {
    // exporters often produce the target shape with a Constant node
    Graph g("const_shape");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {2, 3, 4}, DType::FLOAT32));
    Node c("shape", OpType::Constant, {}, {"s"});
    c.add_attr("value_ints", std::vector<int64_t>{4, -1});
    g.add_node(c);
    g.add_node(Node("reshape", OpType::Reshape, {"x", "s"}, {"y"}));
    infer_shapes(g);
    CHECK_EQ(info(g, "y").shape, (Shape{4, 6}));
}

// TESTS - FLATTEN

static Shape flatten_shape(const Shape& in, int64_t axis) {
    Graph g = one_node(OpType::Flatten, {in});
    g.nodes[0].add_attr("axis", axis);
    infer_shapes(g);
    return info(g, "out0").shape;
}

TEST(shape_flatten) {
    CHECK_EQ(flatten_shape({2, 3, 4, 5}, 1), (Shape{2, 60}));
    CHECK_EQ(flatten_shape({2, 3, 4, 5}, 2), (Shape{6, 20}));
    CHECK_EQ(flatten_shape({2, 3, 4, 5}, 0), (Shape{1, 120}));
    CHECK_EQ(flatten_shape({2, 3, 4, 5}, 4), (Shape{120, 1}));
    CHECK_EQ(flatten_shape({2, 3, 4, 5}, -1), (Shape{24, 5}));
    CHECK_EQ(flatten_shape({-1, 16, 1, 1}, 1), (Shape{-1, 16}));    // after GlobalAveragePool
    CHECK_EQ(flatten_shape({8, -1, 4}, 1), (Shape{8, -1}));

    // default axis is 1
    Graph g = one_node(OpType::Flatten, {{2, 3, 4}});
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").shape, (Shape{2, 12}));

    Graph bad = one_node(OpType::Flatten, {{2, 3}});
    bad.nodes[0].add_attr("axis", int64_t{3});
    CHECK(contains(shape_error(bad), "axis 3 is out of range [-2, 2] for rank 2"));
}

// TESTS - SOFTMAX

TEST(shape_softmax) {
    Graph g = one_node(OpType::Softmax, {{-1, 10}});
    infer_shapes(g);
    CHECK_EQ(info(g, "out0").shape, (Shape{-1, 10}));

    Graph axis = one_node(OpType::Softmax, {{2, 3, 4}});
    axis.nodes[0].add_attr("axis", int64_t{-3});
    infer_shapes(axis);
    CHECK_EQ(info(axis, "out0").shape, (Shape{2, 3, 4}));

    Graph bad_axis = one_node(OpType::Softmax, {{2, 3}});
    bad_axis.nodes[0].add_attr("axis", int64_t{2});
    CHECK(contains(shape_error(bad_axis), "axis 2 is out of range [-2, 1]"));

    Graph scalar = one_node(OpType::Softmax, {{}});
    CHECK(contains(shape_error(scalar), "rank >= 1"));

    Graph ints = one_node(OpType::Softmax, {{4}}, DType::INT64);
    CHECK(contains(shape_error(ints), "must be float32 or float64"));
}

// TESTS - CONVOLUTION

// Conv on x with weight w, attributes set by the caller
static Graph conv_graph(const Shape& x, const Shape& w, bool bias = false) {
    std::vector<Shape> ins{x, w};
    if (bias) ins.push_back({w[0]});
    return one_node(OpType::Conv, ins);
}

static Shape conv_out(Graph g) {
    infer_shapes(g);
    return info(g, "out0").shape;
}

TEST(shape_conv_defaults) {
    // lenet's second conv: no padding, stride 1
    CHECK_EQ(conv_out(conv_graph({1, 6, 14, 14}, {16, 6, 5, 5}, true)), (Shape{1, 16, 10, 10}));
    CHECK_EQ(conv_out(conv_graph({-1, 3, 8, 8}, {4, 3, 1, 1})), (Shape{-1, 4, 8, 8}));   // batch stays dynamic
    CHECK_EQ(conv_out(conv_graph({2, 1, 3, 3}, {1, 1, 3, 3})), (Shape{2, 1, 1, 1}));     // kernel fills the input
}

TEST(shape_conv_pads_strides_dilations) {
    Graph same = conv_graph({1, 1, 28, 28}, {6, 1, 5, 5});
    same.nodes[0].add_attr("pads", std::vector<int64_t>{2, 2, 2, 2});
    CHECK_EQ(conv_out(same), (Shape{1, 6, 28, 28}));

    // asymmetric pads: H gets 1 + 0, W gets 0 + 2
    Graph asym = conv_graph({1, 1, 5, 5}, {1, 1, 3, 3});
    asym.nodes[0].add_attr("pads", std::vector<int64_t>{1, 0, 0, 2});
    CHECK_EQ(conv_out(asym), (Shape{1, 1, 4, 5}));

    // stride 2 rounds down: (7 - 3) / 2 + 1 = 3, (8 - 3) / 2 + 1 = 3
    Graph stride = conv_graph({1, 1, 7, 8}, {1, 1, 3, 3});
    stride.nodes[0].add_attr("strides", std::vector<int64_t>{2, 2});
    CHECK_EQ(conv_out(stride), (Shape{1, 1, 3, 3}));

    // dilation 2 makes a 3x3 kernel span 5x5
    Graph dil = conv_graph({1, 1, 9, 9}, {1, 1, 3, 3});
    dil.nodes[0].add_attr("dilations", std::vector<int64_t>{2, 1});
    CHECK_EQ(conv_out(dil), (Shape{1, 1, 5, 7}));
}

TEST(shape_conv_groups) {
    // depthwise: group == C, one input channel per filter
    Graph dw = conv_graph({1, 8, 6, 6}, {8, 1, 3, 3});
    dw.nodes[0].add_attr("group", int64_t{8});
    CHECK_EQ(conv_out(dw), (Shape{1, 8, 4, 4}));

    Graph two = conv_graph({1, 4, 6, 6}, {6, 2, 3, 3});
    two.nodes[0].add_attr("group", int64_t{2});
    CHECK_EQ(conv_out(two), (Shape{1, 6, 4, 4}));
}

TEST(shape_conv_auto_pad) {
    auto with_pad = [](const char* mode, const Shape& stride) {
        Graph g = conv_graph({1, 1, 7, 7}, {1, 1, 4, 4});
        g.nodes[0].add_attr("auto_pad", std::string(mode));
        g.nodes[0].add_attr("strides", stride);
        return g;
    };
    CHECK_EQ(conv_out(with_pad("SAME_UPPER", {1, 1})), (Shape{1, 1, 7, 7}));
    CHECK_EQ(conv_out(with_pad("SAME_LOWER", {1, 1})), (Shape{1, 1, 7, 7}));
    CHECK_EQ(conv_out(with_pad("SAME_UPPER", {2, 3})), (Shape{1, 1, 4, 3}));   // ceil(7 / s)
    CHECK_EQ(conv_out(with_pad("VALID", {1, 1})), (Shape{1, 1, 4, 4}));

    // the odd pixel of padding goes to the end for UPPER, the start for LOWER
    std::vector<int64_t> x{1, 1, 7, 7}, w{1, 1, 4, 4};
    Graph upper = with_pad("SAME_UPPER", {1, 1}), lower = with_pad("SAME_LOWER", {1, 1});
    ConvGeometry u = conv_geometry(upper.nodes[0], x, w), l = conv_geometry(lower.nodes[0], x, w);
    CHECK_EQ(u.pad_top, 1); CHECK_EQ(u.pad_bottom, 2);
    CHECK_EQ(l.pad_top, 2); CHECK_EQ(l.pad_bottom, 1);
}

TEST(shape_conv_errors) {
    auto err = [](Graph g) { return shape_error(g); };
    CHECK(contains(err(conv_graph({1, 3, 8}, {4, 3, 3})), "only 2-D convolution is supported"));
    CHECK(contains(err(conv_graph({1, -1, 8, 8}, {4, 3, 3, 3})), "must have static C, H and W"));
    CHECK(contains(err(conv_graph({1, 3, 8, 8}, {4, 2, 3, 3})), "input has 3 channels, but the weight"));
    CHECK(contains(err(conv_graph({1, 3, 2, 2}, {4, 3, 3, 3})), "is larger than the padded input 2x2"));

    Graph bias = one_node(OpType::Conv, {{1, 3, 8, 8}, {4, 3, 3, 3}, {3}});
    CHECK(contains(shape_error(bias), "bias must be [4]"));

    Graph group = conv_graph({1, 4, 8, 8}, {3, 2, 3, 3});
    group.nodes[0].add_attr("group", int64_t{2});
    CHECK(contains(shape_error(group), "3 output channels are not divisible by group 2"));

    Graph ks = conv_graph({1, 3, 8, 8}, {4, 3, 3, 3});
    ks.nodes[0].add_attr("kernel_shape", std::vector<int64_t>{5, 5});
    CHECK(contains(shape_error(ks), "kernel_shape does not match"));

    Graph stride = conv_graph({1, 3, 8, 8}, {4, 3, 3, 3});
    stride.nodes[0].add_attr("strides", std::vector<int64_t>{0, 1});
    CHECK(contains(shape_error(stride), "strides must be at least 1"));

    Graph pads = conv_graph({1, 3, 8, 8}, {4, 3, 3, 3});
    pads.nodes[0].add_attr("pads", std::vector<int64_t>{1, 1});
    CHECK(contains(shape_error(pads), "pads must have 4 values"));

    Graph neg = conv_graph({1, 3, 8, 8}, {4, 3, 3, 3});
    neg.nodes[0].add_attr("pads", std::vector<int64_t>{-1, 0, 0, 0});
    CHECK(contains(shape_error(neg), "pads must not be negative"));

    Graph mode = conv_graph({1, 3, 8, 8}, {4, 3, 3, 3});
    mode.nodes[0].add_attr("auto_pad", std::string("SAME"));
    CHECK(contains(shape_error(mode), "unknown auto_pad 'SAME'"));

    Graph ints = one_node(OpType::Conv, {{1, 3, 8, 8}, {4, 3, 3, 3}}, DType::INT64);
    CHECK(contains(shape_error(ints), "must be float32 or float64"));
}

// TESTS - MAXPOOL

static Graph pool_graph(const Shape& x, const Shape& kernel) {
    Graph g = one_node(OpType::MaxPool, {x});
    g.nodes[0].add_attr("kernel_shape", kernel);
    return g;
}

TEST(shape_maxpool_basic) {
    // lenet: 2x2 windows, stride 2
    Graph lenet = pool_graph({1, 6, 28, 28}, {2, 2});
    lenet.nodes[0].add_attr("strides", std::vector<int64_t>{2, 2});
    CHECK_EQ(conv_out(lenet), (Shape{1, 6, 14, 14}));

    // default stride is 1, channels and a dynamic batch pass through
    CHECK_EQ(conv_out(pool_graph({-1, 3, 5, 5}, {3, 3})), (Shape{-1, 3, 3, 3}));

    // resnet's stem: 3x3, stride 2, pad 1
    Graph stem = pool_graph({1, 64, 112, 112}, {3, 3});
    stem.nodes[0].add_attr("strides", std::vector<int64_t>{2, 2});
    stem.nodes[0].add_attr("pads", std::vector<int64_t>{1, 1, 1, 1});
    CHECK_EQ(conv_out(stem), (Shape{1, 64, 56, 56}));

    Graph dil = pool_graph({1, 1, 7, 7}, {2, 2});
    dil.nodes[0].add_attr("dilations", std::vector<int64_t>{3, 1});
    CHECK_EQ(conv_out(dil), (Shape{1, 1, 4, 6}));      // 2 taps 3 apart span 4 rows

    Graph same = pool_graph({1, 1, 5, 5}, {2, 2});
    same.nodes[0].add_attr("strides", std::vector<int64_t>{2, 2});
    same.nodes[0].add_attr("auto_pad", std::string("SAME_UPPER"));
    CHECK_EQ(conv_out(same), (Shape{1, 1, 3, 3}));
}

TEST(shape_maxpool_ceil_mode) {
    auto ceil = [](const Shape& x, int64_t k, int64_t stride, const std::vector<int64_t>& pads) {
        Graph g = pool_graph(x, {k, k});
        g.nodes[0].add_attr("strides", std::vector<int64_t>{stride, stride});
        g.nodes[0].add_attr("pads", pads);
        g.nodes[0].add_attr("ceil_mode", int64_t{1});
        return conv_out(g);
    };
    // 5 wide, 2x2 stride 2: floor gives 2, ceil keeps the partial last window
    CHECK_EQ(ceil({1, 1, 5, 5}, 2, 2, {0, 0, 0, 0}), (Shape{1, 1, 3, 3}));
    CHECK_EQ(ceil({1, 1, 6, 6}, 3, 2, {1, 1, 1, 1}), (Shape{1, 1, 4, 4}));
    // rounding up would add a window starting in the end padding, which is dropped
    // (onnx's shape inference says 3 here; onnx runtime and its reference give 2)
    CHECK_EQ(ceil({1, 1, 4, 4}, 2, 2, {0, 0, 1, 1}), (Shape{1, 1, 2, 2}));
    CHECK_EQ(ceil({1, 1, 7, 7}, 2, 3, {0, 0, 1, 1}), (Shape{1, 1, 3, 3}));

    // ceil_mode is ignored with auto_pad, which already rounds up
    Graph g = pool_graph({1, 1, 5, 5}, {2, 2});
    g.nodes[0].add_attr("strides", std::vector<int64_t>{2, 2});
    g.nodes[0].add_attr("auto_pad", std::string("VALID"));
    g.nodes[0].add_attr("ceil_mode", int64_t{1});
    CHECK_EQ(conv_out(g), (Shape{1, 1, 2, 2}));
}

TEST(shape_maxpool_errors) {
    Graph no_kernel = one_node(OpType::MaxPool, {{1, 3, 8, 8}});
    CHECK(contains(shape_error(no_kernel), "kernel_shape is required"));
    Graph rank = pool_graph({1, 3, 8}, {2, 2});
    CHECK(contains(shape_error(rank), "only 2-D pooling is supported"));
    Graph dyn = pool_graph({1, 3, -1, 8}, {2, 2});
    CHECK(contains(shape_error(dyn), "must have static C, H and W"));
    Graph k1 = pool_graph({1, 3, 8, 8}, {2});
    CHECK(contains(shape_error(k1), "kernel_shape must have 2 values"));
    Graph k0 = pool_graph({1, 3, 8, 8}, {0, 2});
    CHECK(contains(shape_error(k0), "kernel_shape must be at least 1"));
    Graph big = pool_graph({1, 3, 2, 2}, {3, 3});
    CHECK(contains(shape_error(big), "larger than the padded input 2x2"));

    Graph pads = pool_graph({1, 3, 8, 8}, {2, 2});
    pads.nodes[0].add_attr("pads", std::vector<int64_t>{2, 0, 0, 0});
    CHECK(contains(shape_error(pads), "pads must be smaller than the kernel"));

    Graph order = pool_graph({1, 3, 8, 8}, {2, 2});
    order.nodes[0].add_attr("storage_order", int64_t{1});
    CHECK(contains(shape_error(order), "storage_order 1"));

    Graph indices = one_node(OpType::MaxPool, {{1, 3, 8, 8}}, DType::FLOAT32, 2);
    indices.nodes[0].add_attr("kernel_shape", std::vector<int64_t>{2, 2});
    CHECK(contains(shape_error(indices), "the Indices output is not supported"));

    Graph ints = one_node(OpType::MaxPool, {{1, 3, 8, 8}}, DType::INT64);
    ints.nodes[0].add_attr("kernel_shape", std::vector<int64_t>{2, 2});
    CHECK(contains(shape_error(ints), "must be float32 or float64"));
}

// TESTS - SPECIALISING INPUT SHAPES

static Graph dynamic_mlp() {
    // x[?, 16] -> Gemm -> Relu -> y[?, 4]
    Graph g("dyn");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {-1, 16}, DType::FLOAT32));
    Tensor w("w", {4, 16}, DType::FLOAT32);
    w.set_float_data(std::vector<float>(64, 0.0f));
    w.shape = {4, 16};
    g.add_initializer(w);
    Node fc("fc", OpType::Gemm, {"x", "w"}, {"h"});
    fc.add_attr("transB", int64_t{1});
    g.add_node(fc);
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"y"}));
    return g;
}

TEST(specialize_dynamic_tensors_listed) {
    Graph g = dynamic_mlp();
    infer_shapes(g);
    // input first, then node outputs in execution order
    CHECK_EQ(dynamic_tensors(g), (std::vector<std::string>{"x", "h", "y"}));
}

TEST(specialize_batch_makes_everything_static) {
    Graph g = dynamic_mlp();
    CHECK_EQ(set_batch_size(g, 8), 1);
    infer_shapes(g);
    CHECK(dynamic_tensors(g).empty());
    CHECK_EQ(info(g, "h").shape, (Shape{8, 4}));
    CHECK_EQ(info(g, "y").shape, (Shape{8, 4}));
}

TEST(specialize_batch_leaves_fixed_inputs) {
    Graph g = one_node(OpType::Add, {{2, 3}, {-1, 3}});
    CHECK_EQ(set_batch_size(g, 5), 1);
    CHECK_EQ(info(g, "in0").shape, (Shape{2, 3}));   // already fixed
    CHECK_EQ(info(g, "in1").shape, (Shape{5, 3}));

    Graph scalar = one_node(OpType::Relu, {{}});
    CHECK_EQ(set_batch_size(scalar, 5), 0);           // nothing to change

    CHECK_THROWS(set_batch_size(g, 0));
}

TEST(specialize_input_shape) {
    Graph g = dynamic_mlp();
    set_input_shape(g, "x", {3, 16});
    infer_shapes(g);
    CHECK_EQ(info(g, "y").shape, (Shape{3, 4}));
    CHECK(dynamic_tensors(g).empty());
}

TEST(specialize_input_shape_errors) {
    auto error = [](const std::string& input, const Shape& shape) {
        Graph g = dynamic_mlp();
        try {
            set_input_shape(g, input, shape);
        } catch (const ShapeError& e) {
            return std::string(e.what());
        }
        throw std::runtime_error("expected ShapeError");
    };
    CHECK(contains(error("x", {3, 16, 1}), "rank 3 instead of 2"));
    CHECK(contains(error("x", {3, 15}), "dim 1 is fixed at 16"));
    CHECK(contains(error("x", {-1, 16}), "dim 0 must be >= 0"));
    CHECK(contains(error("w", {4, 16}), "'w' is not a graph input (inputs: x)"));
    CHECK(contains(error("nope", {1}), "is not a graph input"));
}

TEST(specialize_declared_intermediates_follow_inputs) {
    // the model declared h as [?, 4]; after fixing the batch it becomes [2, 4]
    Graph g = dynamic_mlp();
    g.add_value_info(Tensor("h", {-1, 4}, DType::FLOAT32));
    set_batch_size(g, 2);
    infer_shapes(g);
    CHECK_EQ(info(g, "h").shape, (Shape{2, 4}));
}
