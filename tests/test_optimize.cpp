#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "analysis/specialize.h"
#include "core/graph.h"
#include "opt/optimize.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace null;

static void add_weights(Graph& g, const std::string& name) {
    Tensor t(name, {4}, DType::FLOAT32);
    t.set_float_data({1, 2, 3, 4});
    g.add_initializer(t);
}

// x[4] in, every tensor float32[4], weights of 4 floats
static Graph chain(const std::vector<Node>& nodes, const std::vector<std::string>& outputs = {"y"},
                   const std::vector<std::string>& weights = {}) {
    Graph g("g");
    for (const std::string& w : weights) add_weights(g, w);
    g.inputs = {"x"};
    g.outputs = outputs;
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    for (const Node& n : nodes) g.add_node(n);
    infer_shapes(g);
    return g;
}

static const Node& node(const Graph& g, const std::string& name) {
    for (const Node& n : g.nodes)
        if (n.name == name) return n;
    throw std::runtime_error("no node '" + name + "'");
}

// TESTS - REMOVING IDENTITIES

TEST(opt_identity_in_the_middle) {
    Graph g = chain({Node("r1", OpType::Relu, {"x"}, {"a"}), Node("id", OpType::Identity, {"a"}, {"b"}),
                     Node("r2", OpType::Relu, {"b"}, {"y"})});
    CHECK_EQ(remove_identities(g), size_t{1});
    CHECK(node(g, "id").is_dead);
    CHECK_EQ(node(g, "r2").inputs, (std::vector<std::string>{"a"}));
    CHECK(!g.get_tensor_info("b"));
    CHECK_EQ(g.live_node_count(), size_t{2});
    CHECK(g.validate().empty());
}

TEST(opt_identity_writing_a_graph_output) {
    // the output keeps its name: the Relu now writes y itself
    Graph g = chain({Node("r", OpType::Relu, {"x"}, {"a"}), Node("id", OpType::Identity, {"a"}, {"y"}),
                     Node("r2", OpType::Relu, {"a"}, {"z"})}, {"y", "z"});
    CHECK_EQ(remove_identities(g), size_t{1});
    CHECK_EQ(node(g, "r").outputs, (std::vector<std::string>{"y"}));
    CHECK_EQ(node(g, "r2").inputs, (std::vector<std::string>{"y"}));
    CHECK(!g.get_tensor_info("a"));
    CHECK_EQ(g.outputs, (std::vector<std::string>{"y", "z"}));
    CHECK(g.validate().empty());
}

TEST(opt_identities_that_stay) {
    // from a graph input or an initializer straight to a graph output: no node to rename
    Graph direct = chain({Node("id", OpType::Identity, {"x"}, {"y"})});
    CHECK_EQ(remove_identities(direct), size_t{0});
    Graph g("w");
    g.outputs = {"y"};
    add_weights(g, "w");
    g.add_node(Node("id", OpType::Identity, {"w"}, {"y"}));
    infer_shapes(g);
    CHECK_EQ(remove_identities(g), size_t{0});
    // an input that is itself a graph output
    Graph both = chain({Node("r", OpType::Relu, {"x"}, {"a"}), Node("id", OpType::Identity, {"a"}, {"y"})},
                       {"a", "y"});
    CHECK_EQ(remove_identities(both), size_t{0});
}

TEST(opt_dropout) {
    // the mask unused: removed, mask and all
    Graph g = chain({Node("d", OpType::Dropout, {"x"}, {"a", "mask"}), Node("r", OpType::Relu, {"a"}, {"y"})});
    CHECK_EQ(remove_identities(g), size_t{1});
    CHECK_EQ(node(g, "r").inputs, (std::vector<std::string>{"x"}));
    CHECK(!g.get_tensor_info("mask"));
    // the mask read by another node, or a training_mode input: it stays
    Node cast("c", OpType::Cast, {"mask"}, {"z"});
    cast.add_attr("to", int64_t{1});
    Graph used = chain({Node("d", OpType::Dropout, {"x"}, {"a", "mask"}), Node("r", OpType::Relu, {"a"}, {"y"}), cast},
                       {"y", "z"});
    CHECK_EQ(remove_identities(used), size_t{0});
    Graph training("t");
    training.inputs = {"x", "ratio", "mode"};
    training.outputs = {"y"};
    training.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    training.add_value_info(Tensor("ratio", {}, DType::FLOAT32));
    training.add_value_info(Tensor("mode", {}, DType::BOOL));
    training.add_node(Node("d", OpType::Dropout, {"x", "ratio", "mode"}, {"a"}));
    training.add_node(Node("r", OpType::Relu, {"a"}, {"y"}));
    infer_shapes(training);
    CHECK_EQ(remove_identities(training), size_t{0});
}

// TESTS - DEAD CODE

TEST(opt_dead_branch_and_its_weights) {
    // b is computed from the weight w but reaches no output; v is never read at all
    Graph g = chain({Node("r", OpType::Relu, {"x"}, {"y"}), Node("add", OpType::Add, {"x", "w"}, {"b"}),
                     Node("neg", OpType::Relu, {"b"}, {"c"})}, {"y"}, {"w", "v"});
    CHECK_EQ(eliminate_dead_nodes(g), size_t{4});         // two nodes, two initializers
    CHECK(node(g, "add").is_dead && node(g, "neg").is_dead && !node(g, "r").is_dead);
    CHECK(g.initializers.empty());
    CHECK(!g.get_tensor_info("b") && !g.get_tensor_info("c") && !g.get_tensor_info("w"));
    CHECK_EQ(eliminate_dead_nodes(g), size_t{0});
}

TEST(opt_dead_code_keeps_what_is_needed) {
    // a node with one used and one unused output lives; an initializer that is
    // also a graph input stays even when nothing reads it
    Graph g = chain({Node("d", OpType::Dropout, {"x"}, {"y", "mask"})});
    add_weights(g, "w");
    g.inputs.push_back("w");
    CHECK_EQ(eliminate_dead_nodes(g), size_t{0});
    CHECK(g.initializers.count("w"));
}

// TESTS - THE PASS MANAGER

TEST(opt_rounds_until_nothing_changes) {
    // two identities in a row before the output, a dropout, and a dead relu
    // that only the removed identities fed
    Graph g = chain({Node("r", OpType::Relu, {"x"}, {"a"}), Node("i1", OpType::Identity, {"a"}, {"b"}),
                     Node("d", OpType::Dropout, {"b"}, {"c"}), Node("i2", OpType::Identity, {"c"}, {"y"}),
                     Node("lost", OpType::Relu, {"b"}, {"z"})});
    std::vector<PassResult> r = optimize(g);
    CHECK_EQ(r.size(), size_t{5});
    CHECK_EQ(r[0].pass, std::string("fold-constants"));
    CHECK_EQ(r[0].changes, size_t{0});
    CHECK_EQ(r[1].pass, std::string("remove-identities"));
    CHECK_EQ(r[1].changes, size_t{3});
    CHECK_EQ(r[2].pass, std::string("fold-batchnorm"));
    CHECK_EQ(r[2].changes, size_t{0});
    CHECK_EQ(r[3].pass, std::string("fuse-relu"));
    CHECK_EQ(r[3].changes, size_t{0});
    CHECK_EQ(r[4].pass, std::string("dead-code"));
    CHECK_EQ(r[4].changes, size_t{1});
    CHECK_EQ(g.live_node_count(), size_t{1});
    CHECK_EQ(node(g, "r").outputs, (std::vector<std::string>{"y"}));
    CHECK(g.validate().empty());

    // optimising again finds nothing, and shape inference still agrees
    for (const PassResult& again : optimize(g)) CHECK_EQ(again.changes, size_t{0});
    infer_shapes(g);
    CHECK_EQ(g.get_tensor_info("y")->shape, (std::vector<int64_t>{4}));
}

// a pass that removes one Identity per call, so a chain of them takes as many
// rounds as it has links
static size_t remove_one_identity(Graph& g) {
    for (Node& n : g.nodes) {
        if (n.is_dead || n.op != OpType::Identity) continue;
        for (Node& other : g.nodes)
            for (std::string& in : other.inputs)
                if (in == n.outputs[0]) in = n.inputs[0];
        n.is_dead = true;
        return 1;
    }
    return 0;
}

static int g_calls = 0;
static size_t never_done(Graph&) {
    ++g_calls;
    return 1;
}

TEST(opt_pass_manager_rounds) {
    // four identities, one per round, then a round that finds nothing
    Graph g = chain({Node("i1", OpType::Identity, {"x"}, {"a"}), Node("i2", OpType::Identity, {"a"}, {"b"}),
                     Node("i3", OpType::Identity, {"b"}, {"c"}), Node("i4", OpType::Identity, {"c"}, {"d"}),
                     Node("r", OpType::Relu, {"d"}, {"y"})});
    std::vector<PassResult> r = run_passes(g, {{"one-at-a-time", remove_one_identity}});
    CHECK_EQ(r.size(), size_t{1});
    CHECK_EQ(r[0].changes, size_t{4});
    CHECK_EQ(node(g, "r").inputs, (std::vector<std::string>{"x"}));

    // a pass that always claims a change is stopped by the bound
    std::vector<PassResult> stuck = run_passes(g, {{"never-done", never_done}});
    CHECK_EQ(g_calls, kMaxPassRounds);
    CHECK_EQ(stuck[0].changes, static_cast<size_t>(kMaxPassRounds));
}

// TESTS - CONSTANT FOLDING

static void add_floats(Graph& g, const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& v) {
    Tensor t(name, shape, DType::FLOAT32);
    t.set_float_data(v);
    t.shape = shape;
    g.add_initializer(t);
}

static void add_ints(Graph& g, const std::string& name, const std::vector<int64_t>& shape, const std::vector<int64_t>& v) {
    Tensor t(name, shape, DType::INT64);
    t.set_int64_data(v);
    t.shape = shape;
    g.add_initializer(t);
}

static std::vector<float> floats(const Graph& g, const std::string& name) {
    const Tensor& t = g.initializers.at(name);
    std::vector<float> v(static_cast<size_t>(t.num_elements()));
    std::memcpy(v.data(), t.data.data(), v.size() * 4);
    return v;
}

static std::vector<int64_t> ints(const Graph& g, const std::string& name) {
    const Tensor& t = g.initializers.at(name);
    return std::vector<int64_t>(t.int64_ptr(), t.int64_ptr() + t.num_elements());
}

// x[4] -> Add(x, c) -> y, c from a Constant node with the given attribute
static Graph with_constant(Node c) {
    Graph g("c");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    g.add_node(c);
    g.add_node(Node("add", OpType::Add, {"x", "c"}, {"y"}));
    infer_shapes(g);
    return g;
}

TEST(opt_fold_constant_nodes) {
    Node floats4("k", OpType::Constant, {}, {"c"});
    floats4.add_attr("value_floats", std::vector<float>{1, 2, 3, 4});
    Graph g = with_constant(floats4);
    CHECK_EQ(fold_constants(g), size_t{1});
    CHECK(node(g, "k").is_dead);
    CHECK_EQ(floats(g, "c"), (std::vector<float>{1, 2, 3, 4}));
    CHECK_EQ(g.initializers.at("c").shape, (std::vector<int64_t>{4}));

    Node scalar("k", OpType::Constant, {}, {"c"});
    scalar.add_attr("value_float", 0.5f);
    Graph s = with_constant(scalar);
    fold_constants(s);
    CHECK_EQ(s.initializers.at("c").shape, (std::vector<int64_t>{}));
    CHECK_EQ(floats(s, "c"), (std::vector<float>{0.5f}));

    Node tensor("k", OpType::Constant, {}, {"c"});
    Tensor value("whatever", {4}, DType::FLOAT32);
    value.set_float_data({4, 3, 2, 1});
    tensor.add_attr("value", value);
    Graph t = with_constant(tensor);
    fold_constants(t);
    CHECK_EQ(floats(t, "c"), (std::vector<float>{4, 3, 2, 1}));
    CHECK_EQ(t.initializers.at("c").name, std::string("c"));      // named after the output, not the attribute

    // int64 forms, as a Reshape target and a Gather index
    Graph r("r");
    r.inputs = {"x"};
    r.outputs = {"y", "z"};
    r.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    Node shape("k", OpType::Constant, {}, {"shape"});
    shape.add_attr("value_ints", std::vector<int64_t>{2, 2});
    Node index("i", OpType::Constant, {}, {"index"});
    index.add_attr("value_int", int64_t{3});
    r.add_node(shape);
    r.add_node(index);
    r.add_node(Node("reshape", OpType::Reshape, {"x", "shape"}, {"y"}));
    r.add_node(Node("gather", OpType::Gather, {"x", "index"}, {"z"}));
    infer_shapes(r);
    CHECK_EQ(fold_constants(r), size_t{2});
    CHECK_EQ(ints(r, "shape"), (std::vector<int64_t>{2, 2}));
    CHECK_EQ(ints(r, "index"), (std::vector<int64_t>{3}));
    CHECK_EQ(r.initializers.at("index").shape, (std::vector<int64_t>{}));

    // a Constant that is a graph output stays a node: the generated code must produce it
    Graph out("o");
    out.outputs = {"c"};
    Node direct("k", OpType::Constant, {}, {"c"});
    direct.add_attr("value_floats", std::vector<float>{1});
    out.add_node(direct);
    infer_shapes(out);
    CHECK_EQ(fold_constants(out), size_t{0});
}

TEST(opt_fold_float_operations) {
    // y = x + (Relu(a) + b): the bracket is all constants, folded in one pass,
    // in order; then dead code drops a, b and the intermediate
    Graph g("f");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {4}, DType::FLOAT32));
    add_floats(g, "a", {4}, {-1, 2, -3, 4});
    add_floats(g, "b", {1}, {10});
    g.add_node(Node("relu", OpType::Relu, {"a"}, {"r"}));
    g.add_node(Node("plus", OpType::Add, {"r", "b"}, {"s"}));
    g.add_node(Node("add", OpType::Add, {"x", "s"}, {"y"}));
    infer_shapes(g);
    std::vector<PassResult> res = optimize(g);
    CHECK_EQ(res[0].changes, size_t{2});
    CHECK_EQ(floats(g, "s"), (std::vector<float>{10, 12, 10, 14}));
    CHECK_EQ(g.live_node_count(), size_t{1});
    CHECK_EQ(g.initializers.size(), size_t{1});

    // the other operations, each checked against its definition
    Graph t("t");
    t.inputs = {"x"};
    t.outputs = {"y1", "y2", "y3", "y4"};
    t.add_value_info(Tensor("x", {1}, DType::FLOAT32));
    add_floats(t, "m", {2, 3}, {1, 2, 3, 4, 5, 6});
    add_floats(t, "row", {3}, {1, 0, -1});
    add_ints(t, "flat", {1}, {6});
    t.add_node(Node("tr", OpType::Transpose, {"m"}, {"mt"}));                 // [3, 2]
    t.add_node(Node("sub", OpType::Sub, {"m", "row"}, {"ms"}));               // broadcast over rows
    t.add_node(Node("div", OpType::Div, {"row", "row"}, {"ones"}));           // 1, NaN, 1
    t.add_node(Node("rs", OpType::Reshape, {"m", "flat"}, {"mf"}));
    t.add_node(Node("o1", OpType::Mul, {"x", "mt"}, {"y1"}));
    t.add_node(Node("o2", OpType::Mul, {"x", "ms"}, {"y2"}));
    t.add_node(Node("o3", OpType::Mul, {"x", "ones"}, {"y3"}));
    t.add_node(Node("o4", OpType::Mul, {"x", "mf"}, {"y4"}));
    infer_shapes(t);
    CHECK_EQ(fold_constants(t), size_t{4});
    CHECK_EQ(floats(t, "mt"), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    CHECK_EQ(t.initializers.at("mt").shape, (std::vector<int64_t>{3, 2}));
    CHECK_EQ(floats(t, "ms"), (std::vector<float>{0, 2, 4, 3, 5, 7}));
    std::vector<float> ones = floats(t, "ones");
    CHECK(ones[0] == 1 && std::isnan(ones[1]) && ones[2] == 1);
    CHECK_EQ(floats(t, "mf"), (std::vector<float>{1, 2, 3, 4, 5, 6}));
    CHECK_EQ(t.initializers.at("mf").shape, (std::vector<int64_t>{6}));

    // abs, neg and sqrt fold: exact, or correctly rounded
    Graph u("u");
    u.inputs = {"x"};
    u.outputs = {"y1", "y2", "y3"};
    u.add_value_info(Tensor("x", {1}, DType::FLOAT32));
    add_floats(u, "v", {4}, {-4, 2, 0, -0.0f});
    u.add_node(Node("abs", OpType::Abs, {"v"}, {"va"}));
    u.add_node(Node("neg", OpType::Neg, {"v"}, {"vn"}));
    u.add_node(Node("sqrt", OpType::Sqrt, {"va"}, {"vs"}));
    u.add_node(Node("o1", OpType::Mul, {"x", "va"}, {"y1"}));
    u.add_node(Node("o2", OpType::Mul, {"x", "vn"}, {"y2"}));
    u.add_node(Node("o3", OpType::Mul, {"x", "vs"}, {"y3"}));
    infer_shapes(u);
    CHECK_EQ(fold_constants(u), size_t{3});
    std::vector<float> va = floats(u, "va"), vn = floats(u, "vn"), vs = floats(u, "vs");
    CHECK_EQ(va, (std::vector<float>{4, 2, 0, 0}));
    CHECK(!std::signbit(va[3]));                                          // abs(-0) = +0
    CHECK_EQ(vn, (std::vector<float>{4, -2, 0, 0}));
    CHECK(std::signbit(vn[2]) && !std::signbit(vn[3]));                   // neg flips the zeros' signs
    CHECK_EQ(vs, (std::vector<float>{2, std::sqrt(2.0f), 0, 0}));
}

TEST(opt_fold_refuses) {
    Graph g("r");
    g.inputs = {"x", "over"};
    g.outputs = {"y1", "y2", "out", "y4", "y5", "y6", "y7", "y8", "y9"};
    g.add_value_info(Tensor("x", {1}, DType::FLOAT32));
    add_floats(g, "col", {2, 1}, {1, 2});
    add_floats(g, "row", {3}, {1, 2, 3});
    add_floats(g, "over", {3}, {1, 2, 3});                 // an initializer the caller may replace
    add_floats(g, "w", {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    g.add_node(Node("grow", OpType::Add, {"col", "row"}, {"big"}));       // 6 values from 5: not folded
    g.add_node(Node("o1", OpType::Mul, {"x", "big"}, {"y1"}));
    g.add_node(Node("input", OpType::Relu, {"over"}, {"r"}));             // reads a graph input
    g.add_node(Node("o2", OpType::Mul, {"x", "r"}, {"y2"}));
    g.add_node(Node("output", OpType::Relu, {"row"}, {"out"}));           // is a graph output
    g.add_node(Node("mm", OpType::MatMul, {"w", "w"}, {"sq"}));           // sums: left for run time
    g.add_node(Node("o4", OpType::Mul, {"x", "sq"}, {"y4"}));
    // the C library's expf and tanhf are not correctly rounded: the compiler's
    // libm need not give the target's bits, so these too wait for run time
    g.add_node(Node("sig", OpType::Sigmoid, {"row"}, {"s"}));
    g.add_node(Node("o5", OpType::Mul, {"x", "s"}, {"y5"}));
    g.add_node(Node("tanh", OpType::Tanh, {"row"}, {"t"}));
    g.add_node(Node("o6", OpType::Mul, {"x", "t"}, {"y6"}));
    g.add_node(Node("exp", OpType::Exp, {"row"}, {"e"}));                 // expf and logf likewise
    g.add_node(Node("o7", OpType::Mul, {"x", "e"}, {"y7"}));
    g.add_node(Node("log", OpType::Log, {"row"}, {"l"}));
    g.add_node(Node("o8", OpType::Mul, {"x", "l"}, {"y8"}));
    g.add_node(Node("pow", OpType::Pow, {"row", "row"}, {"p"}));          // and powf
    g.add_node(Node("o9", OpType::Mul, {"x", "p"}, {"y9"}));
    infer_shapes(g);
    CHECK_EQ(fold_constants(g), size_t{0});
}

TEST(opt_fold_shape_chain) {
    // x.view(x.size(0), -1) with the batch fixed: the whole chain becomes one
    // int64 initializer, and only the Reshape is left
    Graph g("view");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {-1, 1, 4, 4}, DType::FLOAT32));
    add_ints(g, "zero", {}, {0});
    add_ints(g, "axes", {1}, {0});
    add_ints(g, "rest", {1}, {-1});
    g.add_node(Node("shape", OpType::Shape, {"x"}, {"s"}));
    g.add_node(Node("gather", OpType::Gather, {"s", "zero"}, {"batch"}));
    g.add_node(Node("unsqueeze", OpType::Unsqueeze, {"batch", "axes"}, {"batch1"}));
    Node cat("concat", OpType::Concat, {"batch1", "rest"}, {"target"});
    cat.add_attr("axis", int64_t{0});
    g.add_node(cat);
    g.add_node(Node("reshape", OpType::Reshape, {"x", "target"}, {"y"}));
    set_batch_size(g, 1);
    infer_shapes(g);
    std::vector<PassResult> res = optimize(g);
    CHECK_EQ(res[0].changes, size_t{4});
    CHECK_EQ(g.live_node_count(), size_t{1});
    CHECK_EQ(ints(g, "target"), (std::vector<int64_t>{1, -1}));
    CHECK_EQ(g.initializers.size(), size_t{1});
    CHECK(g.validate().empty());
    // and the graph still infers to the same shapes
    infer_shapes(g);
    CHECK_EQ(g.get_tensor_info("y")->shape, (std::vector<int64_t>{1, 16}));
}

// TESTS - FUSING RELU

// x[1, 4] -> Gemm(w [3, 4], transB) -> h -> Relu -> y, plus extra nodes
static Graph gemm_relu(const std::vector<Node>& extra = {}, const std::vector<std::string>& outputs = {"y"}) {
    Graph g("fuse");
    g.inputs = {"x"};
    g.outputs = outputs;
    g.add_value_info(Tensor("x", {1, 4}, DType::FLOAT32));
    Tensor w("w", {3, 4}, DType::FLOAT32);
    w.set_float_data(std::vector<float>(12, 0.5f));
    g.add_initializer(w);
    Node gemm("gemm", OpType::Gemm, {"x", "w"}, {"h"});
    gemm.add_attr("transB", int64_t{1});
    g.add_node(gemm);
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"y"}));
    for (const Node& n : extra) g.add_node(n);
    infer_shapes(g);
    return g;
}

TEST(opt_fuse_relu_into_gemm) {
    Graph g = gemm_relu();
    CHECK_EQ(fuse_relu(g), size_t{1});
    CHECK_EQ(node(g, "gemm").attr_int(kFusedRelu, 0), int64_t{1});
    CHECK_EQ(node(g, "gemm").outputs, (std::vector<std::string>{"y"}));
    CHECK(node(g, "relu").is_dead);
    CHECK(!g.get_tensor_info("h"));
    CHECK(g.validate().empty());
    CHECK_EQ(fuse_relu(g), size_t{0});
}

TEST(opt_fuse_relu_only_when_alone) {
    // the value before the Relu read by someone else, or a graph output: no fusion
    Graph shared = gemm_relu({Node("other", OpType::Add, {"h", "h"}, {"z"})}, {"y", "z"});
    CHECK_EQ(fuse_relu(shared), size_t{0});
    Graph output = gemm_relu({}, {"y", "h"});
    CHECK_EQ(fuse_relu(output), size_t{0});
    // a second Relu after a fused one stays a Relu
    Graph twice = gemm_relu({Node("again", OpType::Relu, {"y"}, {"z"})}, {"z"});
    CHECK_EQ(fuse_relu(twice), size_t{1});
    CHECK(!node(twice, "again").is_dead);
    // MatMul has no fused form
    Graph mm("mm");
    mm.inputs = {"x"};
    mm.outputs = {"y"};
    mm.add_value_info(Tensor("x", {2, 2}, DType::FLOAT32));
    mm.add_node(Node("mm", OpType::MatMul, {"x", "x"}, {"h"}));
    mm.add_node(Node("relu", OpType::Relu, {"h"}, {"y"}));
    infer_shapes(mm);
    CHECK_EQ(fuse_relu(mm), size_t{0});
}

TEST(opt_fuse_relu_into_conv_through_an_identity) {
    // Conv -> Identity -> Relu: identities go first, then the Relu fuses
    Graph g("conv");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {1, 1, 3, 3}, DType::FLOAT32));
    Tensor w("w", {2, 1, 2, 2}, DType::FLOAT32);
    w.set_float_data(std::vector<float>(8, -0.25f));
    g.add_initializer(w);
    g.add_node(Node("conv", OpType::Conv, {"x", "w"}, {"c"}));
    g.add_node(Node("id", OpType::Identity, {"c"}, {"c2"}));
    g.add_node(Node("relu", OpType::Relu, {"c2"}, {"y"}));
    infer_shapes(g);
    std::vector<PassResult> r = optimize(g);
    CHECK_EQ(r[1].changes, size_t{1});                  // the Identity
    CHECK_EQ(r[3].changes, size_t{1});                  // the Relu
    CHECK_EQ(g.live_node_count(), size_t{1});
    CHECK_EQ(node(g, "conv").attr_int(kFusedRelu, 0), int64_t{1});
    CHECK_EQ(node(g, "conv").outputs, (std::vector<std::string>{"y"}));
}

// x[1, 1, 3, 3] -> Conv(w[2, 1, 2, 2], b?) -> c -> BatchNormalization -> y
static Graph conv_bn(bool with_bias) {
    Graph g("cbn");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {1, 1, 3, 3}, DType::FLOAT32));
    add_floats(g, "w", {2, 1, 2, 2}, {1, 2, 3, 4, -1, -2, -3, -4});
    if (with_bias) add_floats(g, "b", {2}, {0.5f, -0.5f});
    add_floats(g, "scale", {2}, {2.0f, 0.5f});
    add_floats(g, "beta", {2}, {1.0f, -1.0f});
    add_floats(g, "mean", {2}, {0.25f, 3.0f});
    add_floats(g, "var", {2}, {4.0f, 0.25f});
    g.add_node(Node("conv", OpType::Conv,
                    with_bias ? std::vector<std::string>{"x", "w", "b"} : std::vector<std::string>{"x", "w"}, {"c"}));
    Node bn("bn", OpType::BatchNormalization, {"c", "scale", "beta", "mean", "var"}, {"y"});
    bn.add_attr("epsilon", 0.0f);
    g.add_node(bn);
    infer_shapes(g);
    return g;
}

TEST(opt_fold_batchnorm) {
    // a = scale / sqrt(var) = {1, 1}; then W' = W, B' = (b - mean) * a + beta
    Graph g = conv_bn(true);
    CHECK_EQ(fold_batchnorm(g), size_t{1});
    CHECK(node(g, "bn").is_dead);
    const Node conv = node(g, "conv");
    CHECK_EQ(conv.outputs, (std::vector<std::string>{"y"}));
    CHECK_EQ(floats(g, conv.inputs[1]), (std::vector<float>{1, 2, 3, 4, -1, -2, -3, -4}));
    CHECK_EQ(floats(g, conv.inputs[2]), (std::vector<float>{0.5f - 0.25f + 1.0f, -0.5f - 3.0f - 1.0f}));
    CHECK(g.get_tensor_info("c") == nullptr);
    // the original weights are untouched: another node may read them
    CHECK_EQ(floats(g, "w"), (std::vector<float>{1, 2, 3, 4, -1, -2, -3, -4}));
    CHECK(conv.inputs[1] != "w");

    // scales that are not 1, in double and rounded once; a Conv with no bias gets one
    Graph s = conv_bn(false);
    s.initializers.at("var").set_float_data({1.0f, 9.0f});
    s.initializers.at("scale").set_float_data({3.0f, 1.0f});
    CHECK_EQ(fold_batchnorm(s), size_t{1});
    const Node c2 = node(s, "conv");
    CHECK_EQ(c2.inputs.size(), size_t{3});
    std::vector<float> w2 = floats(s, c2.inputs[1]), b2 = floats(s, c2.inputs[2]);
    CHECK_EQ(w2[0], 3.0f);
    CHECK_EQ(w2[3], 12.0f);
    CHECK_EQ(w2[4], static_cast<float>(-1.0 / 3.0));
    CHECK_EQ(b2, (std::vector<float>{-0.25f * 3 + 1.0f, static_cast<float>(-3.0 / 3.0 - 1.0)}));

    // in the full optimiser a Relu after the BatchNorm then fuses into the Conv
    Graph r = conv_bn(true);
    r.outputs = {"z"};
    r.add_node(Node("relu", OpType::Relu, {"y"}, {"z"}));
    infer_shapes(r);
    std::vector<PassResult> res = optimize(r);
    CHECK_EQ(res[2].changes, size_t{1});
    CHECK_EQ(res[3].changes, size_t{1});
    CHECK_EQ(r.live_node_count(), size_t{1});
    CHECK_EQ(node(r, "conv").outputs, (std::vector<std::string>{"z"}));
}

TEST(opt_fold_batchnorm_refuses) {
    // the Conv's output read by another node too
    Graph shared = conv_bn(true);
    shared.outputs = {"y", "z"};
    shared.add_node(Node("other", OpType::Relu, {"c"}, {"z"}));
    infer_shapes(shared);
    CHECK_EQ(fold_batchnorm(shared), size_t{0});
    // the Conv's output a graph output
    Graph out = conv_bn(true);
    out.outputs = {"y", "c"};
    CHECK_EQ(fold_batchnorm(out), size_t{0});
    // a parameter, or the weights, the caller may replace
    Graph param = conv_bn(true);
    param.inputs.push_back("mean");
    CHECK_EQ(fold_batchnorm(param), size_t{0});
    Graph weights = conv_bn(true);
    weights.inputs.push_back("w");
    CHECK_EQ(fold_batchnorm(weights), size_t{0});
    // not after a Conv
    Graph relu("r");
    relu.inputs = {"x"};
    relu.outputs = {"y"};
    relu.add_value_info(Tensor("x", {1, 2, 3}, DType::FLOAT32));
    for (const char* n : {"scale", "beta", "mean", "var"}) add_floats(relu, n, {2}, {1, 1});
    relu.add_node(Node("relu", OpType::Relu, {"x"}, {"r"}));
    relu.add_node(Node("bn", OpType::BatchNormalization, {"r", "scale", "beta", "mean", "var"}, {"y"}));
    infer_shapes(relu);
    CHECK_EQ(fold_batchnorm(relu), size_t{0});
}
