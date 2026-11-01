#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "core/graph.h"
#include "opt/optimize.h"

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
    CHECK_EQ(r.size(), size_t{2});
    CHECK_EQ(r[0].pass, std::string("remove-identities"));
    CHECK_EQ(r[0].changes, size_t{3});
    CHECK_EQ(r[1].pass, std::string("dead-code"));
    CHECK_EQ(r[1].changes, size_t{1});
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
