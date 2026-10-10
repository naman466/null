#include "test_framework.h"
#include "fixtures.h"
#include "core/graph.h"

using namespace null;
using namespace null_test;

TEST(graph_initializer_lookup) {
    Graph g = make_simple_matmul_graph();
    CHECK(g.is_initializer("W"));
    CHECK(g.is_initializer("b"));
    CHECK(!g.is_initializer("x"));

    // initializers are also registered as value info
    const Tensor* w = g.get_tensor_info("W");
    CHECK(w != nullptr);
    CHECK(w->is_constant);
    CHECK_EQ(w->num_elements(), 8);
    CHECK(g.get_tensor_info("missing") == nullptr);
}

TEST(graph_input_lookup) {
    Graph g = make_simple_matmul_graph();
    CHECK(g.is_graph_input("x"));
    CHECK(!g.is_graph_input("W"));
    CHECK(!g.is_graph_input("y"));
}

TEST(graph_topo_sort_in_order) {
    Graph g = make_simple_matmul_graph();
    auto order = g.topological_sort();
    CHECK_EQ(order, (std::vector<size_t>{0, 1}));
}

TEST(graph_topo_sort_reorders) {
    // consumer added before producer
    Graph g("reversed");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_node(Node("relu", OpType::Relu, {"h"}, {"y"}));
    g.add_node(Node("neg",  OpType::Neg,  {"x"}, {"h"}));
    auto order = g.topological_sort();
    CHECK_EQ(order, (std::vector<size_t>{1, 0}));
}

TEST(graph_topo_sort_stable) {
    // n1 and n2 are independent and keep their original order,
    // n0 depends on n2 so it runs last
    Graph g("stable");
    g.inputs = {"x"};
    g.outputs = {"u", "s"};
    g.add_node(Node("n0", OpType::Relu, {"t"}, {"u"}));
    g.add_node(Node("n1", OpType::Neg,  {"x"}, {"s"}));
    g.add_node(Node("n2", OpType::Abs,  {"x"}, {"t"}));
    auto order = g.topological_sort();
    CHECK_EQ(order, (std::vector<size_t>{1, 2, 0}));
}

TEST(graph_topo_sort_diamond) {
    // x -> a, x -> b, (a, b) -> c
    Graph g("diamond");
    g.inputs = {"x"};
    g.outputs = {"c"};
    g.add_node(Node("c", OpType::Add,  {"a", "b"}, {"c"}));
    g.add_node(Node("a", OpType::Relu, {"x"}, {"a"}));
    g.add_node(Node("b", OpType::Neg,  {"x"}, {"b"}));
    auto order = g.topological_sort();
    CHECK_EQ(order, (std::vector<size_t>{1, 2, 0}));
}

TEST(graph_topo_sort_cycle_throws) {
    Graph g("cycle");
    g.add_node(Node("a", OpType::Relu, {"q"}, {"p"}));
    g.add_node(Node("b", OpType::Relu, {"p"}, {"q"}));
    CHECK_THROWS(g.topological_sort());
}

TEST(graph_validate_ok) {
    CHECK(make_simple_matmul_graph().validate().empty());
    CHECK(make_relu_graph().validate().empty());
}

TEST(graph_validate_undefined_input) {
    Graph g = make_relu_graph();
    g.add_node(Node("bad", OpType::Add, {"y", "ghost"}, {"z"}));
    auto errors = g.validate();
    CHECK_EQ(errors.size(), 1u);
    CHECK(errors[0].find("ghost") != std::string::npos);
}

TEST(graph_validate_undefined_output) {
    Graph g = make_relu_graph();
    g.outputs.push_back("nowhere");
    auto errors = g.validate();
    CHECK_EQ(errors.size(), 1u);
    CHECK(errors[0].find("nowhere") != std::string::npos);
}

TEST(graph_validate_allows_empty_optional_input) {
    // onnx uses "" for omitted optional inputs
    Graph g = make_relu_graph();
    g.add_node(Node("clip", OpType::Clip, {"y", "", ""}, {"z"}));
    CHECK(g.validate().empty());
}

TEST(graph_validate_reports_cycle) {
    Graph g("cycle");
    g.add_node(Node("a", OpType::Relu, {"q"}, {"p"}));
    g.add_node(Node("b", OpType::Relu, {"p"}, {"q"}));
    auto errors = g.validate();
    CHECK_EQ(errors.size(), 1u);
    CHECK(errors[0].find("Cycle") != std::string::npos);
}

TEST(graph_sorted_nodes_skip_dead) {
    Graph g = make_simple_matmul_graph();
    g.add_node(Node("unused", OpType::Relu, {"x"}, {"junk"}));
    g.nodes.back().is_dead = true;

    CHECK_EQ(g.nodes.size(), 3u);
    CHECK_EQ(g.live_node_count(), 2u);

    auto sorted = g.sorted_nodes();
    CHECK_EQ(sorted.size(), 2u);
    CHECK_EQ(sorted[0]->name, "mm_node");
    CHECK_EQ(sorted[1]->name, "add_node");
}

TEST(graph_summary) {
    std::string s = make_simple_matmul_graph().summary();
    CHECK(s.find("test_matmul") != std::string::npos);
    CHECK(s.find("Nodes: 2 (live: 2)") != std::string::npos);
    CHECK(s.find("Initializers: 2") != std::string::npos);
}
