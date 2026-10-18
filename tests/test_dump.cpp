#include "test_framework.h"
#include "fixtures.h"
#include "core/dump.h"

#include <string>
#include <vector>

using namespace null;
using namespace null_test;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

// TESTS - FORMATTING HELPERS

TEST(dump_dims_and_types) {
    CHECK_EQ(dims_str({}), "[]");
    CHECK_EQ(dims_str({1, -1, 28}), "[1, ?, 28]");
    CHECK_EQ(type_str(Tensor("t", {2, 3}, DType::INT64)), "int64[2, 3]");
    CHECK_EQ(type_str(Tensor("s", {}, DType::BOOL)), "bool[]");
    CHECK_EQ(dtype_name(DType::FLOAT32), "float32");
    CHECK_EQ(dtype_name(DType::UNDEFINED), "undefined");
}

TEST(dump_numbers) {
    CHECK_EQ(group_digits(0), "0");
    CHECK_EQ(group_digits(999), "999");
    CHECK_EQ(group_digits(1000), "1,000");
    CHECK_EQ(group_digits(25557032), "25,557,032");
    CHECK_EQ(count_str(1, "node"), "1 node");
    CHECK_EQ(count_str(0, "node"), "0 nodes");
    CHECK_EQ(count_str(1500, "parameter"), "1,500 parameters");
    CHECK_EQ(bytes_str(40), "40 B");
    CHECK_EQ(bytes_str(1536), "1.5 KB");
    CHECK_EQ(bytes_str(100 * 1024 * 1024), "100.0 MB");
}

TEST(dump_attribute_values) {
    CHECK_EQ(attr_str(Attribute("a", int64_t{-1})), "-1");
    CHECK_EQ(attr_str(Attribute("a", 0.5f)), "0.5");
    CHECK_EQ(attr_str(Attribute("a", 1e-5f)), "1e-05");
    CHECK_EQ(attr_str(Attribute("a", std::string("SAME_UPPER"))), "\"SAME_UPPER\"");
    CHECK_EQ(attr_str(Attribute("a", std::vector<int64_t>{1, 1, 2, 2})), "[1, 1, 2, 2]");
    CHECK_EQ(attr_str(Attribute("a", std::vector<int64_t>{})), "[]");
    CHECK_EQ(attr_str(Attribute("a", std::vector<float>{2.0f, 0.25f})), "[2, 0.25]");
    Tensor t("v", {4}, DType::INT64);
    t.set_int64_data({1, 3, 224, 224});
    CHECK_EQ(attr_str(Attribute("a", t)), "tensor int64[4]");
}

// TESTS - GRAPH DUMP

TEST(dump_full_graph) {
    // the whole output for a known graph, so format changes are deliberate
    std::string expected =
        "graph 'test_matmul': 2 nodes, 2 initializers, 10 parameters (40 B)\n"
        "\n"
        "inputs\n"
        "  x  float32[1, 4]\n"
        "\n"
        "outputs\n"
        "  y  float32[1, 2]\n"
        "\n"
        "initializers\n"
        "  W  float32[4, 2]  32 B\n"
        "  b  float32[2]     8 B\n"
        "\n"
        "nodes\n"
        "  0  mm_node: MatMul(x, W) -> mm: float32[1, 2]\n"
        "  1  add_node: Add(mm, b) -> y: float32[1, 2]\n";
    CHECK_EQ(dump_graph(make_simple_matmul_graph()), expected);
}

TEST(dump_nodes_in_execution_order) {
    // nodes stored consumer first are printed producer first
    Graph g("reversed");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_node(Node("second", OpType::Relu, {"h"}, {"y"}));
    g.add_node(Node("first",  OpType::Neg,  {"x"}, {"h"}));
    std::string s = dump_graph(g);
    CHECK(s.find("0  first: Neg(x) -> h") < s.find("1  second: Relu(h) -> y"));
}

TEST(dump_attributes_and_optionals) {
    Graph g("attrs");
    g.inputs = {"x"};
    g.outputs = {"y"};
    Node clip("clip", OpType::Clip, {"x", "", "max"}, {"y", ""});
    clip.add_attr("pads", std::vector<int64_t>{1, 1});
    clip.add_attr("alpha", 0.5f);
    g.add_node(clip);
    std::string s = dump_graph(g);
    // empty optional names print as _, attributes sorted by name
    CHECK(contains(s, "clip: Clip(x, _, max) -> y, _  {alpha=0.5, pads=[1, 1]}"));
}

TEST(dump_unknown_metadata_and_dead_nodes) {
    Graph g = make_relu_graph();
    g.inputs.push_back("mystery");   // no value info
    g.add_node(Node("gone", OpType::Neg, {"x"}, {"junk"}));
    g.nodes.back().is_dead = true;
    std::string s = dump_graph(g);
    CHECK(contains(s, "  mystery  (unknown)\n"));
    CHECK(contains(s, "1 node, 0 initializers"));
    CHECK(!contains(s, "gone"));
    CHECK(!contains(s, "\ninitializers\n"));   // section omitted when empty
}

TEST(dump_dynamic_dims) {
    Graph g("dyn");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor("x", {-1, 3, 224, 224}, DType::FLOAT32));
    g.add_value_info(Tensor("y", {-1, 1000}, DType::FLOAT32));
    g.add_node(Node("r", OpType::Relu, {"x"}, {"y"}));
    std::string s = dump_graph(g);
    CHECK(contains(s, "  x  float32[?, 3, 224, 224]\n"));
    CHECK(contains(s, "-> y: float32[?, 1000]"));
}
