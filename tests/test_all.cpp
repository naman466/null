#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <cmath>

#include "../src/core/tensor.h"
#include "../src/core/node.h"
#include "../src/core/graph.h"

using namespace null;

// test framework

static int s_pass = 0, s_fail = 0;
static std::string s_current_suite;
static std::vector<std::string> s_failures;

#define TEST(name) static void test_##name()
#define RUN(name) do { \
    try { test_##name(); s_pass++; std::cout << "  [PASS] " #name "\n"; } \
    catch (std::exception& e) { \
        s_fail++; \
        std::string msg = "  [FAIL] " #name " -- " + std::string(e.what()); \
        std::cout << msg << "\n"; \
        s_failures.push_back(msg); \
    } \
} while(0)

#define SUITE(name) do { \
    s_current_suite = name; \
    std::cout << "\n=== " << name << " ===\n"; \
} while(0)

#define CHECK(cond) do { \
    if (!(cond)) throw std::runtime_error("CHECK failed: " #cond " at line " + std::to_string(__LINE__)); \
} while(0)

template<typename A, typename B>
inline void check_eq_impl(const A& a, const B& b, const char* expr, int line) {
    if (!(a == b)) {
        std::ostringstream _ss;
        _ss << "CHECK_EQ failed at line " << line << ": " << expr;
        throw std::runtime_error(_ss.str());
    }
}
#define CHECK_EQ(a, b) check_eq_impl((a), (b), #a " == " #b, __LINE__)

#define CHECK_NEAR(a, b, eps) do { \
    float _a = (float)(a), _b = (float)(b); \
    if (std::fabs(_a - _b) > (eps)) { \
        std::ostringstream _ss; \
        _ss << "CHECK_NEAR failed: " << _a << " vs " << _b \
            << " (tol=" << (eps) << ") at line " << __LINE__; \
        throw std::runtime_error(_ss.str()); \
    } \
} while(0)

#define CHECK_THROWS(expr) do { \
    bool _threw = false; \
    try { expr; } catch(...) { _threw = true; } \
    if (!_threw) throw std::runtime_error("CHECK_THROWS: no exception at line " + std::to_string(__LINE__)); \
} while(0)

// TESTS -- GRAPH

static Graph make_simple_matmul_graph() {
    // input: x[1,4], weight: W[4,2], bias: b[2] -> output: y[1,2]
    Graph g("test_matmul");
    g.inputs = {"x"};
    g.outputs = {"y"};

    // value info
    g.add_value_info(Tensor{"x",  {1, 4}, DType::FLOAT32});
    g.add_value_info(Tensor{"mm", {1, 2}, DType::FLOAT32});
    g.add_value_info(Tensor{"y",  {1, 2}, DType::FLOAT32});

    // initializers
    Tensor W{"W", {4, 2}, DType::FLOAT32};
    W.set_float_data({1,0, 0,1, 1,0, 0,1});
    W.shape = {4, 2};
    g.add_initializer(W);

    Tensor b{"b", {2}, DType::FLOAT32};
    b.set_float_data({0.5f, -0.5f});
    g.add_initializer(b);

    // matmul node
    Node mm("mm_node", OpType::MatMul, {"x", "W"}, {"mm"});
    g.add_node(mm);

    // add bias
    Node add("add_node", OpType::Add, {"mm", "b"}, {"y"});
    g.add_node(add);

    return g;
}

static Graph make_relu_graph() {
    Graph g("test_relu");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor{"x", {1, 4}, DType::FLOAT32});
    g.add_value_info(Tensor{"y", {1, 4}, DType::FLOAT32});
    Node relu("relu", OpType::Relu, {"x"}, {"y"});
    g.add_node(relu);
    return g;
}

// TESTS - IR

TEST(tensor_basic_construction) {
    Tensor t("t", {2, 3}, DType::FLOAT32);
    CHECK_EQ(t.name, "t");
    CHECK_EQ(t.shape.size(), 2u);
    CHECK_EQ(t.shape[0], 2);
    CHECK_EQ(t.shape[1], 3);
    CHECK_EQ(t.dtype, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), 6);
    CHECK_EQ(t.byte_size(), 24);
}

TEST(tensor_scalar) {
    Tensor t("s", {}, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), 1);
    CHECK_EQ(t.byte_size(), 4);
}

TEST(tensor_dynamic_dim) {
    Tensor t("d", {-1, 3}, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), -1);
    CHECK_EQ(t.byte_size(), -1);
}

TEST(tensor_set_float_data) {
    Tensor t;
    t.name = "w";
    t.shape = {2, 2};
    t.set_float_data({1.0f, 2.0f, 3.0f, 4.0f});
    CHECK(t.is_constant);
    CHECK(t.float_ptr() != nullptr);
    CHECK_NEAR(t.float_ptr()[0], 1.0f, 1e-6f);
    CHECK_NEAR(t.float_ptr()[3], 4.0f, 1e-6f);
}

// TESTS - NODE / ATTRIBUTE

TEST(attribute_typed_getters) {
    CHECK_EQ(Attribute("i", int64_t{3}).as_int(), 3);
    CHECK_NEAR(Attribute("f", 0.5f).as_float(), 0.5f, 1e-6f);
    CHECK_EQ(Attribute("s", std::string("NOTSET")).as_string(), "NOTSET");
    CHECK_EQ(Attribute("is", std::vector<int64_t>{1, 2}).as_ints(),
             (std::vector<int64_t>{1, 2}));
    CHECK_EQ(Attribute("fs", std::vector<float>{1.5f}).as_floats(),
             (std::vector<float>{1.5f}));
}

TEST(attribute_tensor_value) {
    Tensor t("value", {2}, DType::FLOAT32);
    t.set_float_data({1.0f, 2.0f});
    Attribute a("value", t);
    CHECK(a.has_tensor());
    CHECK_EQ(a.as_tensor().num_elements(), 2);
    CHECK_NEAR(a.as_tensor().float_ptr()[1], 2.0f, 1e-6f);
}

TEST(attribute_has_type) {
    Attribute a("axis", int64_t{1});
    CHECK(a.has_int());
    CHECK(!a.has_float());
    CHECK(!a.has_string());
    CHECK(!a.has_ints());
    CHECK(!a.has_floats());
    CHECK(!a.has_tensor());
}

TEST(attribute_wrong_type_throws) {
    Attribute a("alpha", 1.0f);
    CHECK_THROWS(a.as_int());
    CHECK_THROWS(a.as_string());
    CHECK_THROWS(a.as_ints());
    CHECK_THROWS(a.as_floats());
    CHECK_THROWS(a.as_tensor());
}

TEST(node_construction) {
    Node n("gemm", OpType::Gemm, {"a", "b", "c"}, {"y"});
    CHECK_EQ(n.name, "gemm");
    CHECK_EQ(n.op, OpType::Gemm);
    CHECK_EQ(n.inputs.size(), 3u);
    CHECK_EQ(n.outputs, (std::vector<std::string>{"y"}));
    CHECK(!n.is_dead);
    CHECK(n.attrs.empty());

    Node empty;
    CHECK_EQ(empty.op, OpType::Unknown);
}

TEST(node_attr_present) {
    Node n("conv", OpType::Conv, {"x", "w"}, {"y"});
    n.add_attr("group", int64_t{2});
    n.add_attr("epsilon", 1e-5f);
    n.add_attr("auto_pad", std::string("SAME_UPPER"));
    n.add_attr("pads", std::vector<int64_t>{1, 1, 1, 1});
    n.add_attr("scales", std::vector<float>{2.0f, 2.0f});

    CHECK(n.has_attr("group"));
    CHECK_EQ(n.attr_int("group"), 2);
    CHECK_NEAR(n.attr_float("epsilon"), 1e-5f, 1e-9f);
    CHECK_EQ(n.attr_str("auto_pad"), "SAME_UPPER");
    CHECK_EQ(n.attr_ints("pads"), (std::vector<int64_t>{1, 1, 1, 1}));
    CHECK_EQ(n.attr_floats("scales"), (std::vector<float>{2.0f, 2.0f}));
    CHECK_EQ(n.get_attr("group").name, "group");
}

TEST(node_attr_int_literal_is_int) {
    // a plain int literal must select the INT alternative, not FLOAT
    Node n("softmax", OpType::Softmax, {"x"}, {"y"});
    n.add_attr("axis", 1);
    CHECK(n.get_attr("axis").has_int());
    CHECK_EQ(n.attr_int("axis"), 1);
}

TEST(node_attr_defaults) {
    Node n("gemm", OpType::Gemm, {"a", "b"}, {"y"});
    CHECK(!n.has_attr("alpha"));
    CHECK_EQ(n.attr_int("transA"), 0);
    CHECK_EQ(n.attr_int("transB", 1), 1);
    CHECK_NEAR(n.attr_float("alpha", 1.0f), 1.0f, 1e-6f);
    CHECK_EQ(n.attr_str("mode", "constant"), "constant");
    CHECK(n.attr_ints("pads").empty());
    CHECK_EQ(n.attr_floats("scales", {1.0f}), (std::vector<float>{1.0f}));
}

TEST(node_get_attr_missing_throws) {
    Node n("relu", OpType::Relu, {"x"}, {"y"});
    CHECK_THROWS(n.get_attr("alpha"));
}

TEST(node_attr_wrong_type_throws) {
    Node n("gemm", OpType::Gemm, {"a", "b"}, {"y"});
    n.add_attr("alpha", 1.0f);
    // present but wrong type throws instead of falling back to the default
    CHECK_THROWS(n.attr_int("alpha", 7));
    CHECK_THROWS(n.attr_str("alpha"));
}

TEST(optype_round_trip) {
    // every onnx op maps to its name and back
    for (int i = static_cast<int>(OpType::Add); i <= static_cast<int>(OpType::Identity); ++i) {
        OpType op = static_cast<OpType>(i);
        if (op == OpType::ReluFused) continue;
        std::string s = optype_to_str(op);
        CHECK(s != "Unknown");
        if (str_to_optype(s) != op)
            throw std::runtime_error("optype round trip failed for " + s);
    }
}

TEST(optype_unknown_and_internal) {
    CHECK_EQ(str_to_optype("LSTM"), OpType::Unknown);
    CHECK_EQ(str_to_optype("relu"), OpType::Unknown);   // names are case sensitive
    CHECK_EQ(optype_to_str(OpType::Unknown), "Unknown");
    // fused ops are optimizer-only and never parsed from a model
    CHECK_EQ(optype_to_str(OpType::ReluFused), "ReluFused");
    CHECK_EQ(str_to_optype("ReluFused"), OpType::Unknown);
}

// TESTS - GRAPH

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

int main() {
    std::cout << "Starting tests.................\n";

    SUITE("Tensor IR");
    RUN(tensor_basic_construction);
    RUN(tensor_scalar);
    RUN(tensor_dynamic_dim);
    RUN(tensor_set_float_data);

    SUITE("Node / Attribute");
    RUN(attribute_typed_getters);
    RUN(attribute_tensor_value);
    RUN(attribute_has_type);
    RUN(attribute_wrong_type_throws);
    RUN(node_construction);
    RUN(node_attr_present);
    RUN(node_attr_int_literal_is_int);
    RUN(node_attr_defaults);
    RUN(node_get_attr_missing_throws);
    RUN(node_attr_wrong_type_throws);
    RUN(optype_round_trip);
    RUN(optype_unknown_and_internal);

    SUITE("Graph IR");
    RUN(graph_initializer_lookup);
    RUN(graph_input_lookup);
    RUN(graph_topo_sort_in_order);
    RUN(graph_topo_sort_reorders);
    RUN(graph_topo_sort_stable);
    RUN(graph_topo_sort_diamond);
    RUN(graph_topo_sort_cycle_throws);
    RUN(graph_validate_ok);
    RUN(graph_validate_undefined_input);
    RUN(graph_validate_undefined_output);
    RUN(graph_validate_allows_empty_optional_input);
    RUN(graph_validate_reports_cycle);
    RUN(graph_sorted_nodes_skip_dead);
    RUN(graph_summary);

    // results
    std::cout << "Results: " << s_pass << " passed, " << s_fail << " failed\n";

    if (!s_failures.empty()) {
        std::cout << "\nFailures:\n";
        for (const auto& f : s_failures) {
            std::cout << f << "\n";
        }
    }
    return s_fail == 0 ? 0 : 1;
}