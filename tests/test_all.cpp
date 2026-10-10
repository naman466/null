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