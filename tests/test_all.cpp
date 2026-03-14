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
