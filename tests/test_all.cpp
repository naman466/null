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
