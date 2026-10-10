#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <cmath>

// minimal unit testing framework
// tests register themselves at static init, test_main.cpp runs them

namespace null_test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) {
        registry().push_back({name, fn});
    }
};

} // namespace null_test

// defines a test and registers it, tests run in declaration order
#define TEST(name) \
    static void test_##name(); \
    static null_test::Registrar s_registrar_##name(#name, &test_##name); \
    static void test_##name()

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
