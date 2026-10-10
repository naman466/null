#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include "test_framework.h"

#ifndef NULL_TEST_SUITE
#define NULL_TEST_SUITE "tests"
#endif

// usage: test_<suite> [filter]
// runs every registered test, or only those whose name contains filter
int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";

    int pass = 0, fail = 0;
    std::vector<std::string> failures;

    std::cout << "=== " << NULL_TEST_SUITE << " ===\n";

    for (const auto& t : null_test::registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos)
            continue;
        try {
            t.fn();
            pass++;
            std::cout << "  [PASS] " << t.name << "\n";
        } catch (std::exception& e) {
            fail++;
            std::string msg = "  [FAIL] " + std::string(t.name) + " -- " + e.what();
            std::cout << msg << "\n";
            failures.push_back(msg);
        }
    }

    std::cout << "Results: " << pass << " passed, " << fail << " failed\n";

    if (!failures.empty()) {
        std::cout << "\nFailures:\n";
        for (const auto& f : failures) {
            std::cout << f << "\n";
        }
    }
    return fail == 0 ? 0 : 1;
}
