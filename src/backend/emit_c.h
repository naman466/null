#pragma once
#include <stdexcept>
#include <string>
#include <vector>

#include "core/graph.h"

namespace null {

// c code generation
// turns a graph whose every tensor has a known static shape (run shape
// inference and fix dynamic inputs first) into one self-contained C99
// source file defining
//
//   void model_run(const float* <input>..., float* <output>...);
//
// with the weights as static const arrays and the runtime kernels embedded.

class EmitError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct EmitOptions {
    // paste the runtime header into the output, so model.c needs nothing else
    bool embed_runtime = true;
    // name of the generated entry point
    std::string function_name = "model_run";
};

// the C source, or EmitError naming the first thing that can't be emitted
std::string emit_c(const Graph& g, const EmitOptions& options = {});

// one part of the runtime, runtime/null/<name>.h, embedded at build time
struct RuntimePart {
    const char* name;   // "core", "elementwise", ...
    const char* text;   // the file's contents
};

// every part, in the order runtime/null_runtime.h includes them, core first
const std::vector<RuntimePart>& runtime_parts();

// exact C99 literal for a float: a hex float built from its bits (so it
// round-trips exactly), or NAN / INFINITY / -INFINITY
std::string c_float_literal(float v);

} // namespace null
