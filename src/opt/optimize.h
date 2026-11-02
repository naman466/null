#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "core/graph.h"

namespace null {

// the optimiser
// passes that rewrite the graph without changing what it computes. they run
// after shape inference, so a pass may rely on every tensor's type and shape,
// and they leave value_info describing the tensors that remain. a removed node
// is marked dead rather than erased, like everywhere else in the IR

// what one pass did, summed over every round
struct PassResult {
    std::string pass;
    size_t changes = 0;
};

// a pass rewrites the graph in place and returns how many changes it made
struct Pass {
    const char* name;
    size_t (*run)(Graph&);
};

// a bound on the rounds, far more than real passes need: each round that
// changes something removes at least one node or initializer. it only guards
// against a pass that never stops finding work
constexpr int kMaxPassRounds = 16;

// runs the passes in order, and again, until a whole round changes nothing
std::vector<PassResult> run_passes(Graph& g, const std::vector<Pass>& passes);

// run_passes with every pass below, identities first so the nodes they
// bypass can die
std::vector<PassResult> optimize(Graph& g);

// the passes, each returning how many changes it made

// computes at compile time what does not depend on the inputs, turning each
// result into an initializer: Constant nodes; int64 values shape inference
// worked out (a folded Shape chain); and float nodes whose inputs are all
// initializers, for the operations computed exactly or with one rounding per
// element (Relu, Add, Sub, Mul, Div, Transpose and the reshapes), using the
// runtime's own kernels. graph outputs are never folded, nor initializers the
// caller may replace (graph inputs), nor results larger than their inputs
size_t fold_constants(Graph& g);

// Identity, and Dropout at inference, pass their input through: consumers are
// rewired to the input, or, when the output is a graph output, the input's
// producer is made to write the output directly, so the graph's interface
// keeps its names. a Dropout whose mask is used, or which has a training_mode
// input, stays
size_t remove_identities(Graph& g);

// a Relu whose input comes from a Gemm or Conv, and is read by nothing else,
// is fused into it: the node gets the kFusedRelu attribute and writes the
// Relu's output, and its kernel applies the Relu as it writes. the fused
// kernels use the Relu kernel's own expression, so the bits are the same
size_t fuse_relu(Graph& g);

// nodes whose outputs reach no graph output are marked dead, and initializers
// that nothing reads any more are dropped (unless they are graph inputs)
size_t eliminate_dead_nodes(Graph& g);

} // namespace null
