#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/graph.h"

namespace null {

// shape inference
// computes the element type and shape of every node output from its inputs
// and records them in Graph::value_info

class ShapeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// numpy style broadcast of two shapes
// dims are aligned from the right, a 1 stretches to the other size
// -1 (dynamic) broadcasts as: (-1, 1) -> -1, (-1, n) -> n, (-1, -1) -> -1
// throws ShapeError if two known dims differ and neither is 1
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b);

// whether infer_shapes knows how to handle this operator
bool has_shape_rule(OpType op);

// infer the type and shape of every live node output, in execution order
//   - every node input must already have metadata (graph inputs and
//     initializers do, earlier node outputs get it from this pass)
//   - an output the model already declared is checked against the inferred
//     result: a conflict throws, a known declared dim refines an inferred -1
// throws ShapeError naming the node on any failure
void infer_shapes(Graph& g);

} // namespace null
