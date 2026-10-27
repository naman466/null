#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "analysis/shape_inference.h"
#include "core/graph.h"

namespace null {

// specialising a graph to concrete input shapes
// generated C needs every size at compile time, so dynamic dims (-1) in the
// graph inputs have to be fixed before shape inference can make everything static

// replace a graph input's shape with a concrete one
// the new shape must have the same rank, agree with every known dim, and
// contain no dynamic dims; throws ShapeError otherwise
void set_input_shape(Graph& g, const std::string& input, const std::vector<int64_t>& shape);

// set a dynamic leading (batch) dim of every graph input to n (n >= 1)
// inputs whose first dim is already fixed are left alone
// returns how many inputs were changed
int set_batch_size(Graph& g, int64_t n);

// every tensor whose shape still has a dynamic dim, or no metadata at all:
// graph inputs first, then node outputs in execution order
std::vector<std::string> dynamic_tensors(const Graph& g);

} // namespace null
