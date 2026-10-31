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

// everything about a 2-D convolution that follows from its attributes and
// its input and weight shapes, shared by the shape rule and the emitter so
// both always agree on padding
struct ConvGeometry {
    int64_t n, c, h, w;              // input  [N, C, H, W]
    int64_t m, kh, kw;               // weight [M, C / group, kH, kW]
    int64_t group;
    int64_t stride_h, stride_w, dil_h, dil_w;
    int64_t pad_top, pad_left, pad_bottom, pad_right;
    int64_t out_h, out_w;            // output [N, M, out_h, out_w]
};

// computes the geometry of a Conv node, throws ShapeError naming the node
// if the shapes or attributes are invalid. needs static input and weight shapes
ConvGeometry conv_geometry(const Node& node, const std::vector<int64_t>& x,
                           const std::vector<int64_t>& weight);

// the same for a 2-D MaxPool: a window of kh x kw slides over each channel
// separately, so the output is [N, C, out_h, out_w]
struct PoolGeometry {
    int64_t n, c, h, w;              // input  [N, C, H, W]
    int64_t kh, kw;                  // kernel_shape
    int64_t stride_h, stride_w, dil_h, dil_w;
    int64_t pad_top, pad_left, pad_bottom, pad_right;
    int64_t out_h, out_w;            // output [N, C, out_h, out_w]
};

PoolGeometry pool_geometry(const Node& node, const std::vector<int64_t>& x);

// which input dims a ReduceMean or GlobalAveragePool averages away, one flag
// per input dim (none for a ReduceMean that passes its input through). throws
// ShapeError naming the node if the axes are invalid or not constant
std::vector<bool> reduced_axes(const Node& node, const Graph& g);

// infer the type and shape of every live node output, in execution order
//   - every node input must already have metadata (graph inputs and
//     initializers do, earlier node outputs get it from this pass)
//   - an output the model already declared is checked against the inferred
//     result: a conflict throws, a known declared dim refines an inferred -1
// throws ShapeError naming the node on any failure
void infer_shapes(Graph& g);

} // namespace null
