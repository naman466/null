#pragma once
#include "core/tensor.h"
#include "core/node.h"
#include "core/graph.h"

// shared test graphs

namespace null_test {

using namespace null;

inline Graph make_simple_matmul_graph() {
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

inline Graph make_relu_graph() {
    Graph g("test_relu");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_value_info(Tensor{"x", {1, 4}, DType::FLOAT32});
    g.add_value_info(Tensor{"y", {1, 4}, DType::FLOAT32});
    Node relu("relu", OpType::Relu, {"x"}, {"y"});
    g.add_node(relu);
    return g;
}

} // namespace null_test
