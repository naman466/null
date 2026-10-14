#pragma once
#include <stdexcept>
#include <string>

#include "core/tensor.h"
#include "core/node.h"
#include "core/graph.h"
#include "frontend/protobuf.h"

namespace null::onnx {

// decodes onnx protobuf messages into the null IR
// field numbers follow onnx/onnx.proto (IR version 10)

class OnnxError : public std::runtime_error {
public:
    explicit OnnxError(const std::string& msg)
        : std::runtime_error("onnx: " + msg), detail_(msg) {}

    // message without the "onnx: " prefix, for wrapping in outer context
    const std::string& detail() const { return detail_; }

private:
    std::string detail_;
};

// an operator null can't compile, separate so a graph can report them all
class UnsupportedOpError : public OnnxError {
public:
    UnsupportedOpError(const std::string& msg, std::string op)
        : OnnxError(msg), op_type(std::move(op)) {}

    std::string op_type;   // "Op", or "domain::Op" outside the default domain
};

// supported range of the default onnx operator set
// operator semantics change between versions (axes moving from attributes
// to inputs in 13), so models outside this range are rejected
constexpr int64_t kMinOpset = 13;
constexpr int64_t kMaxOpset = 28;

// map an onnx TensorProto.DataType to a DType, throws if unsupported
DType to_dtype(int32_t onnx_type);

// TensorProto -> Tensor with constant data
// r reads the body of one TensorProto message
Tensor parse_tensor(pb::Reader r);

// AttributeProto -> Attribute
// supports FLOAT, INT, STRING, TENSOR, FLOATS and INTS
Attribute parse_attribute(pb::Reader r);

// a graph input, output or intermediate value as declared in the model
// kept separate from Tensor because onnx distinguishes "unknown shape"
// from a scalar (empty shape), and names symbolic dims
struct ValueInfo {
    std::string name;
    DType dtype = DType::UNDEFINED;        // UNDEFINED when no type is given
    bool has_shape = false;                // false means unknown rank
    std::vector<int64_t> shape;            // -1 for unknown or symbolic dims
    std::vector<std::string> dim_params;   // symbolic dim names, "" for fixed dims

    // metadata-only tensor, unknown rank must be resolved by the caller
    Tensor to_tensor() const { return Tensor(name, shape, dtype); }
};

// ValueInfoProto -> ValueInfo, only tensor types are supported
ValueInfo parse_value_info(pb::Reader r);

// NodeProto -> Node
// only the default onnx domain is supported, unknown operators throw
// UnsupportedOpError
Node parse_node(pb::Reader r);

// a loaded model, the graph plus the metadata later stages may need
struct Model {
    int64_t ir_version = 0;
    int64_t opset_version = 0;   // of the default onnx domain
    std::string producer_name;
    std::string producer_version;
    Graph graph;
};

// GraphProto -> Graph
//   - initializers are not listed in graph.inputs, even when old models
//     (IR < 4) also declare them as inputs
//   - graph inputs need a known element type and rank
//   - value info with unknown type or rank is left out of graph.value_info
//   - unnamed nodes get a unique "<Op>_<index>" name
//   - all unsupported operators are reported together
//   - the finished graph must pass Graph::validate()
Graph parse_graph(pb::Reader r);

// ModelProto -> Model, checks the opset before parsing the graph
Model parse_model(pb::Reader r);

// read and parse an .onnx file
// every failure, including malformed protobuf, is reported as OnnxError
Model load_model(const std::string& path);

} // namespace null::onnx
