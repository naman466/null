#pragma once
#include <stdexcept>
#include <string>

#include "core/tensor.h"
#include "core/node.h"
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
Node parse_node(pb::Reader r);

} // namespace null::onnx
