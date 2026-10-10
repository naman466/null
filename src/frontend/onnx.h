#pragma once
#include <stdexcept>
#include <string>

#include "core/tensor.h"
#include "frontend/protobuf.h"

namespace null::onnx {

// decodes onnx protobuf messages into the null IR
// field numbers follow onnx/onnx.proto (IR version 10)

class OnnxError : public std::runtime_error {
public:
    explicit OnnxError(const std::string& msg) : std::runtime_error("onnx: " + msg) {}
};

// map an onnx TensorProto.DataType to a DType, throws if unsupported
DType to_dtype(int32_t onnx_type);

// TensorProto -> Tensor with constant data
// r reads the body of one TensorProto message
Tensor parse_tensor(pb::Reader r);

} // namespace null::onnx
