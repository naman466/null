#include "frontend/onnx.h"

#include <bit>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

namespace null::onnx {

// onnx stores raw_data little endian, Tensor::data holds host order bytes
static_assert(std::endian::native == std::endian::little,
              "null assumes a little endian host");

namespace {

// TensorProto field numbers
enum TensorField : uint32_t {
    kTensorDims         = 1,
    kTensorDataType     = 2,
    kTensorFloatData    = 4,
    kTensorInt32Data    = 5,
    kTensorInt64Data    = 7,
    kTensorName         = 8,
    kTensorRawData      = 9,
    kTensorDoubleData   = 10,
    kTensorExternalData = 13,
    kTensorDataLocation = 14,
};

constexpr int32_t kDataLocationExternal = 1;

// AttributeProto field numbers
enum AttributeField : uint32_t {
    kAttrName        = 1,
    kAttrF           = 2,
    kAttrI           = 3,
    kAttrS           = 4,
    kAttrT           = 5,
    kAttrG           = 6,
    kAttrFloats      = 7,
    kAttrInts        = 8,
    kAttrStrings     = 9,
    kAttrTensors     = 10,
    kAttrGraphs      = 11,
    kAttrTp          = 14,
    kAttrTypeProtos  = 15,
    kAttrType        = 20,
    kAttrRefAttrName = 21,
    kAttrSparse      = 22,
    kAttrSparseList  = 23,
};

// AttributeProto.AttributeType
enum AttributeType : int32_t {
    kAttrTypeUndefined     = 0,
    kAttrTypeFloat         = 1,
    kAttrTypeInt           = 2,
    kAttrTypeString        = 3,
    kAttrTypeTensor        = 4,
    kAttrTypeGraph         = 5,
    kAttrTypeFloats        = 6,
    kAttrTypeInts          = 7,
    kAttrTypeStrings       = 8,
    kAttrTypeTensors       = 9,
    kAttrTypeGraphs        = 10,
    kAttrTypeSparseTensor  = 11,
    kAttrTypeSparseTensors = 12,
    kAttrTypeTypeProto     = 13,
    kAttrTypeTypeProtos    = 14,
};

// NodeProto field numbers
enum NodeField : uint32_t {
    kNodeInput     = 1,
    kNodeOutput    = 2,
    kNodeName      = 3,
    kNodeOpType    = 4,
    kNodeAttribute = 5,
    kNodeDomain    = 7,
};

// ValueInfoProto / TypeProto / TensorShapeProto field numbers
enum ValueInfoField : uint32_t {
    kValueInfoName = 1,
    kValueInfoType = 2,
};

enum TypeField : uint32_t {
    kTypeTensor       = 1,
    kTypeSequence     = 4,
    kTypeMap          = 5,
    kTypeSparseTensor = 8,
    kTypeOptional     = 9,
};

enum TensorTypeField : uint32_t {
    kTensorTypeElemType = 1,
    kTensorTypeShape    = 2,
};

enum ShapeField : uint32_t {
    kShapeDim = 1,
};

enum DimField : uint32_t {
    kDimValue = 1,
    kDimParam = 2,
};

const char* attr_type_name(int32_t type) {
    switch (type) {
        case kAttrTypeGraph:         return "GRAPH";
        case kAttrTypeStrings:       return "STRINGS";
        case kAttrTypeTensors:       return "TENSORS";
        case kAttrTypeGraphs:        return "GRAPHS";
        case kAttrTypeSparseTensor:  return "SPARSE_TENSOR";
        case kAttrTypeSparseTensors: return "SPARSE_TENSORS";
        case kAttrTypeTypeProto:     return "TYPE_PROTO";
        case kAttrTypeTypeProtos:    return "TYPE_PROTOS";
        default:                     return "UNDEFINED";
    }
}

// copy values into data as type T
template<typename T, typename Src>
void store_as(const std::vector<Src>& values, std::vector<uint8_t>& data) {
    data.resize(values.size() * sizeof(T));
    for (size_t i = 0; i < values.size(); ++i) {
        T v = static_cast<T>(values[i]);
        std::memcpy(data.data() + i * sizeof(T), &v, sizeof(T));
    }
}

} // namespace

DType to_dtype(int32_t onnx_type) {
    // DType values mirror onnx's, only the subset we support is accepted
    switch (onnx_type) {
        case 1:  return DType::FLOAT32;
        case 2:  return DType::UINT8;
        case 3:  return DType::INT8;
        case 4:  return DType::UINT16;
        case 5:  return DType::INT16;
        case 6:  return DType::INT32;
        case 7:  return DType::INT64;
        case 9:  return DType::BOOL;
        case 11: return DType::FLOAT64;
        default:
            throw OnnxError("unsupported tensor data type " + std::to_string(onnx_type));
    }
}

Tensor parse_tensor(pb::Reader r) {
    std::vector<int64_t> dims;
    int32_t data_type = 0;
    std::string name;

    // a tensor stores its values in exactly one of these
    std::string_view raw;
    bool has_raw = false;
    std::vector<float> float_data;
    std::vector<int64_t> int32_data;   // int32 values, widened by the reader
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;

    bool external = false;

    while (!r.eof()) {
        pb::Tag tag = r.read_tag();
        switch (tag.field) {
            case kTensorDims:         r.read_repeated_int64(tag.wire, dims); break;
            case kTensorDataType:     data_type = r.read_int32(); break;
            case kTensorFloatData:    r.read_repeated_float(tag.wire, float_data); break;
            case kTensorInt32Data:    r.read_repeated_int64(tag.wire, int32_data); break;
            case kTensorInt64Data:    r.read_repeated_int64(tag.wire, int64_data); break;
            case kTensorName:         name = r.read_string(); break;
            case kTensorRawData:      raw = r.read_bytes(); has_raw = true; break;
            case kTensorDoubleData:   r.read_repeated_double(tag.wire, double_data); break;
            case kTensorExternalData: external = true; r.skip(tag.wire); break;
            case kTensorDataLocation:
                if (r.read_int32() == kDataLocationExternal) external = true;
                break;
            default:                  r.skip(tag.wire); break;
        }
    }

    auto fail = [&](const std::string& msg) -> OnnxError {
        return OnnxError("tensor '" + name + "': " + msg);
    };

    for (int64_t d : dims) {
        if (d < 0) throw fail("negative dimension " + std::to_string(d));
    }
    if (external) throw fail("external data is not supported");

    Tensor t(name, dims, DType::UNDEFINED);
    try {
        t.dtype = to_dtype(data_type);
    } catch (const OnnxError&) {
        throw fail("unsupported data type " + std::to_string(data_type));
    }
    t.is_constant = true;

    int sources = has_raw + !float_data.empty() + !int32_data.empty()
                + !int64_data.empty() + !double_data.empty();
    if (sources > 1) throw fail("data stored in more than one field");

    const size_t count = static_cast<size_t>(t.num_elements());
    auto check_count = [&](size_t got) {
        if (got != count)
            throw fail("expected " + std::to_string(count) + " values, got " + std::to_string(got));
    };
    auto require_dtype = [&](bool ok, const char* field) {
        if (!ok) throw fail(std::string(field) + " does not match data type " + dtype_to_str(t.dtype));
    };

    if (has_raw) {
        const size_t expected = count * dtype_size(t.dtype);
        if (raw.size() != expected)
            throw fail("raw_data has " + std::to_string(raw.size()) +
                       " bytes, expected " + std::to_string(expected));
        t.data.assign(raw.begin(), raw.end());
    } else if (!float_data.empty()) {
        require_dtype(t.dtype == DType::FLOAT32, "float_data");
        check_count(float_data.size());
        store_as<float>(float_data, t.data);
    } else if (!int32_data.empty()) {
        // int32_data also carries the narrow integer types and bool
        check_count(int32_data.size());
        switch (t.dtype) {
            case DType::INT32:  store_as<int32_t>(int32_data, t.data);  break;
            case DType::INT16:  store_as<int16_t>(int32_data, t.data);  break;
            case DType::UINT16: store_as<uint16_t>(int32_data, t.data); break;
            case DType::INT8:   store_as<int8_t>(int32_data, t.data);   break;
            case DType::UINT8:  store_as<uint8_t>(int32_data, t.data);  break;
            case DType::BOOL:
                for (auto& v : int32_data) v = (v != 0);
                store_as<uint8_t>(int32_data, t.data);
                break;
            default: require_dtype(false, "int32_data");
        }
    } else if (!int64_data.empty()) {
        require_dtype(t.dtype == DType::INT64, "int64_data");
        check_count(int64_data.size());
        store_as<int64_t>(int64_data, t.data);
    } else if (!double_data.empty()) {
        require_dtype(t.dtype == DType::FLOAT64, "double_data");
        check_count(double_data.size());
        store_as<double>(double_data, t.data);
    } else if (count != 0) {
        throw fail("no data for " + std::to_string(count) + " elements");
    }

    return t;
}

Attribute parse_attribute(pb::Reader r) {
    std::string name;
    int32_t type = kAttrTypeUndefined;
    bool is_ref = false;

    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::optional<pb::Reader> t;   // parsed once the type is known
    std::vector<float> floats;
    std::vector<int64_t> ints;
    bool has_f = false, has_i = false, has_s = false;

    // value kind seen on the wire for a type we don't support,
    // used when the type field is missing
    int32_t unsupported_seen = kAttrTypeUndefined;

    while (!r.eof()) {
        pb::Tag tag = r.read_tag();
        switch (tag.field) {
            case kAttrName:   name = r.read_string(); break;
            case kAttrType:   type = r.read_int32(); break;
            case kAttrF:      f = r.read_float(); has_f = true; break;
            case kAttrI:      i = r.read_int64(); has_i = true; break;
            case kAttrS:      s = r.read_string(); has_s = true; break;
            case kAttrT:      t = r.read_message(); break;
            case kAttrFloats: r.read_repeated_float(tag.wire, floats); break;
            case kAttrInts:   r.read_repeated_int64(tag.wire, ints); break;
            case kAttrRefAttrName: is_ref = true; r.skip(tag.wire); break;
            case kAttrG:          unsupported_seen = kAttrTypeGraph;         r.skip(tag.wire); break;
            case kAttrStrings:    unsupported_seen = kAttrTypeStrings;       r.skip(tag.wire); break;
            case kAttrTensors:    unsupported_seen = kAttrTypeTensors;       r.skip(tag.wire); break;
            case kAttrGraphs:     unsupported_seen = kAttrTypeGraphs;        r.skip(tag.wire); break;
            case kAttrTp:         unsupported_seen = kAttrTypeTypeProto;     r.skip(tag.wire); break;
            case kAttrTypeProtos: unsupported_seen = kAttrTypeTypeProtos;    r.skip(tag.wire); break;
            case kAttrSparse:     unsupported_seen = kAttrTypeSparseTensor;  r.skip(tag.wire); break;
            case kAttrSparseList: unsupported_seen = kAttrTypeSparseTensors; r.skip(tag.wire); break;
            default:              r.skip(tag.wire); break;
        }
    }

    if (name.empty()) throw OnnxError("attribute without a name");

    auto fail = [&](const std::string& msg) -> OnnxError {
        return OnnxError("attribute '" + name + "': " + msg);
    };

    // references to an enclosing function's attributes only appear in function bodies
    if (is_ref) throw fail("ref_attr_name is not supported");

    // very old models omit the type, infer it from the field that was set
    if (type == kAttrTypeUndefined) {
        if      (t)                    type = kAttrTypeTensor;
        else if (has_s)                type = kAttrTypeString;
        else if (!floats.empty())      type = kAttrTypeFloats;
        else if (!ints.empty())        type = kAttrTypeInts;
        else if (has_f)                type = kAttrTypeFloat;
        else if (has_i)                type = kAttrTypeInt;
        else if (unsupported_seen)     type = unsupported_seen;
        else throw fail("no type and no value");
    }

    // missing scalars take the protobuf default, like the official parser
    switch (type) {
        case kAttrTypeFloat:  return Attribute(name, f);
        case kAttrTypeInt:    return Attribute(name, i);
        case kAttrTypeString: return Attribute(name, s);
        case kAttrTypeFloats: return Attribute(name, std::move(floats));
        case kAttrTypeInts:   return Attribute(name, std::move(ints));
        case kAttrTypeTensor:
            if (!t) throw fail("TENSOR attribute has no tensor");
            try {
                return Attribute(name, parse_tensor(*t));
            } catch (const OnnxError& e) {
                throw fail(e.detail());
            }
        case kAttrTypeGraph:
        case kAttrTypeStrings:
        case kAttrTypeTensors:
        case kAttrTypeGraphs:
        case kAttrTypeSparseTensor:
        case kAttrTypeSparseTensors:
        case kAttrTypeTypeProto:
        case kAttrTypeTypeProtos:
            throw fail(std::string(attr_type_name(type)) + " attributes are not supported");
        default:
            throw fail("unknown attribute type " + std::to_string(type));
    }
}

ValueInfo parse_value_info(pb::Reader r) {
    ValueInfo vi;
    std::optional<pb::Reader> type;

    while (!r.eof()) {
        pb::Tag tag = r.read_tag();
        switch (tag.field) {
            case kValueInfoName: vi.name = r.read_string(); break;
            case kValueInfoType: type = r.read_message(); break;
            default:             r.skip(tag.wire); break;
        }
    }

    if (vi.name.empty()) throw OnnxError("value info without a name");

    auto fail = [&](const std::string& msg) -> OnnxError {
        return OnnxError("value '" + vi.name + "': " + msg);
    };

    // no type at all, nothing is known about the value
    if (!type) return vi;

    // TypeProto, only tensors are supported
    std::optional<pb::Reader> tensor_type;
    while (!type->eof()) {
        pb::Tag tag = type->read_tag();
        switch (tag.field) {
            case kTypeTensor:       tensor_type = type->read_message(); break;
            case kTypeSequence:     throw fail("sequence types are not supported");
            case kTypeMap:          throw fail("map types are not supported");
            case kTypeSparseTensor: throw fail("sparse tensor types are not supported");
            case kTypeOptional:     throw fail("optional types are not supported");
            default:                type->skip(tag.wire); break;
        }
    }
    if (!tensor_type) return vi;

    // TypeProto.Tensor
    int32_t elem_type = 0;
    std::optional<pb::Reader> shape;
    while (!tensor_type->eof()) {
        pb::Tag tag = tensor_type->read_tag();
        switch (tag.field) {
            case kTensorTypeElemType: elem_type = tensor_type->read_int32(); break;
            case kTensorTypeShape:    shape = tensor_type->read_message(); break;
            default:                  tensor_type->skip(tag.wire); break;
        }
    }

    if (elem_type != 0) {
        try {
            vi.dtype = to_dtype(elem_type);
        } catch (const OnnxError&) {
            throw fail("unsupported data type " + std::to_string(elem_type));
        }
    }

    // a missing shape means unknown rank, an empty one means a scalar
    if (!shape) return vi;
    vi.has_shape = true;

    while (!shape->eof()) {
        pb::Tag tag = shape->read_tag();
        if (tag.field != kShapeDim) { shape->skip(tag.wire); continue; }

        // TensorShapeProto.Dimension, a fixed value, a symbolic name, or neither
        pb::Reader dim = shape->read_message();
        std::optional<int64_t> value;
        std::string param;
        while (!dim.eof()) {
            pb::Tag dt = dim.read_tag();
            switch (dt.field) {
                // a oneof, the last field on the wire wins
                case kDimValue: value = dim.read_int64(); param.clear(); break;
                case kDimParam: param = dim.read_string(); value.reset(); break;
                default:        dim.skip(dt.wire); break;
            }
        }
        if (value && *value < 0)
            throw fail("negative dimension " + std::to_string(*value));
        vi.shape.push_back(value ? *value : -1);
        vi.dim_params.push_back(param);
    }

    return vi;
}

Node parse_node(pb::Reader r) {
    Node n;
    std::string op_type;
    std::string domain;
    std::vector<pb::Reader> attrs;   // parsed after we know the node's name

    while (!r.eof()) {
        pb::Tag tag = r.read_tag();
        switch (tag.field) {
            case kNodeInput:     n.inputs.push_back(r.read_string()); break;
            case kNodeOutput:    n.outputs.push_back(r.read_string()); break;
            case kNodeName:      n.name = r.read_string(); break;
            case kNodeOpType:    op_type = r.read_string(); break;
            case kNodeAttribute: attrs.push_back(r.read_message()); break;
            case kNodeDomain:    domain = r.read_string(); break;
            default:             r.skip(tag.wire); break;
        }
    }

    auto fail = [&](const std::string& msg) -> OnnxError {
        std::string where = n.name.empty() ? "<unnamed>" : "'" + n.name + "'";
        if (!op_type.empty()) where += " (" + op_type + ")";
        return OnnxError("node " + where + ": " + msg);
    };

    if (op_type.empty()) throw fail("missing op_type");
    if (!domain.empty() && domain != "ai.onnx")
        throw fail("unsupported operator domain '" + domain + "'");

    n.op = str_to_optype(op_type);
    if (n.op == OpType::Unknown) throw fail("unsupported operator '" + op_type + "'");

    for (const pb::Reader& ar : attrs) {
        Attribute a;
        try {
            a = parse_attribute(ar);
        } catch (const OnnxError& e) {
            throw fail(e.detail());
        }
        if (n.has_attr(a.name)) throw fail("duplicate attribute '" + a.name + "'");
        n.attrs.emplace(a.name, std::move(a));
    }

    return n;
}

} // namespace null::onnx
