#include "frontend/onnx.h"

#include <bit>
#include <cstring>
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

} // namespace null::onnx
