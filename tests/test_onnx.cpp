#include "test_framework.h"
#include "pb_writer.h"
#include "frontend/onnx.h"

#include <cstring>
#include <string>
#include <vector>

using namespace null;
using namespace null::onnx;
using null_test::PbWriter;

// TensorProto field numbers
enum : uint32_t {
    DIMS = 1, DATA_TYPE = 2, FLOAT_DATA = 4, INT32_DATA = 5, STRING_DATA = 6,
    INT64_DATA = 7, NAME = 8, RAW_DATA = 9, DOUBLE_DATA = 10, DOC_STRING = 12,
    EXTERNAL_DATA = 13, DATA_LOCATION = 14,
};

// onnx data type codes
enum : int64_t {
    T_FLOAT = 1, T_UINT8 = 2, T_INT8 = 3, T_UINT16 = 4, T_INT16 = 5, T_INT32 = 6,
    T_INT64 = 7, T_STRING = 8, T_BOOL = 9, T_FLOAT16 = 10, T_DOUBLE = 11,
};

template<typename T>
static std::string raw_bytes(const std::vector<T>& values) {
    std::string s(values.size() * sizeof(T), '\0');
    std::memcpy(s.data(), values.data(), s.size());
    return s;
}

// start a TensorProto with a name, data type and shape
static PbWriter tensor_header(const std::string& name, int64_t dtype,
                              const std::vector<int64_t>& dims) {
    PbWriter w;
    for (int64_t d : dims) w.int_field(DIMS, d);   // dims are unpacked in proto2
    w.int_field(DATA_TYPE, dtype);
    w.bytes_field(NAME, name);
    return w;
}

template<typename T>
static std::vector<T> values_of(const Tensor& t) {
    std::vector<T> out(t.data.size() / sizeof(T));
    std::memcpy(out.data(), t.data.data(), t.data.size());
    return out;
}

// TESTS - PB WRITER

TEST(pb_writer_encoding) {
    // test helper sanity check against known bytes
    PbWriter w;
    w.int_field(1, 300).bytes_field(2, "hi").float_field(3, 1.0f).int_field(4, -1);
    std::vector<uint8_t> expected = {
        0x08, 0xAC, 0x02,
        0x12, 0x02, 'h', 'i',
        0x1D, 0x00, 0x00, 0x80, 0x3F,
        0x20, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01,
    };
    CHECK_EQ(w.buf, expected);
}

// TESTS - DTYPE

TEST(dtype_mapping) {
    CHECK_EQ(to_dtype(T_FLOAT),  DType::FLOAT32);
    CHECK_EQ(to_dtype(T_UINT8),  DType::UINT8);
    CHECK_EQ(to_dtype(T_INT8),   DType::INT8);
    CHECK_EQ(to_dtype(T_UINT16), DType::UINT16);
    CHECK_EQ(to_dtype(T_INT16),  DType::INT16);
    CHECK_EQ(to_dtype(T_INT32),  DType::INT32);
    CHECK_EQ(to_dtype(T_INT64),  DType::INT64);
    CHECK_EQ(to_dtype(T_BOOL),   DType::BOOL);
    CHECK_EQ(to_dtype(T_DOUBLE), DType::FLOAT64);
}

TEST(dtype_unsupported_throws) {
    CHECK_THROWS(to_dtype(0));          // UNDEFINED
    CHECK_THROWS(to_dtype(T_STRING));
    CHECK_THROWS(to_dtype(T_FLOAT16));
    CHECK_THROWS(to_dtype(16));         // BFLOAT16
    CHECK_THROWS(to_dtype(999));
}

// TESTS - TENSOR

TEST(tensor_raw_float) {
    std::vector<float> vals = {1, 2, 3, 4, 5, 6};
    PbWriter w = tensor_header("weight", T_FLOAT, {2, 3});
    w.bytes_field(RAW_DATA, raw_bytes(vals));

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.name, "weight");
    CHECK_EQ(t.shape, (std::vector<int64_t>{2, 3}));
    CHECK_EQ(t.dtype, DType::FLOAT32);
    CHECK(t.is_constant);
    CHECK_EQ(t.data.size(), 24u);
    CHECK_EQ(values_of<float>(t), vals);
    CHECK_NEAR(t.float_ptr()[5], 6.0f, 1e-6f);
}

TEST(tensor_raw_int64) {
    std::vector<int64_t> vals = {1, -1, 1LL << 40};
    PbWriter w = tensor_header("shape", T_INT64, {3});
    w.bytes_field(RAW_DATA, raw_bytes(vals));

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.dtype, DType::INT64);
    CHECK_EQ(values_of<int64_t>(t), vals);
    CHECK_EQ(t.int64_ptr()[2], 1LL << 40);
}

TEST(tensor_float_data_packed) {
    PbWriter w = tensor_header("b", T_FLOAT, {3});
    w.packed_floats(FLOAT_DATA, {0.5f, -1.5f, 2.0f});

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(values_of<float>(t), (std::vector<float>{0.5f, -1.5f, 2.0f}));
}

TEST(tensor_float_data_unpacked) {
    PbWriter w = tensor_header("b", T_FLOAT, {2});
    w.float_field(FLOAT_DATA, 1.0f).float_field(FLOAT_DATA, 2.0f);

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(values_of<float>(t), (std::vector<float>{1.0f, 2.0f}));
}

TEST(tensor_int64_data) {
    PbWriter w = tensor_header("axes", T_INT64, {2});
    w.packed_ints(INT64_DATA, {0, -1});

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(values_of<int64_t>(t), (std::vector<int64_t>{0, -1}));
}

TEST(tensor_int32_data) {
    PbWriter w = tensor_header("i", T_INT32, {3});
    w.packed_ints(INT32_DATA, {7, -7, 2147483647});

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.data.size(), 12u);
    CHECK_EQ(values_of<int32_t>(t), (std::vector<int32_t>{7, -7, 2147483647}));
}

TEST(tensor_int32_data_narrow_types) {
    // int32_data also stores int8, uint8, int16, uint16 and bool
    {
        PbWriter w = tensor_header("i8", T_INT8, {3});
        w.packed_ints(INT32_DATA, {-128, 0, 127});
        Tensor t = parse_tensor(w.reader());
        CHECK_EQ(t.data.size(), 3u);
        CHECK_EQ(values_of<int8_t>(t), (std::vector<int8_t>{-128, 0, 127}));
    }
    {
        PbWriter w = tensor_header("u8", T_UINT8, {2});
        w.packed_ints(INT32_DATA, {0, 255});
        Tensor t = parse_tensor(w.reader());
        CHECK_EQ(values_of<uint8_t>(t), (std::vector<uint8_t>{0, 255}));
    }
    {
        PbWriter w = tensor_header("i16", T_INT16, {2});
        w.packed_ints(INT32_DATA, {-300, 300});
        Tensor t = parse_tensor(w.reader());
        CHECK_EQ(t.data.size(), 4u);
        CHECK_EQ(values_of<int16_t>(t), (std::vector<int16_t>{-300, 300}));
    }
    {
        PbWriter w = tensor_header("u16", T_UINT16, {1});
        w.packed_ints(INT32_DATA, {65535});
        Tensor t = parse_tensor(w.reader());
        CHECK_EQ(values_of<uint16_t>(t), (std::vector<uint16_t>{65535}));
    }
    {
        PbWriter w = tensor_header("mask", T_BOOL, {3});
        w.packed_ints(INT32_DATA, {0, 1, 5});
        Tensor t = parse_tensor(w.reader());
        CHECK_EQ(values_of<uint8_t>(t), (std::vector<uint8_t>{0, 1, 1}));
    }
}

TEST(tensor_double_data) {
    PbWriter w = tensor_header("d", T_DOUBLE, {2});
    w.packed_doubles(DOUBLE_DATA, {0.25, -8.0});

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.dtype, DType::FLOAT64);
    CHECK_EQ(values_of<double>(t), (std::vector<double>{0.25, -8.0}));
}

TEST(tensor_scalar) {
    // no dims means a scalar with one element
    PbWriter w = tensor_header("eps", T_FLOAT, {});
    w.packed_floats(FLOAT_DATA, {1e-5f});

    Tensor t = parse_tensor(w.reader());
    CHECK(t.shape.empty());
    CHECK_EQ(t.num_elements(), 1);
    CHECK_NEAR(t.float_ptr()[0], 1e-5f, 1e-12f);
}

TEST(tensor_empty) {
    // a zero sized dim needs no data
    PbWriter w = tensor_header("empty", T_FLOAT, {0, 4});
    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.num_elements(), 0);
    CHECK(t.data.empty());
}

TEST(tensor_field_order_and_unknown_fields) {
    // fields may come in any order, unknown ones are skipped
    PbWriter w;
    w.bytes_field(DOC_STRING, "a bias");
    w.packed_floats(FLOAT_DATA, {3.0f});
    w.int_field(99, 42);
    w.bytes_field(NAME, "bias");
    w.int_field(DIMS, 1);
    w.int_field(DATA_TYPE, T_FLOAT);

    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.name, "bias");
    CHECK_EQ(t.shape, (std::vector<int64_t>{1}));
    CHECK_EQ(values_of<float>(t), (std::vector<float>{3.0f}));
}

// TESTS - TENSOR ERRORS

TEST(tensor_missing_data_throws) {
    PbWriter w = tensor_header("w", T_FLOAT, {2});
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(tensor_raw_size_mismatch_throws) {
    PbWriter w = tensor_header("w", T_FLOAT, {2, 2});
    w.bytes_field(RAW_DATA, raw_bytes(std::vector<float>{1, 2, 3}));
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(tensor_value_count_mismatch_throws) {
    PbWriter w = tensor_header("w", T_FLOAT, {3});
    w.packed_floats(FLOAT_DATA, {1, 2});
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(tensor_field_dtype_mismatch_throws) {
    PbWriter a = tensor_header("w", T_INT64, {1});
    a.packed_floats(FLOAT_DATA, {1.0f});
    CHECK_THROWS(parse_tensor(a.reader()));

    PbWriter b = tensor_header("w", T_FLOAT, {1});
    b.packed_ints(INT64_DATA, {1});
    CHECK_THROWS(parse_tensor(b.reader()));

    PbWriter c = tensor_header("w", T_INT64, {1});
    c.packed_ints(INT32_DATA, {1});
    CHECK_THROWS(parse_tensor(c.reader()));
}

TEST(tensor_multiple_data_fields_throws) {
    PbWriter w = tensor_header("w", T_FLOAT, {1});
    w.packed_floats(FLOAT_DATA, {1.0f});
    w.bytes_field(RAW_DATA, raw_bytes(std::vector<float>{1.0f}));
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(tensor_unsupported_dtype_throws) {
    PbWriter a = tensor_header("s", T_STRING, {1});
    a.bytes_field(STRING_DATA, "hello");
    CHECK_THROWS(parse_tensor(a.reader()));

    PbWriter b = tensor_header("h", T_FLOAT16, {1});
    b.bytes_field(RAW_DATA, std::string(2, '\0'));
    CHECK_THROWS(parse_tensor(b.reader()));

    PbWriter c;   // no data_type at all
    c.int_field(DIMS, 1);
    CHECK_THROWS(parse_tensor(c.reader()));
}

TEST(tensor_negative_dim_throws) {
    PbWriter w = tensor_header("w", T_FLOAT, {-1, 2});
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(tensor_external_data_throws) {
    PbWriter a = tensor_header("w", T_FLOAT, {1});
    a.int_field(DATA_LOCATION, 1);
    CHECK_THROWS(parse_tensor(a.reader()));

    PbWriter entry;
    entry.bytes_field(1, "location").bytes_field(2, "weights.bin");
    PbWriter b = tensor_header("w", T_FLOAT, {1});
    b.message_field(EXTERNAL_DATA, entry);
    CHECK_THROWS(parse_tensor(b.reader()));
}

TEST(tensor_error_names_tensor) {
    PbWriter w = tensor_header("conv1.weight", T_FLOAT, {4});
    try {
        parse_tensor(w.reader());
        throw std::runtime_error("expected OnnxError");
    } catch (const OnnxError& e) {
        CHECK(std::string(e.what()).find("conv1.weight") != std::string::npos);
    }
}
