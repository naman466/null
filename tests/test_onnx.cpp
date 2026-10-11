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

// TESTS - ATTRIBUTE

// AttributeProto field numbers
enum : uint32_t {
    A_NAME = 1, A_F = 2, A_I = 3, A_S = 4, A_T = 5, A_G = 6, A_FLOATS = 7, A_INTS = 8,
    A_STRINGS = 9, A_TENSORS = 10, A_DOC = 13, A_TYPE = 20, A_REF = 21, A_SPARSE = 22,
};

// AttributeProto.AttributeType
enum : int64_t {
    AT_FLOAT = 1, AT_INT = 2, AT_STRING = 3, AT_TENSOR = 4, AT_GRAPH = 5,
    AT_FLOATS = 6, AT_INTS = 7, AT_STRINGS = 8, AT_TENSORS = 9, AT_SPARSE = 11,
};

static PbWriter attr_header(const std::string& name, int64_t type) {
    PbWriter w;
    w.bytes_field(A_NAME, name);
    w.int_field(A_TYPE, type);
    return w;
}

TEST(attr_float) {
    PbWriter w = attr_header("alpha", AT_FLOAT);
    w.float_field(A_F, 0.01f);
    Attribute a = parse_attribute(w.reader());
    CHECK_EQ(a.name, "alpha");
    CHECK(a.has_float());
    CHECK_EQ(a.as_float(), 0.01f);
}

TEST(attr_int) {
    PbWriter w = attr_header("axis", AT_INT);
    w.int_field(A_I, -1);
    Attribute a = parse_attribute(w.reader());
    CHECK(a.has_int());
    CHECK_EQ(a.as_int(), -1);
}

TEST(attr_string) {
    PbWriter w = attr_header("auto_pad", AT_STRING);
    w.bytes_field(A_S, "SAME_UPPER");
    Attribute a = parse_attribute(w.reader());
    CHECK(a.has_string());
    CHECK_EQ(a.as_string(), "SAME_UPPER");
}

TEST(attr_ints_unpacked_and_packed) {
    // proto2 writes ints unpacked, but packed must be accepted too
    PbWriter a = attr_header("pads", AT_INTS);
    a.int_field(A_INTS, 1).int_field(A_INTS, 2).int_field(A_INTS, 1).int_field(A_INTS, 2);
    CHECK_EQ(parse_attribute(a.reader()).as_ints(), (std::vector<int64_t>{1, 2, 1, 2}));

    PbWriter b = attr_header("perm", AT_INTS);
    b.packed_ints(A_INTS, {0, 2, 1});
    CHECK_EQ(parse_attribute(b.reader()).as_ints(), (std::vector<int64_t>{0, 2, 1}));
}

TEST(attr_floats) {
    PbWriter w = attr_header("scales", AT_FLOATS);
    w.float_field(A_FLOATS, 1.0f).float_field(A_FLOATS, 2.0f);
    Attribute a = parse_attribute(w.reader());
    CHECK(a.has_floats());
    CHECK_EQ(a.as_floats(), (std::vector<float>{1.0f, 2.0f}));
}

TEST(attr_empty_list_uses_type) {
    // an empty list has no value fields on the wire, only the type says what it is
    Attribute ints = parse_attribute(attr_header("pads", AT_INTS).reader());
    CHECK(ints.has_ints());
    CHECK(ints.as_ints().empty());

    Attribute floats = parse_attribute(attr_header("scales", AT_FLOATS).reader());
    CHECK(floats.has_floats());
    CHECK(floats.as_floats().empty());
}

TEST(attr_missing_scalar_uses_default) {
    // protobuf default for an unset scalar, as the official parser does
    Attribute i = parse_attribute(attr_header("keepdims", AT_INT).reader());
    CHECK_EQ(i.as_int(), 0);
    Attribute f = parse_attribute(attr_header("beta", AT_FLOAT).reader());
    CHECK_EQ(f.as_float(), 0.0f);
    Attribute s = parse_attribute(attr_header("mode", AT_STRING).reader());
    CHECK_EQ(s.as_string(), "");
}

TEST(attr_tensor) {
    // the value of a Constant node
    PbWriter t = tensor_header("", T_FLOAT, {2});
    t.packed_floats(FLOAT_DATA, {1.5f, -1.5f});
    PbWriter w = attr_header("value", AT_TENSOR);
    w.message_field(A_T, t);

    Attribute a = parse_attribute(w.reader());
    CHECK(a.has_tensor());
    CHECK_EQ(a.as_tensor().shape, (std::vector<int64_t>{2}));
    CHECK_EQ(values_of<float>(a.as_tensor()), (std::vector<float>{1.5f, -1.5f}));
}

TEST(attr_type_inferred_when_missing) {
    // models from before the type field existed
    PbWriter i;
    i.bytes_field(A_NAME, "axis").int_field(A_I, 3);
    CHECK(parse_attribute(i.reader()).has_int());

    PbWriter f;
    f.bytes_field(A_NAME, "eps").float_field(A_F, 1e-5f);
    CHECK(parse_attribute(f.reader()).has_float());

    PbWriter s;
    s.bytes_field(A_NAME, "mode").bytes_field(A_S, "constant");
    CHECK(parse_attribute(s.reader()).has_string());

    PbWriter is;
    is.bytes_field(A_NAME, "shape").int_field(A_INTS, 1).int_field(A_INTS, 2);
    CHECK(parse_attribute(is.reader()).has_ints());

    PbWriter fs;
    fs.bytes_field(A_NAME, "scales").float_field(A_FLOATS, 2.0f);
    CHECK(parse_attribute(fs.reader()).has_floats());

    PbWriter t;
    PbWriter body = tensor_header("", T_INT64, {1});
    body.packed_ints(INT64_DATA, {5});
    t.bytes_field(A_NAME, "value").message_field(A_T, body);
    CHECK(parse_attribute(t.reader()).has_tensor());
}

TEST(attr_unknown_fields_skipped) {
    PbWriter w;
    w.bytes_field(A_DOC, "slope of the negative part");
    w.int_field(A_TYPE, AT_FLOAT);
    w.int_field(99, 1);
    w.float_field(A_F, 0.2f);
    w.bytes_field(A_NAME, "alpha");
    Attribute a = parse_attribute(w.reader());
    CHECK_EQ(a.name, "alpha");
    CHECK_EQ(a.as_float(), 0.2f);
}

TEST(attr_unsupported_types_throw) {
    PbWriter g = attr_header("then_branch", AT_GRAPH);
    g.message_field(A_G, PbWriter{});
    CHECK_THROWS(parse_attribute(g.reader()));

    PbWriter ss = attr_header("activations", AT_STRINGS);
    ss.bytes_field(A_STRINGS, "Sigmoid").bytes_field(A_STRINGS, "Tanh");
    CHECK_THROWS(parse_attribute(ss.reader()));

    PbWriter ts = attr_header("tensors", AT_TENSORS);
    ts.message_field(A_TENSORS, PbWriter{});
    CHECK_THROWS(parse_attribute(ts.reader()));

    PbWriter sp = attr_header("sparse_value", AT_SPARSE);
    sp.message_field(A_SPARSE, PbWriter{});
    CHECK_THROWS(parse_attribute(sp.reader()));

    // no type field, unsupported value on the wire
    PbWriter g2;
    g2.bytes_field(A_NAME, "body").message_field(A_G, PbWriter{});
    CHECK_THROWS(parse_attribute(g2.reader()));

    CHECK_THROWS(parse_attribute(attr_header("x", 999).reader()));
}

TEST(attr_errors) {
    // no name
    PbWriter a;
    a.int_field(A_TYPE, AT_INT).int_field(A_I, 1);
    CHECK_THROWS(parse_attribute(a.reader()));

    // no type and no value
    PbWriter b;
    b.bytes_field(A_NAME, "axis");
    CHECK_THROWS(parse_attribute(b.reader()));

    // TENSOR without a tensor
    CHECK_THROWS(parse_attribute(attr_header("value", AT_TENSOR).reader()));

    // reference to a function attribute
    PbWriter r = attr_header("alpha", AT_FLOAT);
    r.bytes_field(A_REF, "outer_alpha");
    CHECK_THROWS(parse_attribute(r.reader()));
}

TEST(attr_tensor_error_names_attribute) {
    // a bad embedded tensor reports both the attribute and the tensor problem
    PbWriter t = tensor_header("", T_FLOAT16, {1});
    PbWriter w = attr_header("value", AT_TENSOR);
    w.message_field(A_T, t);
    try {
        parse_attribute(w.reader());
        throw std::runtime_error("expected OnnxError");
    } catch (const OnnxError& e) {
        std::string msg = e.what();
        CHECK(msg.find("attribute 'value'") != std::string::npos);
        CHECK(msg.find("unsupported data type") != std::string::npos);
        CHECK_EQ(msg.find("onnx:", 1), std::string::npos);   // prefix only once
    }
}

// TESTS - NODE

// NodeProto field numbers
enum : uint32_t {
    N_INPUT = 1, N_OUTPUT = 2, N_NAME = 3, N_OP_TYPE = 4, N_ATTRIBUTE = 5,
    N_DOC = 6, N_DOMAIN = 7,
};

static PbWriter node_proto(const std::string& name, const std::string& op,
                           const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs) {
    PbWriter w;
    for (auto& s : inputs) w.bytes_field(N_INPUT, s);
    for (auto& s : outputs) w.bytes_field(N_OUTPUT, s);
    if (!name.empty()) w.bytes_field(N_NAME, name);
    w.bytes_field(N_OP_TYPE, op);
    return w;
}

static PbWriter int_attr(const std::string& name, int64_t v) {
    PbWriter a = attr_header(name, AT_INT);
    a.int_field(A_I, v);
    return a;
}

TEST(node_basic) {
    PbWriter w = node_proto("/fc1/Gemm", "Gemm", {"x", "fc1.weight", "fc1.bias"}, {"h"});
    Node n = parse_node(w.reader());
    CHECK_EQ(n.name, "/fc1/Gemm");
    CHECK_EQ(n.op, OpType::Gemm);
    CHECK_EQ(n.inputs, (std::vector<std::string>{"x", "fc1.weight", "fc1.bias"}));
    CHECK_EQ(n.outputs, (std::vector<std::string>{"h"}));
    CHECK(n.attrs.empty());
    CHECK(!n.is_dead);
}

TEST(node_with_attributes) {
    PbWriter pads = attr_header("pads", AT_INTS);
    pads.packed_ints(A_INTS, {1, 1, 1, 1});
    PbWriter alpha = attr_header("auto_pad", AT_STRING);
    alpha.bytes_field(A_S, "NOTSET");

    PbWriter w = node_proto("conv", "Conv", {"x", "w"}, {"y"});
    w.message_field(N_ATTRIBUTE, int_attr("group", 1));
    w.message_field(N_ATTRIBUTE, pads);
    w.message_field(N_ATTRIBUTE, alpha);

    Node n = parse_node(w.reader());
    CHECK_EQ(n.attrs.size(), 3u);
    CHECK_EQ(n.attr_int("group"), 1);
    CHECK_EQ(n.attr_ints("pads"), (std::vector<int64_t>{1, 1, 1, 1}));
    CHECK_EQ(n.attr_str("auto_pad"), "NOTSET");
    CHECK_EQ(n.get_attr("pads").name, "pads");
}

TEST(node_optional_inputs_kept) {
    // "" marks a skipped optional input, positions must be preserved
    PbWriter w = node_proto("clip", "Clip", {"x", "", "max"}, {"y"});
    Node n = parse_node(w.reader());
    CHECK_EQ(n.inputs, (std::vector<std::string>{"x", "", "max"}));
}

TEST(node_multiple_outputs) {
    PbWriter w = node_proto("split", "Split", {"x"}, {"a", "b", "c"});
    Node n = parse_node(w.reader());
    CHECK_EQ(n.outputs, (std::vector<std::string>{"a", "b", "c"}));
}

TEST(node_constant_no_inputs) {
    PbWriter t = tensor_header("", T_INT64, {2});
    t.packed_ints(INT64_DATA, {1, -1});
    PbWriter value = attr_header("value", AT_TENSOR);
    value.message_field(A_T, t);

    PbWriter w = node_proto("c", "Constant", {}, {"shape"});
    w.message_field(N_ATTRIBUTE, value);

    Node n = parse_node(w.reader());
    CHECK_EQ(n.op, OpType::Constant);
    CHECK(n.inputs.empty());
    CHECK_EQ(n.get_attr("value").as_tensor().num_elements(), 2);
}

TEST(node_unnamed) {
    // helper.make_node leaves the name empty
    PbWriter w = node_proto("", "Relu", {"x"}, {"y"});
    Node n = parse_node(w.reader());
    CHECK(n.name.empty());
    CHECK_EQ(n.op, OpType::Relu);
}

TEST(node_default_domain) {
    PbWriter a = node_proto("r", "Relu", {"x"}, {"y"});
    a.bytes_field(N_DOMAIN, "");
    CHECK_EQ(parse_node(a.reader()).op, OpType::Relu);

    PbWriter b = node_proto("r", "Relu", {"x"}, {"y"});
    b.bytes_field(N_DOMAIN, "ai.onnx");
    CHECK_EQ(parse_node(b.reader()).op, OpType::Relu);
}

TEST(node_unknown_fields_skipped) {
    PbWriter w;
    w.bytes_field(N_DOC, "first layer");
    w.bytes_field(N_OP_TYPE, "Relu");
    w.int_field(99, 3);
    w.bytes_field(N_OUTPUT, "y");
    w.bytes_field(N_INPUT, "x");
    Node n = parse_node(w.reader());
    CHECK_EQ(n.op, OpType::Relu);
    CHECK_EQ(n.inputs, (std::vector<std::string>{"x"}));
    CHECK_EQ(n.outputs, (std::vector<std::string>{"y"}));
}

// TESTS - NODE ERRORS

static std::string node_error(const PbWriter& w) {
    try {
        parse_node(w.reader());
    } catch (const OnnxError& e) {
        return e.what();
    }
    throw std::runtime_error("expected OnnxError");
}

TEST(node_unsupported_op_throws) {
    std::string msg = node_error(node_proto("/lstm/LSTM", "LSTM", {"x"}, {"y"}));
    CHECK(msg.find("'/lstm/LSTM'") != std::string::npos);
    CHECK(msg.find("unsupported operator 'LSTM'") != std::string::npos);

    // op names are case sensitive
    CHECK_THROWS(parse_node(node_proto("r", "relu", {"x"}, {"y"}).reader()));
    // optimizer-only ops never come from a model
    CHECK_THROWS(parse_node(node_proto("f", "ReluFused", {"x"}, {"y"}).reader()));
}

TEST(node_other_domain_throws) {
    PbWriter a = node_proto("g", "Gelu", {"x"}, {"y"});
    a.bytes_field(N_DOMAIN, "com.microsoft");
    CHECK(node_error(a).find("domain 'com.microsoft'") != std::string::npos);

    // known op name, but from another domain, is still rejected
    PbWriter b = node_proto("r", "Relu", {"x"}, {"y"});
    b.bytes_field(N_DOMAIN, "ai.onnx.ml");
    CHECK_THROWS(parse_node(b.reader()));
}

TEST(node_missing_op_type_throws) {
    PbWriter w;
    w.bytes_field(N_INPUT, "x").bytes_field(N_OUTPUT, "y");
    CHECK_THROWS(parse_node(w.reader()));
}

TEST(node_duplicate_attribute_throws) {
    PbWriter w = node_proto("s", "Softmax", {"x"}, {"y"});
    w.message_field(N_ATTRIBUTE, int_attr("axis", 1));
    w.message_field(N_ATTRIBUTE, int_attr("axis", -1));
    CHECK(node_error(w).find("duplicate attribute 'axis'") != std::string::npos);
}

TEST(node_bad_attribute_names_node) {
    PbWriter w = node_proto("/rnn/Loop", "Gemm", {"a", "b"}, {"y"});
    PbWriter g = attr_header("body", AT_GRAPH);
    g.message_field(A_G, PbWriter{});
    w.message_field(N_ATTRIBUTE, g);

    std::string msg = node_error(w);
    CHECK(msg.find("node '/rnn/Loop' (Gemm)") != std::string::npos);
    CHECK(msg.find("attribute 'body'") != std::string::npos);
    CHECK_EQ(msg.find("onnx:", 1), std::string::npos);
}

TEST(node_unnamed_error_context) {
    std::string msg = node_error(node_proto("", "NonMaxSuppression", {"b", "s"}, {"i"}));
    CHECK(msg.find("node <unnamed> (NonMaxSuppression)") != std::string::npos);
}

// TESTS - VALUE INFO

// ValueInfoProto / TypeProto / TensorShapeProto field numbers
enum : uint32_t {
    VI_NAME = 1, VI_TYPE = 2, VI_DOC = 3,
    TP_TENSOR = 1, TP_SEQUENCE = 4, TP_MAP = 5, TP_SPARSE = 8, TP_OPTIONAL = 9, TP_DENOTATION = 6,
    TT_ELEM_TYPE = 1, TT_SHAPE = 2,
    SH_DIM = 1,
    DIM_VALUE = 1, DIM_PARAM = 2, DIM_DENOTATION = 3,
};

static PbWriter dim_value(int64_t v) { PbWriter d; d.int_field(DIM_VALUE, v); return d; }
static PbWriter dim_param(const std::string& p) { PbWriter d; d.bytes_field(DIM_PARAM, p); return d; }

static PbWriter shape_of(const std::vector<PbWriter>& dims) {
    PbWriter s;
    for (auto& d : dims) s.message_field(SH_DIM, d);
    return s;
}

// ValueInfoProto for a tensor, shape omitted when has_shape is false
static PbWriter value_info(const std::string& name, int64_t elem_type,
                           const std::vector<PbWriter>& dims, bool has_shape = true) {
    PbWriter tt;
    tt.int_field(TT_ELEM_TYPE, elem_type);
    if (has_shape) tt.message_field(TT_SHAPE, shape_of(dims));
    PbWriter type;
    type.message_field(TP_TENSOR, tt);
    PbWriter vi;
    vi.bytes_field(VI_NAME, name);
    vi.message_field(VI_TYPE, type);
    return vi;
}

TEST(value_info_fixed_shape) {
    PbWriter w = value_info("input", T_FLOAT, {dim_value(1), dim_value(3), dim_value(224), dim_value(224)});
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.name, "input");
    CHECK_EQ(vi.dtype, DType::FLOAT32);
    CHECK(vi.has_shape);
    CHECK_EQ(vi.shape, (std::vector<int64_t>{1, 3, 224, 224}));
    CHECK_EQ(vi.dim_params, (std::vector<std::string>{"", "", "", ""}));

    Tensor t = vi.to_tensor();
    CHECK_EQ(t.name, "input");
    CHECK_EQ(t.num_elements(), 3 * 224 * 224);
    CHECK(!t.is_constant);
    CHECK(t.data.empty());
}

TEST(value_info_symbolic_dims) {
    // pytorch dynamic_axes write named dims like "batch_size"
    PbWriter w = value_info("x", T_FLOAT, {dim_param("batch_size"), dim_value(784)});
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.shape, (std::vector<int64_t>{-1, 784}));
    CHECK_EQ(vi.dim_params, (std::vector<std::string>{"batch_size", ""}));
    CHECK_EQ(vi.to_tensor().num_elements(), -1);
}

TEST(value_info_unknown_dim) {
    // a dimension with neither value nor param is unknown
    PbWriter w = value_info("x", T_FLOAT, {PbWriter{}, dim_value(10)});
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.shape, (std::vector<int64_t>{-1, 10}));
    CHECK_EQ(vi.dim_params, (std::vector<std::string>{"", ""}));
}

TEST(value_info_zero_dim) {
    PbWriter w = value_info("e", T_FLOAT, {dim_value(0), dim_value(4)});
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.shape, (std::vector<int64_t>{0, 4}));
    CHECK_EQ(vi.to_tensor().num_elements(), 0);
}

TEST(value_info_scalar_vs_unknown_rank) {
    // an empty shape is a scalar, a missing shape is unknown rank
    ValueInfo scalar = parse_value_info(value_info("s", T_FLOAT, {}).reader());
    CHECK(scalar.has_shape);
    CHECK(scalar.shape.empty());

    ValueInfo unknown = parse_value_info(value_info("u", T_FLOAT, {}, false).reader());
    CHECK(!unknown.has_shape);
    CHECK(unknown.shape.empty());
    CHECK_EQ(unknown.dtype, DType::FLOAT32);
}

TEST(value_info_no_type) {
    // intermediate values may carry only a name
    PbWriter w;
    w.bytes_field(VI_NAME, "h");
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.name, "h");
    CHECK_EQ(vi.dtype, DType::UNDEFINED);
    CHECK(!vi.has_shape);
}

TEST(value_info_no_elem_type) {
    PbWriter w = value_info("x", 0, {dim_value(2)});
    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.dtype, DType::UNDEFINED);
    CHECK_EQ(vi.shape, (std::vector<int64_t>{2}));
}

TEST(value_info_other_dtypes) {
    CHECK_EQ(parse_value_info(value_info("i", T_INT64, {dim_value(2)}).reader()).dtype, DType::INT64);
    CHECK_EQ(parse_value_info(value_info("b", T_BOOL, {}).reader()).dtype, DType::BOOL);
    CHECK_EQ(parse_value_info(value_info("u", T_UINT8, {}).reader()).dtype, DType::UINT8);
}

TEST(value_info_dim_oneof_last_wins) {
    PbWriter a;
    a.int_field(DIM_VALUE, 5).bytes_field(DIM_PARAM, "N");
    PbWriter b;
    b.bytes_field(DIM_PARAM, "N").int_field(DIM_VALUE, 5);
    ValueInfo vi = parse_value_info(value_info("x", T_FLOAT, {a, b}).reader());
    CHECK_EQ(vi.shape, (std::vector<int64_t>{-1, 5}));
    CHECK_EQ(vi.dim_params, (std::vector<std::string>{"N", ""}));
}

TEST(value_info_unknown_fields_skipped) {
    PbWriter d = dim_value(8);
    d.bytes_field(DIM_DENOTATION, "DATA_BATCH");
    PbWriter tt;
    tt.message_field(TT_SHAPE, shape_of({d}));
    tt.int_field(TT_ELEM_TYPE, T_FLOAT);
    PbWriter type;
    type.bytes_field(TP_DENOTATION, "TENSOR");
    type.message_field(TP_TENSOR, tt);
    PbWriter w;
    w.bytes_field(VI_DOC, "model input");
    w.message_field(VI_TYPE, type);
    w.int_field(99, 1);
    w.bytes_field(VI_NAME, "x");

    ValueInfo vi = parse_value_info(w.reader());
    CHECK_EQ(vi.name, "x");
    CHECK_EQ(vi.dtype, DType::FLOAT32);
    CHECK_EQ(vi.shape, (std::vector<int64_t>{8}));
}

// TESTS - VALUE INFO ERRORS

TEST(value_info_non_tensor_types_throw) {
    for (uint32_t field : {TP_SEQUENCE, TP_MAP, TP_SPARSE, TP_OPTIONAL}) {
        PbWriter type;
        type.message_field(field, PbWriter{});
        PbWriter w;
        w.bytes_field(VI_NAME, "seq");
        w.message_field(VI_TYPE, type);
        CHECK_THROWS(parse_value_info(w.reader()));
    }
}

TEST(value_info_errors) {
    // unsupported element type, names the value
    try {
        parse_value_info(value_info("half_in", T_FLOAT16, {dim_value(1)}).reader());
        throw std::runtime_error("expected OnnxError");
    } catch (const OnnxError& e) {
        CHECK(std::string(e.what()).find("value 'half_in'") != std::string::npos);
    }

    // negative dim_value, including -1
    CHECK_THROWS(parse_value_info(value_info("x", T_FLOAT, {dim_value(-1)}).reader()));
    CHECK_THROWS(parse_value_info(value_info("x", T_FLOAT, {dim_value(-5)}).reader()));

    // no name
    PbWriter w;
    w.message_field(VI_TYPE, PbWriter{});
    CHECK_THROWS(parse_value_info(w.reader()));
}
