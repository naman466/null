#include "test_framework.h"
#include "pb_writer.h"
#include "frontend/onnx.h"
#include "core/graph.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// malformed and hostile inputs found by deliberately attacking the frontend
// and IR, each test pins the fixed (or intentionally accepted) behaviour

using namespace null;
using namespace null::onnx;
using null_test::PbWriter;
using WT = pb::WireType;

// TensorProto with dims, data type and name
static PbWriter tensor_proto(const std::vector<int64_t>& dims, int64_t dtype,
                             const std::string& name = "evil") {
    PbWriter w;
    for (int64_t d : dims) w.int_field(1, d);
    w.int_field(2, dtype);
    w.bytes_field(8, name);
    return w;
}

static std::string error_of(const PbWriter& w) {
    try {
        parse_tensor(w.reader());
    } catch (const std::exception& e) {
        return e.what();
    }
    throw std::runtime_error("expected parse_tensor to throw");
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// TESTS - ELEMENT COUNT OVERFLOW

TEST(overflow_raw_bytes_wrap_to_empty) {
    // 2^62 floats = 2^64 bytes, used to wrap to 0 and match empty raw_data
    PbWriter w = tensor_proto({1LL << 62}, 1);
    w.bytes_field(9, "");
    CHECK(error_of(w).find("too large") != std::string::npos);
}

TEST(overflow_count_without_data) {
    // 2^32 * 2^32 elements used to wrap to 0, so no data was required
    PbWriter w = tensor_proto({1LL << 32, 1LL << 32}, 1);
    CHECK(error_of(w).find("too large") != std::string::npos);
}

TEST(overflow_count_wraps_negative) {
    PbWriter w = tensor_proto({3, 1LL << 62}, 1);
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(overflow_zero_dim_wins) {
    // a zero dim makes the tensor empty, even next to huge dims
    PbWriter w = tensor_proto({1LL << 62, 4, 0}, 1);
    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.num_elements(), 0);
    CHECK_EQ(t.byte_size(), 0);
    CHECK(t.data.empty());
}

TEST(overflow_value_info_metadata) {
    // huge declared shapes are fine as metadata, but asking for the element
    // count must throw instead of overflowing
    PbWriter d; d.int_field(1, 1LL << 40);
    PbWriter shape; shape.message_field(1, d).message_field(1, d);
    PbWriter tt; tt.int_field(1, 1).message_field(2, shape);
    PbWriter type; type.message_field(1, tt);
    PbWriter vi; vi.bytes_field(1, "x").message_field(2, type);

    Tensor t = parse_value_info(vi.reader()).to_tensor();
    CHECK_EQ(t.shape, (std::vector<int64_t>{1LL << 40, 1LL << 40}));
    CHECK_THROWS(t.num_elements());
}

TEST(overflow_core_tensor_sizes) {
    Tensor count("c", {1LL << 62, 4}, DType::FLOAT32);
    CHECK_THROWS(count.num_elements());

    // element count fits, byte size does not
    Tensor bytes("b", {1LL << 62}, DType::FLOAT32);
    CHECK_EQ(bytes.num_elements(), 1LL << 62);
    CHECK_THROWS(bytes.byte_size());

    // the largest representable sizes still work
    Tensor max("m", {std::numeric_limits<int64_t>::max()}, DType::UINT8);
    CHECK_EQ(max.byte_size(), std::numeric_limits<int64_t>::max());
}

// TESTS - FIELD NUMBERS

TEST(field_number_truncation) {
    // field 2^32 + 1 used to truncate to field 1 and be read as dims
    PbWriter w;
    w.varint((((1ULL << 32) | 1) << 3) | 0).varint(7);
    w.int_field(2, 1).bytes_field(8, "t").packed_floats(4, {1, 2, 3, 4, 5, 6, 7});
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(field_number_limit) {
    // 2^29 - 1 is the largest legal field number
    PbWriter ok;
    ok.varint((((1ULL << 29) - 1) << 3) | 0).varint(1);
    pb::Reader r = ok.reader();
    CHECK_EQ(r.read_tag().field, (1u << 29) - 1);

    PbWriter bad;
    bad.varint((1ULL << 29) << 3).varint(1);
    pb::Reader rb = bad.reader();
    CHECK_THROWS(rb.read_tag());
}

// TESTS - WIRE TYPE CONFUSION

TEST(wiretype_confusion_matches_protobuf) {
    // data_type sent a second time as LEN with payload "dims=3": we used to
    // read the length as data_type UINT8 and the payload as an extra dim,
    // accepting uint8[4,3] where protobuf sees float[4]
    PbWriter payload; payload.int_field(1, 3);
    PbWriter w;
    w.int_field(1, 4).int_field(2, 1);
    w.tag(2, WT::LEN).varint(payload.buf.size());
    w.buf.insert(w.buf.end(), payload.buf.begin(), payload.buf.end());
    w.bytes_field(8, "confused").bytes_field(9, std::string(12, '\x01'));
    try {
        parse_tensor(w.reader());
        throw std::runtime_error("expected ParseError");
    } catch (const pb::ParseError& e) {
        CHECK(std::string(e.what()).find("field 2 has wire type LEN, expected VARINT") != std::string::npos);
    }
}

TEST(wiretype_string_as_varint) {
    // name as VARINT 3 used to swallow the next 3 bytes as the name
    PbWriter w;
    w.int_field(8, 3).int_field(1, 1).int_field(2, 1).int_field(1, 1).packed_floats(4, {9.0f});
    CHECK_THROWS(parse_tensor(w.reader()));
}

TEST(wiretype_message_as_varint) {
    // attribute as VARINT 0 used to become an empty message
    PbWriter w;
    w.bytes_field(4, "Relu").bytes_field(1, "x").bytes_field(2, "y").int_field(5, 0);
    CHECK_THROWS(parse_node(w.reader()));
}

TEST(wiretype_checked_in_every_message) {
    // attribute float sent as VARINT
    PbWriter a;
    a.bytes_field(1, "alpha").int_field(20, 1).int_field(2, 5);
    CHECK_THROWS(parse_attribute(a.reader()));

    // attribute int sent as I32
    PbWriter i;
    i.bytes_field(1, "axis").int_field(20, 2).float_field(3, 1.0f);
    CHECK_THROWS(parse_attribute(i.reader()));

    // node op_type sent as VARINT
    PbWriter n;
    n.int_field(4, 2).bytes_field(1, "x").bytes_field(2, "y");
    CHECK_THROWS(parse_node(n.reader()));

    // value info type sent as VARINT
    PbWriter v;
    v.bytes_field(1, "x").int_field(2, 1);
    CHECK_THROWS(parse_value_info(v.reader()));

    // dim_value sent as LEN
    PbWriter d; d.bytes_field(1, "x");
    PbWriter shape; shape.message_field(1, d);
    PbWriter tt; tt.int_field(1, 1).message_field(2, shape);
    PbWriter type; type.message_field(1, tt);
    PbWriter vi; vi.bytes_field(1, "x").message_field(2, type);
    CHECK_THROWS(parse_value_info(vi.reader()));
}

TEST(wiretype_unknown_fields_still_skipped) {
    // only known fields are checked, unknown ones of any wire type are skipped
    PbWriter w = tensor_proto({1}, 1);
    w.int_field(99, 1).float_field(98, 1.0f).double_field(97, 1.0).bytes_field(96, "x");
    w.packed_floats(4, {2.0f});
    CHECK_EQ(parse_tensor(w.reader()).float_ptr()[0], 2.0f);
}

// TESTS - BOOL

TEST(bool_raw_data_normalised) {
    PbWriter w = tensor_proto({3}, 9);
    w.bytes_field(9, std::string("\x00\x02\xff", 3));
    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.data, (std::vector<uint8_t>{0, 1, 1}));
}

// TESTS - ODD BUT ACCEPTED

TEST(accepted_nul_in_names) {
    PbWriter w = tensor_proto({1}, 1, std::string("a\0b", 3));
    w.packed_floats(4, {1});
    CHECK_EQ(parse_tensor(w.reader()).name.size(), 3u);
}

TEST(accepted_node_without_outputs) {
    // legal in a NodeProto, graph validation decides if it matters
    PbWriter w;
    w.bytes_field(4, "Relu").bytes_field(1, "x");
    CHECK(parse_node(w.reader()).outputs.empty());
}

TEST(accepted_empty_output_name) {
    // "" marks an omitted optional output
    PbWriter w;
    w.bytes_field(4, "Dropout").bytes_field(1, "x").bytes_field(2, "y").bytes_field(2, "");
    CHECK_EQ(parse_node(w.reader()).outputs, (std::vector<std::string>{"y", ""}));
}

TEST(accepted_attr_type_value_mismatch) {
    // type INT with only a float value present: the int takes the protobuf
    // default, matching the official parser
    PbWriter w;
    w.bytes_field(1, "axis").int_field(20, 2).float_field(2, 1.5f);
    CHECK_EQ(parse_attribute(w.reader()).as_int(), 0);
}

TEST(accepted_million_dims) {
    // tiny input, huge rank: must be linear time
    PbWriter w;
    for (int i = 0; i < 1000000; ++i) w.int_field(1, 1);
    w.int_field(2, 1).bytes_field(8, "rank1M").packed_floats(4, {1});
    auto t0 = std::chrono::steady_clock::now();
    Tensor t = parse_tensor(w.reader());
    CHECK_EQ(t.shape.size(), 1000000u);
    CHECK_EQ(t.num_elements(), 1);
    CHECK(seconds_since(t0) < 2.0);
}

TEST(empty_input_rejected_cleanly) {
    PbWriter empty;
    CHECK_THROWS(parse_tensor(empty.reader()));
    CHECK_THROWS(parse_attribute(empty.reader()));
    CHECK_THROWS(parse_node(empty.reader()));
    CHECK_THROWS(parse_value_info(empty.reader()));
}

// TESTS - GRAPH

TEST(graph_duplicate_producer) {
    Graph g("dup");
    g.inputs = {"x"};
    g.outputs = {"y"};
    g.add_node(Node("a", OpType::Relu, {"x"}, {"y"}));
    g.add_node(Node("b", OpType::Neg,  {"x"}, {"y"}));
    auto errors = g.validate();
    CHECK_EQ(errors.size(), 1u);
    CHECK(errors[0].find("'y' is produced by both node 'a' and node 'b'") != std::string::npos);
}

TEST(graph_node_overwrites_input_or_initializer) {
    Graph g("shadow");
    g.inputs = {"x"};
    g.outputs = {"y"};
    Tensor w("w", {1}, DType::FLOAT32);
    w.set_float_data({1.0f});
    g.add_initializer(w);
    g.add_node(Node("a", OpType::Relu, {"w"}, {"x"}));
    g.add_node(Node("b", OpType::Neg,  {"x"}, {"w"}));
    g.add_node(Node("c", OpType::Abs,  {"x"}, {"y"}));
    auto errors = g.validate();
    bool input = false, init = false;
    for (auto& e : errors) {
        if (e.find("'x' is produced by both graph input and node 'a'") != std::string::npos) input = true;
        if (e.find("'w' is produced by both initializer and node 'b'") != std::string::npos) init = true;
    }
    CHECK(input);
    CHECK(init);
}

TEST(graph_duplicate_checks_ignore_dead_and_empty) {
    Graph g("ok");
    g.inputs = {"x"};
    g.outputs = {"y"};
    Tensor w("w", {1}, DType::FLOAT32);
    w.set_float_data({1.0f});
    g.add_initializer(w);
    g.inputs.push_back("w");   // pre IR v4 models also list initializers as inputs

    g.add_node(Node("a", OpType::Dropout, {"x"}, {"y", ""}));
    g.add_node(Node("b", OpType::Dropout, {"x"}, {"z", ""}));   // second "" is fine
    g.add_node(Node("old", OpType::Relu, {"x"}, {"y"}));
    g.nodes.back().is_dead = true;                              // dead duplicate is fine
    CHECK(g.validate().empty());
}

TEST(graph_topo_sort_wide_is_fast) {
    // 100k independent nodes took ~5s with the old sorted-vector ready list
    Graph g("wide");
    g.inputs = {"x"};
    const size_t N = 100000;
    for (size_t i = 0; i < N; ++i)
        g.add_node(Node("n" + std::to_string(i), OpType::Relu, {"x"}, {"y" + std::to_string(i)}));

    auto t0 = std::chrono::steady_clock::now();
    auto order = g.topological_sort();
    CHECK(seconds_since(t0) < 2.0);

    CHECK_EQ(order.size(), N);
    for (size_t i = 0; i < N; ++i) CHECK_EQ(order[i], i);
}

TEST(graph_topo_sort_long_reversed_chain) {
    // added back to front, so every node waits on the one after it
    Graph g("chain");
    g.inputs = {"t0"};
    const size_t N = 200000;
    for (size_t k = 0; k < N; ++k) {
        size_t i = N - 1 - k;
        g.add_node(Node("n" + std::to_string(i), OpType::Relu,
                        {"t" + std::to_string(i)}, {"t" + std::to_string(i + 1)}));
    }

    auto t0 = std::chrono::steady_clock::now();
    auto order = g.topological_sort();
    CHECK(seconds_since(t0) < 2.0);

    CHECK_EQ(order.size(), N);
    for (size_t k = 0; k < N; ++k) CHECK_EQ(order[k], N - 1 - k);
}
