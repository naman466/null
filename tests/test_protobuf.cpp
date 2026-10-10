#include "test_framework.h"
#include "frontend/protobuf.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace null::pb;

using Bytes = std::vector<uint8_t>;

static Reader reader(const Bytes& b) {
    return Reader(b.data(), b.size());
}

// TESTS - VARINT

TEST(varint_single_byte) {
    Bytes b = {0x00, 0x01, 0x7F};
    Reader r = reader(b);
    CHECK_EQ(r.read_varint(), 0u);
    CHECK_EQ(r.read_varint(), 1u);
    CHECK_EQ(r.read_varint(), 127u);
    CHECK(r.eof());
}

TEST(varint_multi_byte) {
    // 300 = 0b1_0010_1100 -> 0xAC 0x02
    Bytes b = {0xAC, 0x02, 0x80, 0x01};
    Reader r = reader(b);
    CHECK_EQ(r.read_varint(), 300u);
    CHECK_EQ(r.read_varint(), 128u);
    CHECK(r.eof());
}

TEST(varint_max_uint64) {
    Bytes b = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    Reader r = reader(b);
    CHECK_EQ(r.read_varint(), std::numeric_limits<uint64_t>::max());
    CHECK_EQ(r.pos(), 10u);
}

TEST(varint_negative_int) {
    // negative int32/int64 are sign extended to 10 bytes
    Bytes b = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01,
               0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    Reader r = reader(b);
    CHECK_EQ(r.read_int64(), -1);
    CHECK_EQ(r.read_int32(), -2);
}

TEST(varint_bool) {
    Bytes b = {0x00, 0x01};
    Reader r = reader(b);
    CHECK(!r.read_bool());
    CHECK(r.read_bool());
}

TEST(varint_truncated_throws) {
    Bytes b = {0x80};
    Reader r = reader(b);
    CHECK_THROWS(r.read_varint());

    Bytes empty;
    Reader e = reader(empty);
    CHECK_THROWS(e.read_varint());
}

TEST(varint_too_long_throws) {
    Bytes b(11, 0x80);
    b.push_back(0x01);
    Reader r = reader(b);
    CHECK_THROWS(r.read_varint());
}

// TESTS - TAG

TEST(tag_decoding) {
    // field 1 varint, field 2 len, field 16 i32 (key 133 = 0x85 0x01)
    Bytes b = {0x08, 0x12, 0x85, 0x01, 0x09};
    Reader r = reader(b);

    Tag t = r.read_tag();
    CHECK_EQ(t.field, 1u);
    CHECK(t.wire == WireType::VARINT);

    t = r.read_tag();
    CHECK_EQ(t.field, 2u);
    CHECK(t.wire == WireType::LEN);

    t = r.read_tag();
    CHECK_EQ(t.field, 16u);
    CHECK(t.wire == WireType::I32);

    t = r.read_tag();
    CHECK_EQ(t.field, 1u);
    CHECK(t.wire == WireType::I64);
}

TEST(tag_field_zero_throws) {
    Bytes b = {0x00};
    Reader r = reader(b);
    CHECK_THROWS(r.read_tag());
}

TEST(tag_group_wire_types_throw) {
    // start group (3), end group (4) and undefined types (6, 7)
    for (uint8_t wire : {3, 4, 6, 7}) {
        Bytes b = {static_cast<uint8_t>((1 << 3) | wire)};
        Reader r = reader(b);
        CHECK_THROWS(r.read_tag());
    }
}

// TESTS - FIXED

TEST(fixed32_and_float) {
    // little endian
    Bytes b = {0x78, 0x56, 0x34, 0x12,
               0x00, 0x00, 0x80, 0x3F,    // 1.0f
               0x00, 0x00, 0x00, 0xC0};   // -2.0f
    Reader r = reader(b);
    CHECK_EQ(r.read_fixed32(), 0x12345678u);
    CHECK_EQ(r.read_float(), 1.0f);
    CHECK_EQ(r.read_float(), -2.0f);
    CHECK(r.eof());
}

TEST(fixed64_and_double) {
    Bytes b = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
               0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x3F};   // 0.5
    Reader r = reader(b);
    CHECK_EQ(r.read_fixed64(), 0x0102030405060708ull);
    CHECK_EQ(r.read_double(), 0.5);
    CHECK(r.eof());
}

TEST(fixed_truncated_throws) {
    Bytes b = {0x00, 0x00, 0x80};
    Reader r = reader(b);
    CHECK_THROWS(r.read_float());

    Bytes b8(7, 0x00);
    Reader r8 = reader(b8);
    CHECK_THROWS(r8.read_double());
}

// TESTS - LENGTH DELIMITED

TEST(string_field) {
    // field 2 = "abc"
    Bytes b = {0x12, 0x03, 'a', 'b', 'c'};
    Reader r = reader(b);
    Tag t = r.read_tag();
    CHECK_EQ(t.field, 2u);
    CHECK_EQ(r.read_string(), "abc");
    CHECK(r.eof());
}

TEST(empty_bytes) {
    Bytes b = {0x00};
    Reader r = reader(b);
    CHECK(r.read_bytes().empty());
    CHECK(r.eof());
}

TEST(length_past_end_throws) {
    Bytes b = {0x05, 'a', 'b'};
    Reader r = reader(b);
    CHECK_THROWS(r.read_bytes());
}

TEST(nested_message) {
    // field 1 = message { field 1 = 150 }, then field 2 = 7
    Bytes b = {0x0A, 0x03, 0x08, 0x96, 0x01, 0x10, 0x07};
    Reader r = reader(b);

    Tag t = r.read_tag();
    CHECK_EQ(t.field, 1u);
    Reader inner = r.read_message();
    CHECK_EQ(inner.read_tag().field, 1u);
    CHECK_EQ(inner.read_varint(), 150u);
    CHECK(inner.eof());

    // outer reader continues after the sub-message
    t = r.read_tag();
    CHECK_EQ(t.field, 2u);
    CHECK_EQ(r.read_varint(), 7u);
    CHECK(r.eof());
}

TEST(nested_error_reports_absolute_offset) {
    // inner message holds a varint starting at byte 3 that runs off the
    // end, the error offset is absolute (4), not relative to the inner reader
    Bytes b = {0x0A, 0x02, 0x08, 0x80};
    Reader r = reader(b);
    r.read_tag();
    Reader inner = r.read_message();
    inner.read_tag();
    try {
        inner.read_varint();
        throw std::runtime_error("expected ParseError");
    } catch (const ParseError& e) {
        CHECK_EQ(e.offset, 4u);
    }
}

// TESTS - SKIP

TEST(skip_each_wire_type) {
    Bytes b = {
        0x08, 0x96, 0x01,                                   // 1: varint
        0x11, 1, 2, 3, 4, 5, 6, 7, 8,                       // 2: i64
        0x1A, 0x02, 'h', 'i',                               // 3: len
        0x25, 1, 2, 3, 4,                                   // 4: i32
        0x28, 0x2A,                                         // 5: varint 42
    };
    Reader r = reader(b);
    for (int i = 0; i < 4; ++i) {
        Tag t = r.read_tag();
        r.skip(t.wire);
    }
    Tag t = r.read_tag();
    CHECK_EQ(t.field, 5u);
    CHECK_EQ(r.read_varint(), 42u);
    CHECK(r.eof());
}

TEST(skip_truncated_throws) {
    Bytes b = {0x11, 1, 2, 3};
    Reader r = reader(b);
    Tag t = r.read_tag();
    CHECK_THROWS(r.skip(t.wire));
}

// TESTS - REPEATED

TEST(repeated_int64_unpacked) {
    // field 1 repeated as separate entries: 2, 3
    Bytes b = {0x08, 0x02, 0x08, 0x03};
    Reader r = reader(b);
    std::vector<int64_t> out;
    while (!r.eof()) {
        Tag t = r.read_tag();
        r.read_repeated_int64(t.wire, out);
    }
    CHECK_EQ(out, (std::vector<int64_t>{2, 3}));
}

TEST(repeated_int64_packed) {
    // field 1 packed: 1, 300, -1
    Bytes b = {0x0A, 0x0D, 0x01, 0xAC, 0x02,
               0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    Reader r = reader(b);
    std::vector<int64_t> out;
    Tag t = r.read_tag();
    r.read_repeated_int64(t.wire, out);
    CHECK_EQ(out, (std::vector<int64_t>{1, 300, -1}));
    CHECK(r.eof());
}

TEST(repeated_float_packed_and_unpacked) {
    Bytes b = {0x0A, 0x08, 0x00, 0x00, 0x80, 0x3F,       // packed 1.0, 2.0
                           0x00, 0x00, 0x00, 0x40,
               0x0D, 0x00, 0x00, 0x40, 0x40};            // unpacked 3.0
    Reader r = reader(b);
    std::vector<float> out;
    while (!r.eof()) {
        Tag t = r.read_tag();
        r.read_repeated_float(t.wire, out);
    }
    CHECK_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(repeated_double_packed) {
    Bytes b = {0x0A, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x3F};
    Reader r = reader(b);
    std::vector<double> out;
    Tag t = r.read_tag();
    r.read_repeated_double(t.wire, out);
    CHECK_EQ(out, (std::vector<double>{0.5}));
}

TEST(repeated_wrong_wire_type_throws) {
    Bytes b = {0x0D, 0x00, 0x00, 0x80, 0x3F};
    Reader r = reader(b);
    Tag t = r.read_tag();
    std::vector<int64_t> out;
    CHECK_THROWS(r.read_repeated_int64(t.wire, out));
}

TEST(repeated_packed_partial_element_throws) {
    // packed floats with 6 bytes, the second float is cut off
    Bytes b = {0x0A, 0x06, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00};
    Reader r = reader(b);
    Tag t = r.read_tag();
    std::vector<float> out;
    CHECK_THROWS(r.read_repeated_float(t.wire, out));
}

// TESTS - REALISTIC

TEST(decode_tensor_like_message) {
    // shaped like an onnx TensorProto:
    //   dims (1) = 2, 3   data_type (2) = 1   name (8) = "w"
    //   float_data (4, packed) = 0.5   unknown field 99 skipped
    Bytes b = {0x08, 0x02, 0x08, 0x03,
               0x10, 0x01,
               0x42, 0x01, 'w',
               0x22, 0x04, 0x00, 0x00, 0x00, 0x3F,
               0x98, 0x06, 0x07};

    std::vector<int64_t> dims;
    std::vector<float> floats;
    int32_t dtype = 0;
    std::string name;

    Reader r = reader(b);
    while (!r.eof()) {
        Tag t = r.read_tag();
        switch (t.field) {
            case 1: r.read_repeated_int64(t.wire, dims); break;
            case 2: dtype = r.read_int32(); break;
            case 4: r.read_repeated_float(t.wire, floats); break;
            case 8: name = r.read_string(); break;
            default: r.skip(t.wire); break;
        }
    }
    CHECK_EQ(dims, (std::vector<int64_t>{2, 3}));
    CHECK_EQ(dtype, 1);
    CHECK_EQ(name, "w");
    CHECK_EQ(floats, (std::vector<float>{0.5f}));
}
