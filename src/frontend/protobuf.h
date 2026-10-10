#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

namespace null::pb {

// minimal protobuf wire format reader
// decodes raw fields only, message schemas (onnx) are layered on top
//
// every field on the wire is a key followed by a payload:
//   key = (field_number << 3) | wire_type, encoded as a varint
// see https://protobuf.dev/programming-guides/encoding/

enum class WireType : uint8_t {
    VARINT = 0,   // int32, int64, uint64, bool, enum
    I64    = 1,   // fixed64, double
    LEN    = 2,   // string, bytes, sub-message, packed repeated
    I32    = 5,   // fixed32, float
    // 3 and 4 are deprecated groups, not supported
};

struct Tag {
    uint32_t field;
    WireType wire;
};

class ParseError : public std::runtime_error {
public:
    ParseError(const std::string& msg, size_t offset)
        : std::runtime_error("protobuf parse error at byte " + std::to_string(offset) + ": " + msg),
          offset(offset) {}
    size_t offset;
};

// reads fields sequentially from a byte buffer it does not own
class Reader {
public:
    Reader(const uint8_t* data, size_t size, size_t base_offset = 0)
        : data_(data), size_(size), base_(base_offset) {}

    explicit Reader(std::string_view bytes)
        : Reader(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()) {}

    bool eof() const { return pos_ >= size_; }
    size_t pos() const { return pos_; }

    // field key
    Tag read_tag();

    // scalar payloads
    uint64_t read_varint();
    int64_t  read_int64()  { return static_cast<int64_t>(read_varint()); }
    int32_t  read_int32()  { return static_cast<int32_t>(read_varint()); }
    bool     read_bool()   { return read_varint() != 0; }
    uint32_t read_fixed32();
    uint64_t read_fixed64();
    float    read_float();
    double   read_double();

    // length-delimited payloads
    std::string_view read_bytes();
    std::string read_string() { return std::string(read_bytes()); }
    Reader read_message();   // sub-reader over the embedded message

    // skip the payload of a field we don't care about
    void skip(WireType wire);

    // repeated fields, accept both packed (LEN) and unpacked encodings
    // and append to out
    void read_repeated_int64(WireType wire, std::vector<int64_t>& out);
    void read_repeated_float(WireType wire, std::vector<float>& out);
    void read_repeated_double(WireType wire, std::vector<double>& out);

private:
    const uint8_t* data_;
    size_t size_;
    size_t base_;    // offset of data_ within the outermost buffer, for errors
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& msg) const;
    void require(size_t n) const;
};

} // namespace null::pb
