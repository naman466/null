#include "frontend/protobuf.h"

#include <cstring>

namespace null::pb {

constexpr uint64_t kMaxFieldNumber = (1u << 29) - 1;

static const char* wire_name(WireType w) {
    switch (w) {
        case WireType::VARINT: return "VARINT";
        case WireType::I64:    return "I64";
        case WireType::LEN:    return "LEN";
        case WireType::I32:    return "I32";
    }
    return "?";
}

void Reader::expect(const Tag& t, WireType wire) const {
    if (t.wire != wire)
        fail("field " + std::to_string(t.field) + " has wire type " + wire_name(t.wire) +
             ", expected " + wire_name(wire));
}

void Reader::fail(const std::string& msg) const {
    throw ParseError(msg, base_ + pos_);
}

// make sure n more bytes are available
void Reader::require(size_t n) const {
    if (n > size_ - pos_) fail("unexpected end of buffer");
}

// varints store 7 bits per byte, least significant group first
// the high bit of each byte is set when more bytes follow
uint64_t Reader::read_varint() {
    uint64_t result = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        require(1);
        uint8_t byte = data_[pos_++];
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) return result;
    }
    // a uint64 needs at most 10 bytes
    fail("varint longer than 10 bytes");
}

Tag Reader::read_tag() {
    uint64_t key = read_varint();
    uint32_t field = static_cast<uint32_t>(key >> 3);   // checked below
    uint8_t wire = static_cast<uint8_t>(key & 0x7);

    // field numbers are 1 .. 2^29 - 1, larger keys must not be truncated
    // into a valid-looking small field number
    if ((key >> 3) > kMaxFieldNumber)
        fail("field number " + std::to_string(key >> 3) + " exceeds 2^29 - 1");
    if (field == 0) fail("invalid field number 0");
    switch (wire) {
        case 0: case 1: case 2: case 5:
            return Tag{field, static_cast<WireType>(wire)};
        default:
            fail("unsupported wire type " + std::to_string(wire));
    }
}

// fixed width values are little endian, assemble byte by byte
// so this does not depend on host endianness
uint32_t Reader::read_fixed32() {
    require(4);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<uint32_t>(data_[pos_ + i]) << (8 * i);
    pos_ += 4;
    return v;
}

uint64_t Reader::read_fixed64() {
    require(8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
    pos_ += 8;
    return v;
}

float Reader::read_float() {
    uint32_t bits = read_fixed32();
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

double Reader::read_double() {
    uint64_t bits = read_fixed64();
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

std::string_view Reader::read_bytes() {
    uint64_t len = read_varint();
    if (len > size_ - pos_) fail("length " + std::to_string(len) + " exceeds buffer");
    std::string_view s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
    pos_ += static_cast<size_t>(len);
    return s;
}

Reader Reader::read_message() {
    std::string_view bytes = read_bytes();
    const uint8_t* payload = reinterpret_cast<const uint8_t*>(bytes.data());
    // keep error offsets relative to the outermost buffer
    return Reader(payload, bytes.size(), base_ + static_cast<size_t>(payload - data_));
}

void Reader::skip(WireType wire) {
    switch (wire) {
        case WireType::VARINT: read_varint();  break;
        case WireType::I64:    require(8); pos_ += 8; break;
        case WireType::LEN:    read_bytes();   break;
        case WireType::I32:    require(4); pos_ += 4; break;
    }
}

void Reader::read_repeated_int64(WireType wire, std::vector<int64_t>& out) {
    if (wire == WireType::VARINT) {
        out.push_back(read_int64());
    } else if (wire == WireType::LEN) {
        Reader packed = read_message();
        while (!packed.eof()) out.push_back(packed.read_int64());
    } else {
        fail("expected varint or packed field");
    }
}

void Reader::read_repeated_float(WireType wire, std::vector<float>& out) {
    if (wire == WireType::I32) {
        out.push_back(read_float());
    } else if (wire == WireType::LEN) {
        Reader packed = read_message();
        while (!packed.eof()) out.push_back(packed.read_float());
    } else {
        fail("expected fixed32 or packed field");
    }
}

void Reader::read_repeated_double(WireType wire, std::vector<double>& out) {
    if (wire == WireType::I64) {
        out.push_back(read_double());
    } else if (wire == WireType::LEN) {
        Reader packed = read_message();
        while (!packed.eof()) out.push_back(packed.read_double());
    } else {
        fail("expected fixed64 or packed field");
    }
}

} // namespace null::pb
