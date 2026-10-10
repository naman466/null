#pragma once
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "frontend/protobuf.h"

// minimal protobuf encoder for building test messages
// mirrors the reader in src/frontend/protobuf.h

namespace null_test {

class PbWriter {
public:
    std::vector<uint8_t> buf;

    // raw encoding

    PbWriter& varint(uint64_t v) {
        while (v >= 0x80) {
            buf.push_back(static_cast<uint8_t>(v | 0x80));
            v >>= 7;
        }
        buf.push_back(static_cast<uint8_t>(v));
        return *this;
    }

    PbWriter& tag(uint32_t field, null::pb::WireType wire) {
        return varint((static_cast<uint64_t>(field) << 3) | static_cast<uint8_t>(wire));
    }

    PbWriter& fixed32(uint32_t v) {
        for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
        return *this;
    }

    PbWriter& fixed64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
        return *this;
    }

    // fields

    PbWriter& int_field(uint32_t field, int64_t v) {
        tag(field, null::pb::WireType::VARINT);
        return varint(static_cast<uint64_t>(v));
    }

    PbWriter& float_field(uint32_t field, float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        tag(field, null::pb::WireType::I32);
        return fixed32(bits);
    }

    PbWriter& double_field(uint32_t field, double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, 8);
        tag(field, null::pb::WireType::I64);
        return fixed64(bits);
    }

    PbWriter& bytes_field(uint32_t field, std::string_view s) {
        tag(field, null::pb::WireType::LEN);
        varint(s.size());
        buf.insert(buf.end(), s.begin(), s.end());
        return *this;
    }

    PbWriter& message_field(uint32_t field, const PbWriter& msg) {
        tag(field, null::pb::WireType::LEN);
        varint(msg.buf.size());
        buf.insert(buf.end(), msg.buf.begin(), msg.buf.end());
        return *this;
    }

    // packed repeated fields

    PbWriter& packed_ints(uint32_t field, const std::vector<int64_t>& values) {
        PbWriter body;
        for (int64_t v : values) body.varint(static_cast<uint64_t>(v));
        return message_field(field, body);
    }

    PbWriter& packed_floats(uint32_t field, const std::vector<float>& values) {
        PbWriter body;
        for (float v : values) {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            body.fixed32(bits);
        }
        return message_field(field, body);
    }

    PbWriter& packed_doubles(uint32_t field, const std::vector<double>& values) {
        PbWriter body;
        for (double v : values) {
            uint64_t bits;
            std::memcpy(&bits, &v, 8);
            body.fixed64(bits);
        }
        return message_field(field, body);
    }

    null::pb::Reader reader() const { return null::pb::Reader(buf.data(), buf.size()); }
};

} // namespace null_test
