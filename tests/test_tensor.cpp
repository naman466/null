#include <cstring>
#include "test_framework.h"
#include "core/tensor.h"

using namespace null;

TEST(tensor_basic_construction) {
    Tensor t("t", {2, 3}, DType::FLOAT32);
    CHECK_EQ(t.name, "t");
    CHECK_EQ(t.shape.size(), 2u);
    CHECK_EQ(t.shape[0], 2);
    CHECK_EQ(t.shape[1], 3);
    CHECK_EQ(t.dtype, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), 6);
    CHECK_EQ(t.byte_size(), 24);
}

TEST(tensor_scalar) {
    Tensor t("s", {}, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), 1);
    CHECK_EQ(t.byte_size(), 4);
}

TEST(tensor_dynamic_dim) {
    Tensor t("d", {-1, 3}, DType::FLOAT32);
    CHECK_EQ(t.num_elements(), -1);
    CHECK_EQ(t.byte_size(), -1);
}

TEST(tensor_set_float_data) {
    Tensor t;
    t.name = "w";
    t.shape = {2, 2};
    t.set_float_data({1.0f, 2.0f, 3.0f, 4.0f});
    CHECK(t.is_constant);
    CHECK_EQ(t.floats().size(), size_t{4});
    CHECK_NEAR(t.floats()[0], 1.0f, 1e-6f);
    CHECK_NEAR(t.floats()[3], 4.0f, 1e-6f);
}

TEST(tensor_values_copied_from_bytes) {
    // the bytes are copied out exactly, -0.0 and a NaN payload included, from
    // a buffer whose first value starts at an odd address
    Tensor t("w", {2}, DType::FLOAT32);
    const uint32_t bits[2] = {0x80000000u, 0x7fc01234u};
    std::vector<uint8_t> shifted(1 + sizeof bits);
    std::memcpy(shifted.data() + 1, bits, sizeof bits);
    t.data.assign(shifted.begin() + 1, shifted.end());
    std::vector<float> v = t.floats();
    CHECK_EQ(v.size(), size_t{2});
    uint32_t got[2];
    std::memcpy(got, v.data(), sizeof got);
    CHECK_EQ(got[0], bits[0]);
    CHECK_EQ(got[1], bits[1]);

    // another dtype, or no data, gives no values; a trailing partial value is dropped
    CHECK(t.int64s().empty());
    t.data.push_back(0);
    CHECK_EQ(t.floats().size(), size_t{2});
    Tensor empty("e", {0}, DType::INT64);
    CHECK(empty.int64s().empty());

    Tensor ints("i", {}, DType::INT64);
    ints.set_int64_data({-1, int64_t{1} << 40});
    CHECK_EQ(ints.int64s(), (std::vector<int64_t>{-1, int64_t{1} << 40}));
    CHECK(ints.floats().empty());
}
