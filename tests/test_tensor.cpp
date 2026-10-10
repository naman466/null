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
    CHECK(t.float_ptr() != nullptr);
    CHECK_NEAR(t.float_ptr()[0], 1.0f, 1e-6f);
    CHECK_NEAR(t.float_ptr()[3], 4.0f, 1e-6f);
}
