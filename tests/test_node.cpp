#include "test_framework.h"
#include "core/tensor.h"
#include "core/node.h"

using namespace null;

TEST(attribute_typed_getters) {
    CHECK_EQ(Attribute("i", int64_t{3}).as_int(), 3);
    CHECK_NEAR(Attribute("f", 0.5f).as_float(), 0.5f, 1e-6f);
    CHECK_EQ(Attribute("s", std::string("NOTSET")).as_string(), "NOTSET");
    CHECK_EQ(Attribute("is", std::vector<int64_t>{1, 2}).as_ints(),
             (std::vector<int64_t>{1, 2}));
    CHECK_EQ(Attribute("fs", std::vector<float>{1.5f}).as_floats(),
             (std::vector<float>{1.5f}));
}

TEST(attribute_tensor_value) {
    Tensor t("value", {2}, DType::FLOAT32);
    t.set_float_data({1.0f, 2.0f});
    Attribute a("value", t);
    CHECK(a.has_tensor());
    CHECK_EQ(a.as_tensor().num_elements(), 2);
    CHECK_NEAR(a.as_tensor().float_ptr()[1], 2.0f, 1e-6f);
}

TEST(attribute_has_type) {
    Attribute a("axis", int64_t{1});
    CHECK(a.has_int());
    CHECK(!a.has_float());
    CHECK(!a.has_string());
    CHECK(!a.has_ints());
    CHECK(!a.has_floats());
    CHECK(!a.has_tensor());
}

TEST(attribute_wrong_type_throws) {
    Attribute a("alpha", 1.0f);
    CHECK_THROWS(a.as_int());
    CHECK_THROWS(a.as_string());
    CHECK_THROWS(a.as_ints());
    CHECK_THROWS(a.as_floats());
    CHECK_THROWS(a.as_tensor());
}

TEST(node_construction) {
    Node n("gemm", OpType::Gemm, {"a", "b", "c"}, {"y"});
    CHECK_EQ(n.name, "gemm");
    CHECK_EQ(n.op, OpType::Gemm);
    CHECK_EQ(n.inputs.size(), 3u);
    CHECK_EQ(n.outputs, (std::vector<std::string>{"y"}));
    CHECK(!n.is_dead);
    CHECK(n.attrs.empty());

    Node empty;
    CHECK_EQ(empty.op, OpType::Unknown);
}

TEST(node_attr_present) {
    Node n("conv", OpType::Conv, {"x", "w"}, {"y"});
    n.add_attr("group", int64_t{2});
    n.add_attr("epsilon", 1e-5f);
    n.add_attr("auto_pad", std::string("SAME_UPPER"));
    n.add_attr("pads", std::vector<int64_t>{1, 1, 1, 1});
    n.add_attr("scales", std::vector<float>{2.0f, 2.0f});

    CHECK(n.has_attr("group"));
    CHECK_EQ(n.attr_int("group"), 2);
    CHECK_NEAR(n.attr_float("epsilon"), 1e-5f, 1e-9f);
    CHECK_EQ(n.attr_str("auto_pad"), "SAME_UPPER");
    CHECK_EQ(n.attr_ints("pads"), (std::vector<int64_t>{1, 1, 1, 1}));
    CHECK_EQ(n.attr_floats("scales"), (std::vector<float>{2.0f, 2.0f}));
    CHECK_EQ(n.get_attr("group").name, "group");
}

TEST(node_attr_int_literal_is_int) {
    // a plain int literal must select the INT alternative, not FLOAT
    Node n("softmax", OpType::Softmax, {"x"}, {"y"});
    n.add_attr("axis", 1);
    CHECK(n.get_attr("axis").has_int());
    CHECK_EQ(n.attr_int("axis"), 1);
}

TEST(node_attr_defaults) {
    Node n("gemm", OpType::Gemm, {"a", "b"}, {"y"});
    CHECK(!n.has_attr("alpha"));
    CHECK_EQ(n.attr_int("transA"), 0);
    CHECK_EQ(n.attr_int("transB", 1), 1);
    CHECK_NEAR(n.attr_float("alpha", 1.0f), 1.0f, 1e-6f);
    CHECK_EQ(n.attr_str("mode", "constant"), "constant");
    CHECK(n.attr_ints("pads").empty());
    CHECK_EQ(n.attr_floats("scales", {1.0f}), (std::vector<float>{1.0f}));
}

TEST(node_get_attr_missing_throws) {
    Node n("relu", OpType::Relu, {"x"}, {"y"});
    CHECK_THROWS(n.get_attr("alpha"));
}

TEST(node_attr_wrong_type_throws) {
    Node n("gemm", OpType::Gemm, {"a", "b"}, {"y"});
    n.add_attr("alpha", 1.0f);
    // present but wrong type throws instead of falling back to the default
    CHECK_THROWS(n.attr_int("alpha", 7));
    CHECK_THROWS(n.attr_str("alpha"));
}

TEST(optype_round_trip) {
    // every onnx op maps to its name and back
    for (int i = static_cast<int>(OpType::Add); i <= static_cast<int>(OpType::Identity); ++i) {
        OpType op = static_cast<OpType>(i);
        if (op == OpType::ReluFused) continue;
        std::string s = optype_to_str(op);
        CHECK(s != "Unknown");
        if (str_to_optype(s) != op)
            throw std::runtime_error("optype round trip failed for " + s);
    }
}

TEST(optype_unknown_and_internal) {
    CHECK_EQ(str_to_optype("LSTM"), OpType::Unknown);
    CHECK_EQ(str_to_optype("relu"), OpType::Unknown);   // names are case sensitive
    CHECK_EQ(optype_to_str(OpType::Unknown), "Unknown");
    // fused ops are optimizer-only and never parsed from a model
    CHECK_EQ(optype_to_str(OpType::ReluFused), "ReluFused");
    CHECK_EQ(str_to_optype("ReluFused"), OpType::Unknown);
}
