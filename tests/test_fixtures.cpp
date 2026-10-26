#include "test_framework.h"
#include "frontend/onnx.h"
#include "analysis/shape_inference.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// loads the real .onnx files in tests/fixtures (written by the official onnx
// library, see make_fixtures.py) and checks them against manifest.txt

#ifndef NULL_FIXTURE_DIR
#error "NULL_FIXTURE_DIR must point at tests/fixtures"
#endif

using namespace null;
using namespace null::onnx;

namespace {

struct ValueLine {
    std::string name;
    int dtype;
    std::vector<int64_t> dims;
};

struct NodeLine {
    std::string op;
    std::vector<std::string> inputs, outputs, attrs;
};

struct InitLine {
    std::string name;
    int dtype;
    std::string fnv;
    std::vector<int64_t> dims;
};

struct Expected {
    bool ok = false;
    std::string error;   // substring, for fixtures that must fail
    int64_t ir_version = 0, opset = 0;
    std::string graph;
    std::vector<ValueLine> inputs, outputs;
    std::vector<NodeLine> nodes;
    std::vector<InitLine> inits;
};

std::string fixture_path(const std::string& file) {
    return std::string(NULL_FIXTURE_DIR) + "/" + file;
}

// "a,b,~" -> {"a", "b", ""}, "-" -> {}
std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> out;
    if (s == "-") return out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item == "~" ? "" : item);
    return out;
}

std::vector<int64_t> read_dims(std::istringstream& in) {
    std::vector<int64_t> dims;
    int64_t d;
    while (in >> d) dims.push_back(d);
    return dims;
}

std::map<std::string, Expected> read_manifest() {
    std::ifstream f(fixture_path("manifest.txt"));
    if (!f) throw std::runtime_error("cannot open manifest.txt in " NULL_FIXTURE_DIR);

    std::map<std::string, Expected> all;
    Expected* cur = nullptr;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "model") {
            std::string file, status;
            in >> file >> status;
            cur = &all[file];
            cur->ok = status == "ok";
            if (cur->ok) {
                in >> cur->ir_version >> cur->opset >> cur->graph;
            } else {
                std::getline(in >> std::ws, cur->error);
            }
        } else if (!cur) {
            throw std::runtime_error("manifest line outside a model: " + line);
        } else if (kind == "input" || kind == "output") {
            ValueLine v;
            in >> v.name >> v.dtype;
            v.dims = read_dims(in);
            (kind == "input" ? cur->inputs : cur->outputs).push_back(v);
        } else if (kind == "node") {
            NodeLine n;
            std::string ins, outs, attrs;
            in >> n.op >> ins >> outs >> attrs;
            n.inputs = split_list(ins);
            n.outputs = split_list(outs);
            n.attrs = split_list(attrs);
            cur->nodes.push_back(n);
        } else if (kind == "init") {
            InitLine t;
            in >> t.name >> t.dtype >> t.fnv;
            t.dims = read_dims(in);
            cur->inits.push_back(t);
        } else if (kind == "end") {
            cur = nullptr;
        } else {
            throw std::runtime_error("unknown manifest line: " + line);
        }
    }
    return all;
}

const std::map<std::string, Expected>& manifest() {
    static const std::map<std::string, Expected> m = read_manifest();
    return m;
}

std::string fnv1a64(const std::vector<uint8_t>& data) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (uint8_t b : data) { h ^= b; h *= 0x100000001B3ull; }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

void fail(const std::string& file, const std::string& msg) {
    throw std::runtime_error(file + ": " + msg);
}

void check_value(const std::string& file, const Graph& g, const ValueLine& v) {
    const Tensor* t = g.get_tensor_info(v.name);
    if (!t) fail(file, "no metadata for '" + v.name + "'");
    if (static_cast<int>(t->dtype) != v.dtype)
        fail(file, "'" + v.name + "' has dtype " + std::to_string(static_cast<int>(t->dtype)));
    if (t->shape != v.dims) fail(file, "'" + v.name + "' has shape " + t->shape_str());
}

void check_fixture(const std::string& file) {
    auto it = manifest().find(file);
    if (it == manifest().end()) fail(file, "not in manifest.txt");
    const Expected& e = it->second;

    if (!e.ok) {
        try {
            load_model(fixture_path(file));
        } catch (const OnnxError& err) {
            if (std::string(err.what()).find(e.error) == std::string::npos)
                fail(file, std::string("wrong error: ") + err.what());
            return;
        }
        fail(file, "loaded, but should fail with '" + e.error + "'");
    }

    Model m = load_model(fixture_path(file));
    const Graph& g = m.graph;
    if (m.ir_version != e.ir_version) fail(file, "ir_version " + std::to_string(m.ir_version));
    if (m.opset_version != e.opset) fail(file, "opset " + std::to_string(m.opset_version));
    if (m.producer_name != "null-fixtures") fail(file, "producer '" + m.producer_name + "'");
    if (g.name != e.graph) fail(file, "graph name '" + g.name + "'");

    // inputs and outputs, in order, with metadata
    if (g.inputs.size() != e.inputs.size()) fail(file, "input count " + std::to_string(g.inputs.size()));
    for (size_t i = 0; i < e.inputs.size(); ++i) {
        if (g.inputs[i] != e.inputs[i].name) fail(file, "input " + std::to_string(i) + " is '" + g.inputs[i] + "'");
        check_value(file, g, e.inputs[i]);
    }
    if (g.outputs.size() != e.outputs.size()) fail(file, "output count " + std::to_string(g.outputs.size()));
    for (size_t i = 0; i < e.outputs.size(); ++i) {
        if (g.outputs[i] != e.outputs[i].name) fail(file, "output " + std::to_string(i) + " is '" + g.outputs[i] + "'");
        check_value(file, g, e.outputs[i]);
    }

    // nodes, in file order
    if (g.nodes.size() != e.nodes.size()) fail(file, "node count " + std::to_string(g.nodes.size()));
    for (size_t i = 0; i < e.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        const NodeLine& en = e.nodes[i];
        std::string where = "node " + std::to_string(i) + " (" + n.name + ")";
        if (n.name.empty()) fail(file, where + " has no name");
        if (optype_to_str(n.op) != en.op) fail(file, where + " is " + optype_to_str(n.op));
        if (n.inputs != en.inputs) fail(file, where + " inputs differ");
        if (n.outputs != en.outputs) fail(file, where + " outputs differ");
        std::vector<std::string> attrs;
        for (auto& [name, _] : n.attrs) attrs.push_back(name);
        std::sort(attrs.begin(), attrs.end());
        if (attrs != en.attrs) fail(file, where + " attributes differ");
    }

    // initializers, byte for byte
    if (g.initializers.size() != e.inits.size())
        fail(file, "initializer count " + std::to_string(g.initializers.size()));
    for (const InitLine& ei : e.inits) {
        auto t = g.initializers.find(ei.name);
        if (t == g.initializers.end()) fail(file, "missing initializer '" + ei.name + "'");
        if (static_cast<int>(t->second.dtype) != ei.dtype) fail(file, "'" + ei.name + "' dtype differs");
        if (t->second.shape != ei.dims) fail(file, "'" + ei.name + "' has shape " + t->second.shape_str());
        if (fnv1a64(t->second.data) != ei.fnv) fail(file, "'" + ei.name + "' data differs");
        if (g.is_graph_input(ei.name)) fail(file, "initializer '" + ei.name + "' listed as an input");
    }

    if (!g.validate().empty()) fail(file, "graph does not validate");
}

} // namespace

// TESTS - FIXTURES

TEST(manifest_matches_fixture_files) {
    // every .onnx file has a manifest entry and every entry has a file
    std::vector<std::string> files;
    for (auto& entry : std::filesystem::directory_iterator(NULL_FIXTURE_DIR))
        if (entry.path().extension() == ".onnx") files.push_back(entry.path().filename().string());
    std::sort(files.begin(), files.end());

    std::vector<std::string> listed;
    for (auto& [file, _] : manifest()) listed.push_back(file);
    CHECK_EQ(files, listed);
    CHECK(files.size() >= 10u);
}

TEST(fixture_relu)                   { check_fixture("relu.onnx"); }
TEST(fixture_linear)                 { check_fixture("linear.onnx"); }
TEST(fixture_mlp)                    { check_fixture("mlp.onnx"); }
TEST(fixture_cnn)                    { check_fixture("cnn.onnx"); }
TEST(fixture_optionals)              { check_fixture("optionals.onnx"); }
TEST(fixture_initializers_as_inputs) { check_fixture("initializers_as_inputs.onnx"); }
TEST(fixture_opset11)                { check_fixture("opset11.onnx"); }
TEST(fixture_unsupported_ops)        { check_fixture("unsupported_ops.onnx"); }
TEST(fixture_float16_input)          { check_fixture("float16_input.onnx"); }
TEST(fixture_external_data)          { check_fixture("external_data.onnx"); }

TEST(fixture_cnn_details) {
    // spot checks on values the manifest doesn't capture
    Model m = load_model(fixture_path("cnn.onnx"));
    const Node& conv = m.graph.nodes[0];
    CHECK_EQ(conv.name, "Conv_0");   // unnamed in the file
    CHECK_EQ(conv.attr_ints("kernel_shape"), (std::vector<int64_t>{3, 3}));
    CHECK_EQ(conv.attr_ints("pads"), (std::vector<int64_t>{1, 1, 1, 1}));
    CHECK_NEAR(m.graph.nodes[1].attr_float("epsilon"), 1e-5f, 1e-9f);

    // batch norm variance was generated strictly positive
    const Tensor& var = m.graph.initializers.at("bn.var");
    for (int64_t i = 0; i < var.num_elements(); ++i) CHECK(var.float_ptr()[i] > 0.0f);
}

TEST(fixture_mlp_dynamic_batch) {
    Model m = load_model(fixture_path("mlp.onnx"));
    CHECK_EQ(m.graph.get_tensor_info("x")->num_elements(), -1);
    CHECK_EQ(m.graph.nodes[1].attr_int("transB"), 1);
}

TEST(fixture_optionals_constant) {
    Model m = load_model(fixture_path("optionals.onnx"));
    const Tensor& two = m.graph.nodes[0].get_attr("value").as_tensor();
    CHECK(two.shape.empty());
    CHECK_EQ(two.float_ptr()[0], 2.0f);
    CHECK_EQ(m.graph.initializers.at("shape").int64_ptr()[0], 3);
}

// TESTS - SHAPE INFERENCE ON FIXTURES

TEST(fixture_shapes_mlp) {
    // every intermediate of the real mlp fixture gets a type
    Model m = load_model(fixture_path("mlp.onnx"));
    infer_shapes(m.graph);
    auto shape = [&](const char* name) { return m.graph.get_tensor_info(name)->shape; };
    CHECK_EQ(shape("xf"), (std::vector<int64_t>{-1, 16}));
    CHECK_EQ(shape("h"), (std::vector<int64_t>{-1, 8}));
    CHECK_EQ(shape("hr"), (std::vector<int64_t>{-1, 8}));
    CHECK_EQ(shape("logits"), (std::vector<int64_t>{-1, 3}));
    CHECK_EQ(shape("probs"), (std::vector<int64_t>{-1, 3}));
}

TEST(fixture_shapes_linear_relu_optionals) {
    Model linear = load_model(fixture_path("linear.onnx"));
    infer_shapes(linear.graph);
    CHECK_EQ(linear.graph.get_tensor_info("mm")->shape, (std::vector<int64_t>{1, 3}));

    Model relu = load_model(fixture_path("relu.onnx"));
    infer_shapes(relu.graph);
    CHECK_EQ(relu.graph.get_tensor_info("y")->shape, (std::vector<int64_t>{1, 4}));

    // Constant, Mul, Dropout with an omitted mask, Reshape with an initializer
    // shape, and Clip with an omitted min
    Model opt = load_model(fixture_path("optionals.onnx"));
    infer_shapes(opt.graph);
    CHECK_EQ(opt.graph.get_tensor_info("two")->shape, (std::vector<int64_t>{}));
    CHECK_EQ(opt.graph.get_tensor_info("r")->shape, (std::vector<int64_t>{3, 2}));
    CHECK_EQ(opt.graph.get_tensor_info("y")->shape, (std::vector<int64_t>{3, 2}));
}

TEST(fixture_shapes_cnn_stops_at_conv) {
    // Conv has no rule yet, the error names the node
    Model m = load_model(fixture_path("cnn.onnx"));
    try {
        infer_shapes(m.graph);
        throw std::runtime_error("expected ShapeError");
    } catch (const ShapeError& e) {
        CHECK(std::string(e.what()).find("node 'Conv_0' (Conv): no shape inference rule") != std::string::npos);
    }
}
