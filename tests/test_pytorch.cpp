#include "test_framework.h"
#include "analysis/shape_inference.h"
#include "analysis/specialize.h"
#include "frontend/onnx.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// real models exported from pytorch (both of its onnx exporters), checked
// against facts recorded by tests/fixtures/pytorch/make_pytorch_fixtures.py:
// their operators, their onnx-inferred shapes, and what null is expected to
// make of them today

#ifndef NULL_FIXTURE_DIR
#error "NULL_FIXTURE_DIR must point at tests/fixtures"
#endif

using namespace null;
using namespace null::onnx;

namespace {

struct Expected {
    std::vector<int64_t> input;                       // as exported, -1 dynamic
    std::map<std::string, int> ops;                   // op type -> count
    std::string kind;                                 // compiles, load-error, shape-error
    std::string message;                              // substring, for the errors
    std::vector<std::pair<std::string, std::vector<int64_t>>> shapes;   // onnx's, batch 1
};

std::string dir() { return std::string(NULL_FIXTURE_DIR) + "/pytorch"; }

std::map<std::string, Expected> read_manifest() {
    std::ifstream f(dir() + "/pytorch_manifest.txt");
    if (!f) throw std::runtime_error("cannot open pytorch_manifest.txt");
    std::map<std::string, Expected> all;
    Expected* cur = nullptr;
    std::string line;
    auto ints = [](std::istringstream& in) {
        std::vector<int64_t> v;
        int64_t d;
        while (in >> d) v.push_back(d);
        return v;
    };
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "model") {
            std::string file;
            in >> file;
            cur = &all[file];
        } else if (!cur) {
            throw std::runtime_error("manifest line outside a model: " + line);
        } else if (kind == "input") {
            std::string name;
            in >> name;
            cur->input = ints(in);
        } else if (kind == "ops") {
            std::string item;
            while (in >> item) {
                size_t colon = item.rfind(':');
                cur->ops[item.substr(0, colon)] = std::stoi(item.substr(colon + 1));
            }
        } else if (kind == "expect") {
            in >> cur->kind;
            std::getline(in >> std::ws, cur->message);
        } else if (kind == "shape") {
            std::string name;
            in >> name;
            cur->shapes.push_back({name, ints(in)});
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

void fail(const std::string& file, const std::string& msg) {
    throw std::runtime_error(file + ": " + msg);
}

std::string dims(const std::vector<int64_t>& v) {
    return Tensor("", v, DType::UNDEFINED).shape_str();
}

void check_model(const std::string& file) {
    auto it = manifest().find(file);
    if (it == manifest().end()) fail(file, "not in pytorch_manifest.txt");
    const Expected& e = it->second;
    const std::string path = dir() + "/" + file;

    if (e.kind == "load-error") {
        try {
            load_model(path);
        } catch (const OnnxError& err) {
            if (std::string(err.what()).find(e.message) == std::string::npos)
                fail(file, std::string("wrong load error: ") + err.what());
            return;
        }
        fail(file, "loads now (expected '" + e.message + "'): update the manifest if null gained support");
    }

    Model m = load_model(path);
    Graph& g = m.graph;

    // the graph null loaded has exactly the operators onnx lists
    std::map<std::string, int> ops;
    for (const Node& n : g.nodes) ops[optype_to_str(n.op)]++;
    if (ops != e.ops) fail(file, "operators differ from what onnx lists");

    // the input as exported, dynamic batch included
    if (g.inputs.size() != 1) fail(file, "expected one input");
    if (g.get_tensor_info(g.inputs[0])->shape != e.input)
        fail(file, "input shape " + dims(g.get_tensor_info(g.inputs[0])->shape) + " vs " + dims(e.input));

    // what nullc --batch 1 does
    set_batch_size(g, 1);
    if (e.kind == "shape-error") {
        try {
            infer_shapes(g);
        } catch (const ShapeError& err) {
            if (std::string(err.what()).find(e.message) == std::string::npos)
                fail(file, std::string("wrong shape error: ") + err.what());
            return;
        }
        fail(file, "shape inference succeeds now (expected '" + e.message + "'): update the manifest");
    }

    if (e.kind != "compiles") fail(file, "unknown expectation '" + e.kind + "'");
    infer_shapes(g);
    if (!dynamic_tensors(g).empty()) fail(file, "dynamic dims remain after --batch 1");
    if (e.shapes.empty()) fail(file, "manifest has no shapes to compare");
    for (const auto& [name, want] : e.shapes) {
        const Tensor* t = g.get_tensor_info(name);
        if (!t) fail(file, "no shape for '" + name + "'");
        if (t->shape != want)
            fail(file, "'" + name + "' inferred as " + dims(t->shape) + ", onnx says " + dims(want));
    }
}

} // namespace

// TESTS - PYTORCH EXPORTS

TEST(pytorch_manifest_matches_files) {
    std::vector<std::string> files;
    for (auto& entry : std::filesystem::directory_iterator(dir()))
        if (entry.path().extension() == ".onnx") files.push_back(entry.path().filename().string());
    std::sort(files.begin(), files.end());
    std::vector<std::string> listed;
    for (auto& [file, _] : manifest()) listed.push_back(file);
    CHECK_EQ(files, listed);
    CHECK(files.size() >= 9u);
}

TEST(pytorch_mlp_legacy_dynamic)       { check_model("mlp__legacy_dyn.onnx"); }
TEST(pytorch_mlp_dynamo_dynamic)       { check_model("mlp__dynamo_dyn.onnx"); }
TEST(pytorch_mlp_view_legacy)          { check_model("mlp_view__legacy.onnx"); }
TEST(pytorch_mlp_view_legacy_dynamic)  { check_model("mlp_view__legacy_dyn.onnx"); }
TEST(pytorch_lenet_legacy)             { check_model("lenet__legacy.onnx"); }
TEST(pytorch_lenet_dynamo)             { check_model("lenet__dynamo.onnx"); }
TEST(pytorch_conv_bn_legacy)           { check_model("conv_bn__legacy.onnx"); }
TEST(pytorch_conv_bn_dynamo)           { check_model("conv_bn__dynamo.onnx"); }
TEST(pytorch_transformer_dynamo)       { check_model("transformer__dynamo.onnx"); }

TEST(pytorch_exporters_fold_batchnorm) {
    // both exporters fold an eval-mode BatchNorm into the conv before it,
    // and the newer one turns the adaptive average pool into ReduceMean
    const auto& legacy = manifest().at("conv_bn__legacy.onnx").ops;
    const auto& dynamo = manifest().at("conv_bn__dynamo.onnx").ops;
    CHECK(!legacy.count("BatchNormalization"));
    CHECK(!dynamo.count("BatchNormalization"));
    CHECK(legacy.count("GlobalAveragePool"));
    CHECK(dynamo.count("ReduceMean"));
}
