#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/shape_inference.h"
#include "analysis/specialize.h"
#include "backend/emit_c.h"
#include "core/dump.h"
#include "frontend/onnx.h"

// exit codes
constexpr int kExitOk    = 0;
constexpr int kExitUsage = 1;   // bad command line
constexpr int kExitModel = 2;   // model can't be loaded or compiled as given

static void print_usage(std::ostream& out) {
    out << "Usage: nullc <model.onnx> [options]\n"
           "\n"
           "Loads an ONNX model, infers every tensor's shape, and checks that null\n"
           "can compile it; with -o, writes the compiled model as one C99 file.\n"
           "Compiling needs every shape to be known, so models with dynamic dims\n"
           "(like a variable batch size) need them fixed with --batch or\n"
           "--input-shape.\n"
           "\n"
           "Options:\n"
           "  -o FILE                    write the compiled model to FILE as C99\n"
           "  --batch N                  set the dynamic first dim of every input to N\n"
           "  --input-shape NAME=D0,D1   set the full shape of input NAME (repeatable)\n"
           "  --dump                     print the graph: inputs, outputs, weights and\n"
           "                             nodes with their inferred types (dynamic dims\n"
           "                             are allowed here, shown as ?)\n"
           "  -h, --help                 show this help\n";
}

// whole-string integer parse, nullopt on anything else
static std::optional<int64_t> parse_int(const std::string& s) {
    if (s.empty()) return std::nullopt;
    size_t used = 0;
    try {
        long long v = std::stoll(s, &used);
        if (used != s.size()) return std::nullopt;
        return static_cast<int64_t>(v);
    } catch (...) {
        return std::nullopt;
    }
}

// "x=1,3,224,224" -> {"x", {1, 3, 224, 224}}, split at the last '='
static std::optional<std::pair<std::string, std::vector<int64_t>>> parse_input_shape(const std::string& s) {
    size_t eq = s.rfind('=');
    if (eq == std::string::npos || eq == 0) return std::nullopt;
    std::vector<int64_t> dims;
    std::string rest = s.substr(eq + 1);
    if (!rest.empty()) {
        size_t start = 0;
        while (true) {
            size_t comma = rest.find(',', start);
            auto d = parse_int(rest.substr(start, comma - start));
            if (!d) return std::nullopt;
            dims.push_back(*d);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    return std::make_pair(s.substr(0, eq), dims);
}

static int usage_error(const std::string& msg) {
    std::cerr << "nullc: " << msg << "\n\n";
    print_usage(std::cerr);
    return kExitUsage;
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    std::string model_path;
    bool dump = false;
    std::optional<int64_t> batch;
    std::string output_path;
    std::vector<std::pair<std::string, std::vector<int64_t>>> input_shapes;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];

        // options that take a value, as "--opt value" or "--opt=value"
        auto value_of = [&](const std::string& opt) -> std::optional<std::string> {
            if (arg == opt) {
                if (i + 1 >= args.size()) return std::string();
                return args[++i];
            }
            if (arg.rfind(opt + "=", 0) == 0) return arg.substr(opt.size() + 1);
            return std::nullopt;
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            return kExitOk;
        } else if (arg == "--dump") {
            dump = true;
        } else if (auto v = value_of("-o")) {
            if (v->empty()) return usage_error("-o needs a file name");
            output_path = *v;
        } else if (auto v = value_of("--batch")) {
            batch = parse_int(*v);
            if (!batch || *batch < 1)
                return usage_error("--batch needs a positive integer, got '" + *v + "'");
        } else if (auto v = value_of("--input-shape")) {
            auto parsed = parse_input_shape(*v);
            if (!parsed)
                return usage_error("--input-shape needs NAME=D0,D1,..., got '" + *v + "'");
            input_shapes.push_back(*parsed);
        } else if (!arg.empty() && arg[0] == '-') {
            return usage_error("unknown option '" + arg + "'");
        } else if (model_path.empty()) {
            model_path = arg;
        } else {
            std::cerr << "nullc: only one model can be given, got '" << model_path
                      << "' and '" << arg << "'\n";
            return kExitUsage;
        }
    }

    if (model_path.empty()) {
        print_usage(std::cerr);
        return kExitUsage;
    }

    null::onnx::Model model;
    try {
        model = null::onnx::load_model(model_path);
    } catch (const null::onnx::OnnxError& e) {
        std::cerr << "nullc: error: " << e.detail() << "\n";
        return kExitModel;
    }
    null::Graph& g = model.graph;

    std::cout << "model: " << model_path << " (ir " << model.ir_version << ", opset "
              << model.opset_version;
    if (!model.producer_name.empty()) {
        std::cout << ", " << model.producer_name;
        if (!model.producer_version.empty()) std::cout << " " << model.producer_version;
    }
    std::cout << ")\n";

    // fix dynamic input dims, then infer every other shape from them
    try {
        for (auto& [name, shape] : input_shapes) null::set_input_shape(g, name, shape);
        if (batch && null::set_batch_size(g, *batch) == 0)
            std::cerr << "nullc: note: --batch had no effect, no input has a dynamic first dim\n";
    } catch (const null::ShapeError& e) {
        std::cerr << "nullc: error: " << e.what() << "\n";
        return kExitModel;
    }

    std::string shape_problem;
    try {
        null::infer_shapes(g);
    } catch (const null::ShapeError& e) {
        shape_problem = e.what();
    }

    // --dump is for inspection: show what is known, report problems as warnings
    if (dump) {
        std::cout << null::dump_graph(g);
        if (!shape_problem.empty()) {
            std::cerr << "nullc: warning: shape inference stopped: " << shape_problem << "\n";
        } else if (size_t n = null::dynamic_tensors(g).size()) {
            std::cerr << "nullc: note: " << null::count_str(n, "tensor")
                      << " still have dynamic dims (shown as ?), fix the inputs with "
                         "--batch or --input-shape to compile\n";
        }
        return kExitOk;
    }

    if (!shape_problem.empty()) {
        std::cerr << "nullc: error: shape inference: " << shape_problem << "\n";
        return kExitModel;
    }

    // compiling needs every shape to be static
    std::vector<std::string> dynamic = null::dynamic_tensors(g);
    if (!dynamic.empty()) {
        bool any_input = false;
        for (const std::string& name : dynamic) {
            if (!g.is_graph_input(name)) continue;
            any_input = true;
            std::cerr << "nullc: error: input '" << name << "' has a dynamic shape "
                      << null::type_str(*g.get_tensor_info(name)) << "\n";
        }
        if (!any_input) {
            const null::Tensor* t = g.get_tensor_info(dynamic[0]);
            std::cerr << "nullc: error: tensor '" << dynamic[0] << "' still has "
                      << (t ? "a dynamic shape " + null::type_str(*t) : std::string("no shape"))
                      << " after shape inference\n";
        }
        std::cerr << "nullc: fix dynamic inputs with --batch N or --input-shape NAME=D0,D1,...\n";
        return kExitModel;
    }

    std::cout << "graph '" << g.name << "': " << null::count_str(g.live_node_count(), "node")
              << ", " << null::count_str(g.initializers.size(), "initializer") << ", "
              << null::count_str(g.inputs.size(), "input") << ", "
              << null::count_str(g.outputs.size(), "output") << ", all shapes known, ok\n";

    if (!output_path.empty()) {
        std::string source;
        try {
            source = null::emit_c(g);
        } catch (const null::EmitError& e) {
            std::cerr << "nullc: error: code generation: " << e.what() << "\n";
            return kExitModel;
        }
        std::ofstream out(output_path, std::ios::binary);
        if (!out || !(out << source) || !out.flush()) {
            std::cerr << "nullc: error: cannot write '" << output_path << "'\n";
            return kExitModel;
        }
        std::cout << "wrote " << output_path << " (" << null::bytes_str(source.size()) << ")\n";
    }
    return kExitOk;
}
