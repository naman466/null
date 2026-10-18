#include <iostream>
#include <string>
#include <vector>

#include "core/dump.h"
#include "frontend/onnx.h"

// exit codes
constexpr int kExitOk    = 0;
constexpr int kExitUsage = 1;   // bad command line
constexpr int kExitModel = 2;   // model could not be loaded

static void print_usage(std::ostream& out) {
    out << "Usage: nullc <model.onnx> [--dump]\n"
           "\n"
           "Loads an ONNX model and checks that null can compile it.\n"
           "\n"
           "Options:\n"
           "  --dump      print the loaded graph: inputs, outputs, weights and nodes\n"
           "  -h, --help  show this help\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    std::string model_path;
    bool dump = false;
    for (const std::string& arg : args) {
        if (arg == "-h" || arg == "--help") {
            print_usage(std::cout);
            return kExitOk;
        } else if (arg == "--dump") {
            dump = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "nullc: unknown option '" << arg << "'\n\n";
            print_usage(std::cerr);
            return kExitUsage;
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

    std::cout << "model: " << model_path << " (ir " << model.ir_version << ", opset "
              << model.opset_version;
    if (!model.producer_name.empty()) {
        std::cout << ", " << model.producer_name;
        if (!model.producer_version.empty()) std::cout << " " << model.producer_version;
    }
    std::cout << ")\n";

    if (dump) {
        std::cout << null::dump_graph(model.graph);
    } else {
        const null::Graph& g = model.graph;
        std::cout << "graph '" << g.name << "': " << null::count_str(g.live_node_count(), "node")
                  << ", " << null::count_str(g.initializers.size(), "initializer") << ", "
                  << null::count_str(g.inputs.size(), "input") << ", "
                  << null::count_str(g.outputs.size(), "output") << ", ok\n";
    }
    return kExitOk;
}
