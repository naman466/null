# sabotages for constant folding, and for the emitter skipping what no output needs
CHECK = ("check_optimize.py", 150)

SABOTAGES = [
    ("Sub folded with the Add kernel", "src/opt/optimize.cpp",
     "n.op == OpType::Sub ? null_sub_bcast_f32", "n.op == OpType::Sub ? null_add_bcast_f32"),
    ("the second operand's strides from the first's shape", "src/opt/optimize.cpp",
     "static_cast<int>(w.shape.size()), w.shape.data(), sb);", "static_cast<int>(x.shape.size()), x.shape.data(), sb);"),
    ("Transpose folded with the strides in output order", "src/opt/optimize.cpp",
     "            for (int64_t p : perm) xs.push_back(stride[static_cast<size_t>(p)]);",
     "            for (int64_t p : perm) xs.push_back(stride[xs.size()] + 0 * p);"),
    ("graph outputs folded", "src/opt/optimize.cpp",
     "        if (is_graph_output(g, out)) continue;\n        const Tensor* info", "        const Tensor* info"),
    ("initializers that are graph inputs folded", "src/opt/optimize.cpp",
     "if (in.empty() || !g.is_initializer(in) || g.is_graph_input(in)) constant = false;",
     "if (in.empty() || !g.is_initializer(in)) constant = false;"),
    ("results larger than their inputs folded", "src/opt/optimize.cpp",
     "                info->num_elements() <= inputs_size)", "                inputs_size >= 0)"),
    ("a Constant's tensor keeps the attribute's name", "src/opt/optimize.cpp",
     "        t.name = name;\n", "        (void)name;\n"),
    ("computed shape values not folded", "src/opt/optimize.cpp",
     "std::optional<std::vector<int64_t>> v = computed_value(g, out)", "std::optional<std::vector<int64_t>> v = std::nullopt"),
    ("emitter: code for what no output needs", "src/backend/emit_c.cpp",
     "            if (wanted.count(n)) {", "            if (true) {"),
]
