# sabotages for fusing Relu into Gemm and Conv
CHECK = ("check_optimize.py", 150)

SABOTAGES = [
    ("gemm kernel: the fused relu ignored", "runtime/null/matmul.h",
     "y[i * N + j] = relu && out < 0.0f ? 0.0f : out;", "y[i * N + j] = out + 0.0f * (float)relu;"),
    ("gemm kernel: <= instead of <, so -0.0 becomes +0.0", "runtime/null/matmul.h",
     "y[i * N + j] = relu && out < 0.0f ? 0.0f : out;", "y[i * N + j] = relu && out <= 0.0f ? 0.0f : out;"),
    ("conv kernel: the fused relu ignored", "runtime/null/conv.h",
     "= p->relu && acc < 0.0f ? 0.0f : acc;", "= acc;"),
    ("fusion although the value has another reader", "src/opt/optimize.cpp",
     "        if (readers != 1) continue;\n", "        (void)readers;\n"),
    ("fusion into a MatMul, which has no fused form", "src/opt/optimize.cpp",
     "(p->op != OpType::Gemm && p->op != OpType::Conv)", "(p->op != OpType::Gemm && p->op != OpType::Conv && p->op != OpType::MatMul)"),
    ("fusion although the value is a graph output", "src/opt/optimize.cpp",
     "        if (is_graph_output(g, mid)) continue;\n", "        (void)0;\n"),
    ("emitter: the gemm's fused flag not passed", "src/backend/emit_c.cpp",
     "c_cs << \", \" << n.attr_int(kFusedRelu, 0) << \");", "c_cs << \", \" << 0 << \");"),
    ("frontend: the reserved prefix accepted", "src/frontend/onnx.cpp",
     "        if (a.name.rfind(kCompilerAttrPrefix, 0) == 0)\n", "        if (false)\n"),
]
