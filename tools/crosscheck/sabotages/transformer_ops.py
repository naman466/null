# sabotages for LayerNormalization, Transpose and float Gather
CHECK = ("check_transformer_ops.py", 300)

SABOTAGES = [
    ("layernorm: the sample variance (n - 1)", "runtime/null/norm.h",
     "sd = sqrt(sq / (double)n + (double)epsilon);", "sd = sqrt(sq / (double)(n - 1) + (double)epsilon);"),
    ("layernorm: epsilon ignored", "runtime/null/norm.h",
     "sd = sqrt(sq / (double)n + (double)epsilon);", "sd = sqrt(sq / (double)n);"),
    ("layernorm: the bias never added", "runtime/null/norm.h",
     "            if (bias) v += bias[i];\n", "            (void)bias;\n"),
    ("layernorm: the axis counted from the end wrongly", "src/backend/emit_c.cpp",
     "        if (axis < 0) axis += static_cast<int64_t>(s.size());\n        int64_t outer = 1, count = 1;",
     "        if (axis < 0) axis += static_cast<int64_t>(s.size()) - 1;\n        int64_t outer = 1, count = 1;"),
    ("transpose: every permutation taken as a no-op", "src/backend/emit_c.cpp",
     "if (std::is_sorted(x_strides.begin(), x_strides.end(), std::greater<int64_t>())) {", "if (true) {"),
    ("transpose: input strides in output order", "src/backend/emit_c.cpp",
     "            x_strides.push_back(stride[static_cast<size_t>(p)]);",
     "            x_strides.push_back(stride[x_strides.size()]);"),
    ("transpose kernel: the odometer does not rewind", "runtime/null/layout.h",
     "            at -= (out_shape[d] - 1) * x_strides[d];\n", "            (void)0;\n"),
    ("gather: the single-index alias ignores the block size", "src/backend/emit_c.cpp",
     "            const int64_t offset = (*indices)[0] * inner;", "            const int64_t offset = (*indices)[0];"),
    ("gather: negative indices not made non-negative", "src/backend/emit_c.cpp",
     "        for (int64_t& k : *indices) if (k < 0) k += count;", "        (void)count;"),
]
