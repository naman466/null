# sabotages for GlobalAveragePool / ReduceMean (commit 6ded8f9): every one was caught
CHECK = ("check_reduce.py", 150)

SABOTAGES = [
    ("sum in float", "runtime/null/reduce.h", "        double sum = 0.0;", "        float sum = 0.0f;"),
    ("divide by the kept count", "runtime/null/reduce.h",
     "y[o] = (float)(sum / (double)count);", "y[o] = (float)(sum / (double)kept);"),
    ("read kept dims with reduced strides", "runtime/null/reduce.h",
     "            base += rest % p->shape[d] * stride[d];", "            base += rest % p->shape[d];"),
    ("merge any neighbours", "src/backend/emit_c.cpp",
     "if (!flags.empty() && flags.back() == reduce[i])", "if (!flags.empty())"),
    ("keepdims ignored", "src/analysis/shape_inference.cpp",
     "        else if (keepdims) out.push_back(1);", "        else out.push_back(1);"),
    ("duplicates allowed", "src/analysis/shape_inference.cpp",
     '        if (reduce[d]) c.fail("axes " + shape_of(axes) + " name dim " + std::to_string(d) + " twice");\n',
     "        (void)0;\n"),
    ("noop_with_empty_axes ignored", "src/analysis/shape_inference.cpp",
     '        if (c.node.attr_int("noop_with_empty_axes", 0) == 0) reduce.assign(reduce.size(), true);',
     "        reduce.assign(reduce.size(), true);"),
    ("global pool from dim 1", "src/analysis/shape_inference.cpp",
     "        for (size_t i = 2; i < reduce.size(); ++i) reduce[i] = true;",
     "        for (size_t i = 1; i < reduce.size(); ++i) reduce[i] = true;"),
]
