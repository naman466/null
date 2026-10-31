# sabotages for Shape, Gather, Unsqueeze, Squeeze, Concat and the values they compute
CHECK = ("check_shape_ops.py", 200)

SABOTAGES = [
    ("Shape: negative start and end not counted from the end", "src/analysis/shape_inference.cpp",
     "return std::clamp<int64_t>(v < 0 ? v + rank : v, 0, rank);", "return std::clamp<int64_t>(v, 0, rank);"),
    ("Gather: negative indices fold to 0", "src/analysis/shape_inference.cpp",
     "pos.push_back(*e < 0 ? *e + n : *e);", "pos.push_back(*e < 0 ? 0 : *e);"),
    ("Gather: the indices' dims left out of the shape", "src/analysis/shape_inference.cpp",
     "    out.insert(out.end(), idx.begin(), idx.end());\n", "    (void)idx;\n"),
    ("Unsqueeze: axes counted on the input's rank", "src/analysis/shape_inference.cpp",
     "    const int64_t rank = static_cast<int64_t>(x.size()) + c.in(1).num_elements();",
     "    const int64_t rank = static_cast<int64_t>(x.size()) + 1;"),
    ("Squeeze without axes removes nothing", "src/analysis/shape_inference.cpp",
     "            removed[d] = x[d] == 1;", "            removed[d] = false;"),
    ("Concat: every slice takes the first block", "src/analysis/shape_inference.cpp",
     "v.push_back(values[i][static_cast<size_t>(o * block + k)]);", "v.push_back(values[i][static_cast<size_t>(k)]);"),
    ("Reshape: an unknown target entry taken as 1", "src/analysis/shape_inference.cpp",
     "            out[i] = -1;                    // known once the dynamic dims are fixed",
     "            out[i] = 1;"),
    ("values not passed on within a pass", "src/analysis/shape_inference.cpp",
     "                values[name] = *t.value;\n", "                (void)0;\n"),
    ("emitter: computed values not recognised", "src/backend/emit_c.cpp",
     "        if (!v && info(out).num_elements() != 0) return false;", "        if (true) return false;"),
]
