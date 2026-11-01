# sabotages for the optimiser: removing identities, dead code and the pass manager
CHECK = ("check_optimize.py", 150)

SABOTAGES = [
    ("identities: consumers not rewired", "src/opt/optimize.cpp",
     "            rename_uses(g, out, in);\n", "            (void)0;\n"),
    ("identities: the producer does not take over the output's name", "src/opt/optimize.cpp",
     "                if (o == in) o = out;", "                (void)o;"),
    ("identities: a Dropout removed although its mask is used", "src/opt/optimize.cpp",
     " && is_used(g, n.outputs[1])) continue;  // the mask", " && false) continue;  // the mask"),
    ("dead code: the graph outputs are not needed", "src/opt/optimize.cpp",
     "    std::unordered_set<std::string> needed(g.outputs.begin(), g.outputs.end());",
     "    std::unordered_set<std::string> needed;"),
    ("dead code: a needed node's inputs are not needed", "src/opt/optimize.cpp",
     "            if (!in.empty()) needed.insert(in);", "            (void)in;"),
    ("dead code: initializers that are graph inputs dropped", "src/opt/optimize.cpp",
     "if (!needed.count(it->first) && !g.is_graph_input(it->first)) {", "if (!needed.count(it->first)) {"),
    ("pass manager: a single round", "src/opt/optimize.cpp",
     "        if (changed == 0) break;", "        break;"),
    ("--no-opt ignored", "src/main.cpp",
     "            optimize = false;", "            optimize = true;"),
]
