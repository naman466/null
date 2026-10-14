"""Generate the .onnx test fixtures and their manifest.

The fixtures are written with the official onnx library, so the C++ tests
check null's frontend against real serialized models. Both the .onnx files
and manifest.txt are committed; this script is only needed to regenerate
them after changing it.

    pip install onnx numpy
    python tests/fixtures/make_fixtures.py

manifest.txt describes what load_model must produce for each fixture:

    model <file> ok <ir_version> <opset> <graph_name>
    input <name> <dtype> <dims...>        dims: -1 for dynamic
    output <name> <dtype> <dims...>
    node <op> <inputs> <outputs> <attrs>  lists comma separated, "-" if empty,
                                          an empty name (optional) is written as "~"
    init <name> <dtype> <fnv1a64> <dims...>
    end

    model <file> error <message substring...>
    end

dtype is the onnx TensorProto data type number, which null's DType mirrors.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import set_external_data

HERE = os.path.dirname(os.path.abspath(__file__))
F = TensorProto.FLOAT
rng = np.random.default_rng(5010)
mk, vi = helper.make_node, helper.make_tensor_value_info
manifest = []


def weights(name, shape):
    return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)


def model(graph, opset=17, ir_version=None):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                          producer_name="null-fixtures")
    if ir_version is not None:
        m.ir_version = ir_version
    return m


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def names(xs):
    return ",".join(x if x else "~" for x in xs) or "-"


def value_line(kind, v):
    tt = v.type.tensor_type
    dims = [str(d.dim_value) if d.WhichOneof("value") == "dim_value" else "-1" for d in tt.shape.dim]
    return " ".join([kind, v.name, str(tt.elem_type)] + dims)


def ok(file, m, check=True):
    if check:
        onnx.checker.check_model(m, full_check=True)
    onnx.save(m, os.path.join(HERE, file))
    g = m.graph
    inits = {t.name for t in g.initializer}
    manifest.append(f"model {file} ok {m.ir_version} {m.opset_import[0].version} {g.name}")
    manifest.extend(value_line("input", v) for v in g.input if v.name not in inits)
    manifest.extend(value_line("output", v) for v in g.output)
    for n in g.node:
        manifest.append(f"node {n.op_type} {names(n.input)} {names(n.output)} "
                        f"{names(sorted(a.name for a in n.attribute))}")
    for t in g.initializer:
        arr = np.ascontiguousarray(numpy_helper.to_array(t))
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        dims = " ".join(str(d) for d in t.dims)
        manifest.append(f"init {t.name} {t.data_type} {fnv1a64(arr.tobytes())} {dims}".rstrip())
    manifest.append("end")


def error(file, m, message):
    onnx.save(m, os.path.join(HERE, file))
    manifest.append(f"model {file} error {message}")
    manifest.append("end")


# --- models that must load ---

ok("relu.onnx", model(helper.make_graph(
    [mk("Relu", ["x"], ["y"], name="relu")],
    "relu", [vi("x", F, [1, 4])], [vi("y", F, [1, 4])])))

ok("linear.onnx", model(helper.make_graph(
    [mk("MatMul", ["x", "W"], ["mm"], name="/fc/MatMul"),
     mk("Add", ["mm", "B"], ["y"], name="/fc/Add")],
    "linear", [vi("x", F, [1, 4])], [vi("y", F, [1, 3])],
    initializer=[weights("W", (4, 3)), weights("B", (3,))])))

ok("mlp.onnx", model(helper.make_graph(
    [mk("Flatten", ["x"], ["xf"], name="/flatten", axis=1),
     mk("Gemm", ["xf", "fc1.weight", "fc1.bias"], ["h"], name="/fc1/Gemm", transB=1),
     mk("Relu", ["h"], ["hr"], name="/relu"),
     mk("Gemm", ["hr", "fc2.weight", "fc2.bias"], ["logits"], name="/fc2/Gemm", transB=1),
     mk("Softmax", ["logits"], ["probs"], name="/softmax", axis=1)],
    "mlp", [vi("x", F, ["batch", 1, 4, 4])], [vi("probs", F, ["batch", 3])],
    initializer=[weights("fc1.weight", (8, 16)), weights("fc1.bias", (8,)),
                 weights("fc2.weight", (3, 8)), weights("fc2.bias", (3,))])))

ok("cnn.onnx", model(helper.make_graph(
    [mk("Conv", ["x", "conv.w", "conv.b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     mk("BatchNormalization", ["c", "bn.scale", "bn.bias", "bn.mean", "bn.var"], ["b"], epsilon=1e-5),
     mk("Relu", ["b"], ["r"]),
     mk("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
     mk("GlobalAveragePool", ["p"], ["gap"]),
     mk("Flatten", ["gap"], ["f"]),
     mk("Gemm", ["f", "fc.w", "fc.b"], ["y"], transB=1)],
    "cnn", [vi("x", F, [1, 1, 8, 8])], [vi("y", F, [1, 2])],
    initializer=[weights("conv.w", (4, 1, 3, 3)), weights("conv.b", (4,)),
                 weights("bn.scale", (4,)), weights("bn.bias", (4,)), weights("bn.mean", (4,)),
                 numpy_helper.from_array(np.abs(rng.standard_normal(4)).astype(np.float32) + 0.5, "bn.var"),
                 weights("fc.w", (2, 4)), weights("fc.b", (2,))]), opset=18))

ok("optionals.onnx", model(helper.make_graph(
    [mk("Constant", [], ["two"], value=numpy_helper.from_array(np.array(2.0, np.float32))),
     mk("Mul", ["x", "two"], ["x2"]),
     mk("Dropout", ["x2"], ["d", ""]),
     mk("Reshape", ["d", "shape"], ["r"]),
     mk("Clip", ["r", "", "max"], ["y"])],
    "optionals", [vi("x", F, [2, 3])], [vi("y", F, [3, 2])],
    initializer=[numpy_helper.from_array(np.array([3, 2], np.int64), "shape"),
                 numpy_helper.from_array(np.array(1.0, np.float32), "max")])))

# initializers also listed as graph inputs, as models before IR v4 did
ok("initializers_as_inputs.onnx", model(helper.make_graph(
    [mk("MatMul", ["x", "W"], ["mm"], name="mm"), mk("Add", ["mm", "B"], ["y"], name="add")],
    "init_as_input",
    [vi("x", F, [1, 4]), vi("W", F, [4, 2]), vi("B", F, [2])], [vi("y", F, [1, 2])],
    initializer=[weights("W", (4, 2)), weights("B", (2,))]), opset=13, ir_version=7))

# --- models that must be rejected ---

error("opset11.onnx", model(helper.make_graph(
    [mk("Relu", ["x"], ["y"])], "old", [vi("x", F, [4])], [vi("y", F, [4])]), opset=11),
    "opset 11 is not supported")

error("unsupported_ops.onnx", model(helper.make_graph(
    [mk("Shape", ["x"], ["s"]),
     mk("LSTM", ["x", "w", "r"], ["h"], hidden_size=4),
     mk("LSTM", ["h", "w", "r"], ["y"], hidden_size=4)],
    "rnn", [vi("x", F, [1, 1, 4])], [vi("y", F, [1, 1, 1, 4]), vi("s", TensorProto.INT64, [3])],
    initializer=[weights("w", (1, 16, 4)), weights("r", (1, 16, 4))])),
    "unsupported operators: LSTM (2 nodes), Shape (1 node)")

error("float16_input.onnx", model(helper.make_graph(
    [mk("Relu", ["x"], ["y"])], "half",
    [vi("x", TensorProto.FLOAT16, [4])], [vi("y", TensorProto.FLOAT16, [4])])),
    "unsupported data type 10")

ext = weights("big", (16,))
set_external_data(ext, location="weights.bin")
ext.ClearField("raw_data")
error("external_data.onnx", model(helper.make_graph(
    [mk("Add", ["x", "big"], ["y"])], "ext", [vi("x", F, [16])], [vi("y", F, [16])],
    initializer=[ext])),
    "external data is not supported")

with open(os.path.join(HERE, "manifest.txt"), "w", newline="\n") as f:
    f.write(f"# generated by make_fixtures.py with onnx {onnx.__version__}, do not edit\n")
    f.write("\n".join(manifest) + "\n")

total = sum(os.path.getsize(os.path.join(HERE, p)) for p in os.listdir(HERE) if p.endswith(".onnx"))
print(f"wrote {sum(1 for l in manifest if l.startswith('model'))} fixtures ({total} bytes) and manifest.txt")
