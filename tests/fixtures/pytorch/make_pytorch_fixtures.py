"""Export real PyTorch models to ONNX and record what null must make of them.

These fixtures exist because real exporters produce graphs that hand-built
models don't: Shape/Gather/Concat chains, Identity and Constant clutter,
ReduceMean with axes as an input, BatchNorm already folded away. The .onnx
files and pytorch_manifest.txt are committed, so ctest needs no Python; this
script is only needed to regenerate them.

    pip install torch torchvision onnx onnxscript --index-url https://download.pytorch.org/whl/cpu
    PYTHONUTF8=1 python tests/fixtures/pytorch/make_pytorch_fixtures.py

(PYTHONUTF8=1 because the newer exporter prints emoji, which Windows'
default console encoding cannot print.)

pytorch_manifest.txt, per model:

    model <file> <exporter> torch <version> opset <n>
    input <name> <dims...>                  as exported, -1 for a dynamic dim
    ops <Op:count> ...                      the graph's operators, from onnx
    expect compiles                         nullc --batch 1 succeeds
    expect load-error <message substring>   load_model fails
    expect shape-error <message substring>  loads, but shape inference stops
    shape <tensor> <dims...>                for "compiles": every node output's
                                            shape from onnx's own shape
                                            inference, with the batch fixed to 1
    end

The expectations describe what null can do today. When null gains support for
an operator, the matching test fails on purpose, and the expectation here is
updated in the same commit.
"""
import collections, os, warnings

import onnx
import torch
import torch.nn as nn
from onnx import shape_inference

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


class MLP(nn.Module):
    """Flatten, Linear, ReLU, Linear, Softmax: everything null supports."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class MLPView(nn.Module):
    """the classic tutorial style: flatten with x.view(x.size(0), -1)"""
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(784, 32), nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=1)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, 6, 5, padding=2), nn.Conv2d(6, 16, 5)
        self.fc1, self.fc2, self.fc3 = nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 10)
    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class ConvBN(nn.Module):
    """BatchNorm and an adaptive average pool: both exporters fold the BN away"""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
    def forward(self, x):
        return self.f(x)


def no_rule(op):
    return f"({op}): no shape inference rule for this operator yet"

# (file, model factory, input shape, exporter, dynamic batch, expectation)
FIXTURES = [
    ("mlp__legacy_dyn.onnx",      MLP,     (1, 1, 28, 28), "legacy", True,  ("compiles", "")),
    ("mlp__dynamo_dyn.onnx",      MLP,     (1, 1, 28, 28), "dynamo", True,  ("compiles", "")),
    ("mlp_view__legacy.onnx",     MLPView, (1, 1, 28, 28), "legacy", False, ("compiles", "")),
    ("mlp_view__legacy_dyn.onnx", MLPView, (1, 1, 28, 28), "legacy", True,  ("load-error", "unsupported operators: Shape (1 node)")),
    ("lenet__legacy.onnx",        LeNet,   (1, 1, 28, 28), "legacy", False, ("shape-error", no_rule("MaxPool"))),
    ("lenet__dynamo.onnx",        LeNet,   (1, 1, 28, 28), "dynamo", False, ("shape-error", no_rule("MaxPool"))),
    ("conv_bn__legacy.onnx",      ConvBN,  (1, 3, 32, 32), "legacy", False, ("shape-error", no_rule("GlobalAveragePool"))),
    ("conv_bn__dynamo.onnx",      ConvBN,  (1, 3, 32, 32), "dynamo", False, ("shape-error", no_rule("ReduceMean"))),
    ("transformer__dynamo.onnx",  lambda: nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True),
                                           (1, 8, 32),     "dynamo", False, ("load-error", "unsupported operators: LayerNormalization (2 nodes)")),
]


def export(factory, shape, exporter, dynamic, path):
    torch.manual_seed(5010)
    model = factory().eval()
    x = torch.randn(shape)
    if exporter == "legacy":
        kw = dict(dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}) if dynamic else {}
        torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"], dynamo=False, **kw)
    else:
        kw = dict(dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}}) if dynamic else {}
        torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"], dynamo=True,
                          external_data=False, verbose=False, **kw)


def dims_of(vi):
    return [d.dim_value if d.WhichOneof("value") == "dim_value" else -1 for d in vi.type.tensor_type.shape.dim]


manifest = [f"# generated by make_pytorch_fixtures.py with torch {torch.__version__} and onnx {onnx.__version__}, do not edit"]
for fname, factory, shape, exporter, dynamic, (kind, text) in FIXTURES:
    path = os.path.join(HERE, fname)
    export(factory, shape, exporter, dynamic, path)
    m = onnx.load(path)
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    manifest.append(f"model {fname} {exporter} torch {torch.__version__.split('+')[0]} opset {opset}")
    for vi in m.graph.input:
        manifest.append(" ".join(["input", vi.name] + [str(d) for d in dims_of(vi)]))
    ops = collections.Counter(n.op_type for n in m.graph.node)
    manifest.append("ops " + " ".join(f"{k}:{v}" for k, v in sorted(ops.items())))
    manifest.append(f"expect {kind} {text}".rstrip())
    if kind == "compiles":
        # onnx's own shapes for every node output, with the batch fixed to 1 like --batch 1
        for vi in m.graph.input:
            d = vi.type.tensor_type.shape.dim[0]
            if d.WhichOneof("value") != "dim_value":
                d.ClearField("dim_param")
                d.dim_value = 1
        inferred = shape_inference.infer_shapes(m, strict_mode=True)
        known = {vi.name: dims_of(vi) for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
        for node in m.graph.node:
            for out in node.output:
                if out in known:
                    manifest.append(" ".join(["shape", out] + [str(d) for d in known[out]]))
    manifest.append("end")
    print(f"{fname:28s} {os.path.getsize(path):>8,d} bytes  expect {kind}")

with open(os.path.join(HERE, "pytorch_manifest.txt"), "w", newline="\n") as f:
    f.write("\n".join(manifest) + "\n")
total = sum(os.path.getsize(os.path.join(HERE, f)) for f, *_ in FIXTURES)
print(f"wrote {len(FIXTURES)} fixtures ({total:,d} bytes) and pytorch_manifest.txt")
