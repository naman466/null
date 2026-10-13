// mutation fuzzer for the onnx frontend
//
// usage: fuzz_frontend [iterations] [rng_seed]
//
// starts from well-formed messages built with PbWriter, mutates them at
// random and feeds every result to each parser. a finding is anything other
// than a clean rejection or a consistent result:
//   - an exception other than OnnxError / ParseError
//   - a crash or sanitizer trap (input saved to fuzz_crash.bin)
//   - a tensor whose data size disagrees with its shape, or a bool not 0/1
//   - a malformed ValueInfo
// build with -DNULL_SANITIZE=ON so undefined behaviour traps as well
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "frontend/onnx.h"
#include "pb_writer.h"

using namespace null;
using null_test::PbWriter;
using Bytes = std::string;

// crash reporting

static const Bytes* g_current = nullptr;
static const char* g_stage = "";

extern "C" void on_signal(int sig) {
    if (g_current) {
        if (FILE* f = std::fopen("fuzz_crash.bin", "wb")) {
            std::fwrite(g_current->data(), 1, g_current->size(), f);
            std::fclose(f);
        }
    }
    std::fprintf(stderr, "\n*** signal %d in %s, input saved to fuzz_crash.bin\n", sig, g_stage);
    std::_Exit(2);
}

// findings, the first example of each kind is saved

static std::map<std::string, long> g_findings;

static void finding(const std::string& kind, const Bytes& input) {
    if (g_findings[kind]++ == 0) {
        std::string file = "fuzz_finding_" + std::to_string(g_findings.size()) + ".bin";
        std::ofstream(file, std::ios::binary).write(input.data(), input.size());
        std::cout << "  finding: " << kind << " (saved " << file << ")\n";
    }
}

static void check_tensor(const Tensor& t, const Bytes& input) {
    for (int64_t d : t.shape)
        if (d < 0) finding("tensor accepted with a negative dim", input);
    try {
        if (static_cast<uint64_t>(t.byte_size()) != t.data.size())
            finding("tensor data size does not match its shape", input);
    } catch (const std::overflow_error&) {
        finding("accepted tensor whose size overflows", input);
    }
    if (t.dtype == DType::BOOL)
        for (uint8_t b : t.data)
            if (b > 1) { finding("bool tensor holds a value other than 0/1", input); break; }
}

static void check_value_info(const onnx::ValueInfo& vi, const Bytes& input) {
    if (vi.shape.size() != vi.dim_params.size())
        finding("value info shape and dim_params lengths differ", input);
    if (!vi.has_shape && !vi.shape.empty())
        finding("value info with unknown rank has dims", input);
    for (size_t i = 0; i < vi.shape.size(); ++i) {
        if (vi.shape[i] < -1) finding("value info dim below -1", input);
        if (!vi.dim_params[i].empty() && vi.shape[i] != -1)
            finding("symbolic value info dim is not -1", input);
    }
    // huge declared shapes are legal metadata, only a clean throw is allowed
    try { vi.to_tensor().num_elements(); } catch (const std::overflow_error&) {}
}

template<typename F>
static void run(const char* stage, const Bytes& input, F fn) {
    g_stage = stage;
    try {
        fn();
    } catch (const onnx::OnnxError&) {
    } catch (const pb::ParseError&) {
    } catch (const std::exception& e) {
        finding(std::string(stage) + " threw " + typeid(e).name() + ": " + e.what(), input);
    } catch (...) {
        finding(std::string(stage) + " threw a non-std exception", input);
    }
}

// ModelProto.graph = 7, GraphProto node = 1, initializer = 5,
// input / output / value_info = 11 / 12 / 13
static void walk_model(const Bytes& input) {
    pb::Reader m(input);
    while (!m.eof()) {
        pb::Tag t = m.read_tag();
        if (t.field != 7) { m.skip(t.wire); continue; }
        pb::Reader g = m.read_message(t);
        while (!g.eof()) {
            pb::Tag gt = g.read_tag();
            switch (gt.field) {
                case 1:  onnx::parse_node(g.read_message(gt)); break;
                case 5:  check_tensor(onnx::parse_tensor(g.read_message(gt)), input); break;
                case 11: case 12: case 13:
                    check_value_info(onnx::parse_value_info(g.read_message(gt)), input); break;
                default: g.skip(gt.wire); break;
            }
        }
    }
}

static void exercise(const Bytes& input) {
    g_current = &input;
    run("parse_tensor", input, [&] {
        check_tensor(onnx::parse_tensor(pb::Reader(input)), input);
    });
    run("parse_attribute", input, [&] {
        Attribute a = onnx::parse_attribute(pb::Reader(input));
        if (a.has_tensor()) check_tensor(a.as_tensor(), input);
    });
    run("parse_node", input, [&] {
        Node n = onnx::parse_node(pb::Reader(input));
        for (auto& [_, a] : n.attrs)
            if (a.has_tensor()) check_tensor(a.as_tensor(), input);
    });
    run("parse_value_info", input, [&] {
        check_value_info(onnx::parse_value_info(pb::Reader(input)), input);
    });
    run("walk_model", input, [&] { walk_model(input); });
    g_current = nullptr;
}

// seed corpus, well-formed messages covering every parser path

static Bytes bytes_of(const PbWriter& w) { return Bytes(w.buf.begin(), w.buf.end()); }

template<typename T>
static Bytes raw(const std::vector<T>& v) {
    Bytes s(v.size() * sizeof(T), '\0');
    std::memcpy(s.data(), v.data(), s.size());
    return s;
}

static PbWriter tensor(const std::string& name, int64_t dtype, const std::vector<int64_t>& dims) {
    PbWriter w;
    for (int64_t d : dims) w.int_field(1, d);
    w.int_field(2, dtype).bytes_field(8, name);
    return w;
}

static PbWriter attr(const std::string& name, int64_t type) {
    PbWriter w;
    w.bytes_field(1, name).int_field(20, type);
    return w;
}

static PbWriter value_info(const std::string& name, int64_t dtype,
                           const std::vector<int64_t>& dims, const std::vector<std::string>& params) {
    PbWriter shape;
    for (size_t i = 0; i < dims.size(); ++i) {
        PbWriter d;
        if (!params[i].empty()) d.bytes_field(2, params[i]);
        else d.int_field(1, dims[i]);
        shape.message_field(1, d);
    }
    PbWriter tt; tt.int_field(1, dtype).message_field(2, shape);
    PbWriter type; type.message_field(1, tt);
    PbWriter vi; vi.bytes_field(1, name).message_field(2, type);
    return vi;
}

static PbWriter node(const std::string& name, const std::string& op,
                     const std::vector<std::string>& in, const std::vector<std::string>& out) {
    PbWriter w;
    for (auto& s : in) w.bytes_field(1, s);
    for (auto& s : out) w.bytes_field(2, s);
    w.bytes_field(3, name).bytes_field(4, op);
    return w;
}

static std::vector<Bytes> make_corpus() {
    std::vector<Bytes> c;

    // tensors, raw and typed storage for each supported type
    PbWriter t;
    t = tensor("w", 1, {2, 3});    t.bytes_field(9, raw(std::vector<float>{1, 2, 3, 4, 5, 6})); c.push_back(bytes_of(t));
    t = tensor("d", 11, {2});      t.bytes_field(9, raw(std::vector<double>{0.5, -1})); c.push_back(bytes_of(t));
    t = tensor("i64", 7, {3});     t.bytes_field(9, raw(std::vector<int64_t>{1, -1, 1LL << 40})); c.push_back(bytes_of(t));
    t = tensor("b", 9, {3});       t.bytes_field(9, Bytes("\x00\x01\x01", 3)); c.push_back(bytes_of(t));
    t = tensor("u8", 2, {4});      t.bytes_field(9, Bytes("\x00\x7f\x80\xff", 4)); c.push_back(bytes_of(t));
    t = tensor("f", 1, {3});       t.packed_floats(4, {0.5f, -1.5f, 2}); c.push_back(bytes_of(t));
    t = tensor("f1", 1, {2});      t.float_field(4, 1).float_field(4, 2); c.push_back(bytes_of(t));
    t = tensor("i32", 6, {2});     t.packed_ints(5, {7, -7}); c.push_back(bytes_of(t));
    t = tensor("i8", 3, {3});      t.packed_ints(5, {-128, 0, 127}); c.push_back(bytes_of(t));
    t = tensor("i16", 5, {2});     t.packed_ints(5, {-300, 300}); c.push_back(bytes_of(t));
    t = tensor("bb", 9, {2});      t.packed_ints(5, {0, 1}); c.push_back(bytes_of(t));
    t = tensor("l", 7, {2});       t.packed_ints(7, {0, -1}); c.push_back(bytes_of(t));
    t = tensor("dd", 11, {1});     t.packed_doubles(10, {0.25}); c.push_back(bytes_of(t));
    t = tensor("s", 1, {});        t.packed_floats(4, {1e-5f}); c.push_back(bytes_of(t));
    t = tensor("e", 1, {0, 4});    c.push_back(bytes_of(t));

    // attributes of every supported type
    PbWriter a;
    a = attr("alpha", 1);    a.float_field(2, 0.01f); c.push_back(bytes_of(a));
    a = attr("axis", 2);     a.int_field(3, -1); c.push_back(bytes_of(a));
    a = attr("mode", 3);     a.bytes_field(4, "constant"); c.push_back(bytes_of(a));
    a = attr("scales", 6);   a.float_field(7, 1).float_field(7, 2); c.push_back(bytes_of(a));
    a = attr("pads", 7);     a.int_field(8, 1).int_field(8, 1).int_field(8, 2).int_field(8, 2); c.push_back(bytes_of(a));
    a = attr("perm", 7);     a.packed_ints(8, {0, 2, 1}); c.push_back(bytes_of(a));
    a = attr("empty", 7);    c.push_back(bytes_of(a));
    PbWriter at = tensor("", 7, {2}); at.packed_ints(7, {1, -1});
    a = attr("value", 4);    a.message_field(5, at); c.push_back(bytes_of(a));

    // nodes with attributes, optional inputs and several outputs
    PbWriter conv = node("/conv/Conv", "Conv", {"x", "w", "b"}, {"y"});
    PbWriter ks = attr("kernel_shape", 7); ks.int_field(8, 3).int_field(8, 3);
    PbWriter grp = attr("group", 2); grp.int_field(3, 1);
    conv.message_field(5, ks).message_field(5, grp);
    c.push_back(bytes_of(conv));
    c.push_back(bytes_of(node("clip", "Clip", {"x", "", "max"}, {"y"})));
    c.push_back(bytes_of(node("split", "Split", {"x"}, {"a", "b", "c"})));
    PbWriter cst = node("c", "Constant", {}, {"shape"});
    a = attr("value", 4); a.message_field(5, at);
    cst.message_field(5, a);
    c.push_back(bytes_of(cst));

    // value infos, fixed, symbolic, scalar and unknown rank
    PbWriter vx = value_info("x", 1, {-1, 1, 28, 28}, {"N", "", "", ""});
    c.push_back(bytes_of(vx));
    c.push_back(bytes_of(value_info("s", 1, {}, {})));
    PbWriter tt; tt.int_field(1, 7);
    PbWriter type; type.message_field(1, tt);
    PbWriter vu; vu.bytes_field(1, "u").message_field(2, type);
    c.push_back(bytes_of(vu));

    // a whole model: Reshape -> Gemm -> Relu with initializers and value infos
    PbWriter g;
    g.message_field(1, node("/reshape", "Reshape", {"x", "shape"}, {"xf"}));
    PbWriter gemm = node("/fc/Gemm", "Gemm", {"xf", "W", "B"}, {"h"});
    PbWriter tb = attr("transB", 2); tb.int_field(3, 1);
    gemm.message_field(5, tb);
    g.message_field(1, gemm);
    g.message_field(1, node("/relu", "Relu", {"h"}, {"y"}));
    g.bytes_field(2, "mlp");
    PbWriter W = tensor("W", 1, {4, 784}); W.bytes_field(9, Bytes(4 * 784 * 4, '\x01'));
    PbWriter B = tensor("B", 1, {4});      B.packed_floats(4, {0, 0, 0, 0});
    PbWriter S = tensor("shape", 7, {2});  S.packed_ints(7, {1, 784});
    g.message_field(5, W).message_field(5, B).message_field(5, S);
    g.message_field(11, vx);
    g.message_field(12, value_info("y", 1, {1, 4}, {"", ""}));
    g.message_field(13, value_info("h", 1, {1, 4}, {"", ""}));
    PbWriter opset; opset.bytes_field(1, "").int_field(2, 17);
    PbWriter model;
    model.int_field(1, 10).message_field(8, opset).message_field(7, g);
    c.push_back(bytes_of(model));

    return c;
}

// mutations

static const uint64_t kInteresting[] = {
    0, 1, 127, 128, 255, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1ULL << 32, 1ULL << 62,
    (1ULL << 63) - 1, 1ULL << 63, ~0ULL, (1ULL << 29) - 1, 1ULL << 29,
};

static Bytes varint(uint64_t v) {
    Bytes s;
    while (v >= 0x80) { s.push_back(static_cast<char>(v | 0x80)); v >>= 7; }
    s.push_back(static_cast<char>(v));
    return s;
}

// one top-level protobuf field: its key bytes and its payload bytes
struct Field {
    pb::Tag tag;
    Bytes key;
    Bytes payload;
};

// split a message into its top-level fields, false if it doesn't decode
static bool split_fields(const Bytes& b, std::vector<Field>& out) {
    try {
        pb::Reader r(b);
        while (!r.eof()) {
            size_t start = r.pos();
            pb::Tag t = r.read_tag();
            size_t mid = r.pos();
            r.skip(t.wire);
            out.push_back({t, b.substr(start, mid - start), b.substr(mid, r.pos() - mid)});
        }
        return true;
    } catch (const pb::ParseError&) {
        return false;
    }
}

// structure-aware mutation: keeps the message well-formed at the top level
// but changes which fields exist and their values, reaching semantic bugs
// (like huge dims next to an empty payload) that byte flips rarely hit
static Bytes mutate_fields(const Bytes& b, std::mt19937_64& rng) {
    std::vector<Field> fields;
    if (!split_fields(b, fields) || fields.empty()) return b;
    auto pick = [&](size_t n) { return static_cast<size_t>(rng() % n); };
    size_t i = pick(fields.size());
    switch (rng() % 5) {
        case 0: fields.erase(fields.begin() + i); break;                     // drop
        case 1: fields.insert(fields.begin() + pick(fields.size()), fields[i]); break;  // duplicate
        case 2: std::swap(fields[i], fields[pick(fields.size())]); break;    // reorder
        case 3:                                                              // extreme varint value
            if (fields[i].tag.wire == pb::WireType::VARINT)
                fields[i].payload = varint(kInteresting[pick(std::size(kInteresting))]);
            break;
        case 4:                                                              // empty a LEN payload
            if (fields[i].tag.wire == pb::WireType::LEN) fields[i].payload = varint(0);
            break;
    }
    Bytes out;
    for (auto& f : fields) out += f.key + f.payload;
    return out;
}

static Bytes mutate(const Bytes& seed, const std::vector<Bytes>& corpus, std::mt19937_64& rng) {
    auto pick = [&](size_t n) { return static_cast<size_t>(rng() % n); };
    Bytes b = seed;
    int rounds = 1 + static_cast<int>(rng() % 4);
    for (int r = 0; r < rounds; ++r) {
        size_t n = b.size();
        switch (rng() % 12) {
            case 9: case 10: case 11:
                b = mutate_fields(b, rng);
                break;
            case 0:   // flip one bit
                if (n) b[pick(n)] ^= static_cast<char>(1 << (rng() % 8));
                break;
            case 1: { // boundary byte
                static const char v[] = {0, 1, 0x7F, static_cast<char>(0x80), static_cast<char>(0xFF)};
                if (n) b[pick(n)] = v[pick(5)];
                break;
            }
            case 2:   // delete a few bytes
                if (n) { size_t p = pick(n); b.erase(p, 1 + pick(std::min<size_t>(n - p, 8))); }
                break;
            case 3:   // insert random bytes
                b.insert(n ? pick(n) : 0, 1 + pick(4), static_cast<char>(rng()));
                break;
            case 4:   // duplicate a chunk
                if (n > 1) { size_t a = pick(n), l = 1 + pick(std::min<size_t>(n - a, 32));
                             b.insert(pick(n), b.substr(a, l)); }
                break;
            case 5: { // splice with another seed
                const Bytes& o = corpus[pick(corpus.size())];
                if (n && !o.empty()) { size_t a = pick(o.size());
                    b = b.substr(0, pick(n)) + o.substr(a, pick(o.size() - a + 1)); }
                break;
            }
            case 6:   // insert an interesting varint
                b.insert(n ? pick(n) : 0, varint(kInteresting[pick(std::size(kInteresting))]));
                break;
            case 7:   // truncate
                if (n) b.resize(pick(n));
                break;
            case 8: { // overwrite with an interesting varint
                if (!n) break;
                size_t p = pick(n);
                Bytes v = varint(kInteresting[pick(std::size(kInteresting))]);
                b.replace(p, std::min(v.size(), n - p), v);
                break;
            }
        }
    }
    return b;
}

int main(int argc, char** argv) {
    long iterations = argc > 1 ? std::atol(argv[1]) : 100000;
    uint64_t rng_seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;

    std::signal(SIGILL, on_signal);
    std::signal(SIGSEGV, on_signal);
    std::signal(SIGABRT, on_signal);
    std::signal(SIGFPE, on_signal);

    std::vector<Bytes> corpus = make_corpus();

    // every seed must be accepted by at least one parser, or the corpus is broken
    for (const Bytes& s : corpus) {
        int accepted = 0;
        auto ok = [&](auto fn) { try { fn(); ++accepted; } catch (...) {} };
        ok([&] { onnx::parse_tensor(pb::Reader(s)); });
        ok([&] { onnx::parse_attribute(pb::Reader(s)); });
        ok([&] { onnx::parse_node(pb::Reader(s)); });
        ok([&] { onnx::parse_value_info(pb::Reader(s)); });
        ok([&] { walk_model(s); });
        if (!accepted) {
            std::cout << "corpus seed rejected by every parser\n";
            return 1;
        }
        exercise(s);
    }

    std::cout << "fuzzing " << iterations << " inputs from " << corpus.size()
              << " seeds (rng seed " << rng_seed << ")\n";

    std::mt19937_64 rng(rng_seed);
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iterations; ++i) {
        Bytes input = mutate(corpus[rng() % corpus.size()], corpus, rng);
        exercise(input);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::cout << iterations << " inputs in " << secs << " s ("
              << static_cast<long>(iterations / (secs > 0 ? secs : 1)) << " inputs/s)\n";
    if (g_findings.empty()) {
        std::cout << "no findings\n";
        return 0;
    }
    std::cout << "findings:\n";
    for (auto& [kind, count] : g_findings) std::cout << "  " << count << "x  " << kind << "\n";
    return 1;
}
