# Null: An Educational Deep Learning Compiler

**Null** is an educational deep learning compiler written in **C++20**.

The project focuses on understanding how neural network models can be **compiled ahead of time** into low-level code. Given a trained model, Null analyzes its computation graph, applies a small set of well-defined optimizations, and generates a **standalone C source file (`model.c`)** for inference.

This repository is primarily a **learning and documentation project**. It is intended to explore compiler design concepts as they apply to deep learning systems, rather than to serve as a production-ready framework.

---

## Project Goals

### 1. Static Memory Management

Null explores **static memory planning** for neural networks.

Instead of allocating and freeing memory dynamically during execution, the compiler:

* Analyzes tensor lifetimes at compile time
* Computes the total memory required for intermediate tensors
* Allocates a single contiguous memory buffer
* Reuses this buffer across operations based on liveness information

This design is intended to study memory behavior and lifetime analysis in compiled ML systems.

---

### 2. Minimal Runtime Requirements

The compiler generates plain C code that:

* Can be compiled with a standard C compiler
* Does not rely on external runtimes or language interpreters
* Has no dependency on the C++ standard library at inference time

This keeps the generated output simple and easy to inspect, which aligns with the educational focus of the project.

---

### 3. Educational Clarity

Null is written with readability and traceability in mind.

Key design choices:

* A small, explicit Intermediate Representation (IR)
* Clear separation between frontend, optimizer, and backend stages
* Straightforward implementations of standard compiler analyses

The goal is for students to be able to read the source code and understand how each part of the compiler works.

---

## Compilation Pipeline

A model passes through the following stages:

### 1. Frontend (Model Ingestion)

* Reads a `.onnx` model file
* Parses the Protocol Buffer format
* Converts ONNX operators into the **Null Intermediate Representation (IR)**

---

### 2. Optimizer (Middle-End)

Applies simple graph-level optimizations, such as:

* **Constant Folding**: Precomputing statically known values
* **Dead Code Elimination**: Removing unused operations
* **Operator Fusion**: Combining compatible operations to reduce intermediate tensors

These passes are intentionally kept small and explicit.

---

### 3. Backend (Code Generation)

* Performs **liveness analysis** to determine tensor lifetimes
* Assigns memory offsets within a single static buffer
* Emits C code specialized for the model’s tensor shapes and operations

---

### 4. Runtime Support

* A minimal C runtime provides basic utilities required by the generated code
* No dynamic memory allocation or external libraries are required

---

## Project Structure

```text
Null/
├── docs/               # Educational documentation ("book"-style notes)
├── src/
│   ├── core/           # Graph, Node, and Tensor IR definitions
│   ├── frontend/       # ONNX parsing and IR construction
│   ├── backend/        # Memory planning and C code generation
│   └── main.cpp        # Command-line interface
├── runtime/            # Minimal C runtime for generated models
├── tests/              # Unit tests
└── CMakeLists.txt      # Build configuration
```

---

## Documentation

This project is accompanied by a structured set of notes that document:

* Design decisions
* Background theory
* Implementation details
* Mistakes, trade-offs, and lessons learned

The documentation is written as a **learning resource**, not a formal textbook.

**Documentation site:**
[https://null-compiler.github.io/book/](https://null-compiler.github.io/book/) *(Coming soon)*

### Current Outline

* **Part I: Intermediate Representation**
  Graph structure, tensors, and scheduling

* **Part II: Frontend**
  ONNX parsing and model lowering

* **Part III: Optimization Passes**
  Graph traversal, folding, and fusion

* **Part IV: Backend**
  Liveness analysis, memory planning, and code emission

---

## Getting Started

### Prerequisites

* **Compiler:** GCC 10+ or Clang 12+ (C++20 required)
* **Build System:** CMake 3.15+
* **Tools:** Standard build utilities (`make`, headers, etc.)

### Build Instructions

```bash
git clone https://github.com/your-username/Null.git
cd Null

mkdir build && cd build
cmake ..
make

./null_cli
# Usage: nullc [model_path]
```

---

## Project Status

This project is under active development and experimentation.

### Planned Milestones

* [x] Project setup and build system
* [ ] Core graph IR
* [ ] Topological sorting
* [ ] ONNX frontend
* [ ] Basic C code generation
* [ ] End-to-end execution on a simple model
* [ ] Optimization passes
* [ ] Static memory planning

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.