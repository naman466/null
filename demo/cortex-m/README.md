# LeNet on a Cortex-M

This runs the LeNet-5 that the end-to-end tests use, compiled by `nullc`, on an emulated **Cortex-M4** microcontroller: QEMU's
`mps2-an386` board, with no operating system, no C library and no allocator. It is a `ctest` test, `demo_cortex_m_lenet`, whenever
an Arm cross compiler and QEMU are installed, and CI runs it on Linux.

The program runs the model on the eight recorded inputs of `tests/fixtures/e2e/pytorch__lenet__legacy.vectors`, checks every output
against ONNX Runtime's like the end-to-end tests do (`|got - ref| <= 1e-6 + 1e-5 |ref|`, same predicted class), and writes the
outputs back to the host. The test then builds the same generated C for the host and requires the Cortex-M's outputs to be
**bit for bit** the host's. They are: the kernels' fixed summation order (built with `-ffp-contract=off`, so neither machine fuses
multiply-adds) gives the same floats on a 32-bit Arm core with a single-precision FPU as on an x86-64 desktop.

```text
null demo: LeNet-5 on a Cortex-M4 (qemu mps2-an386), no OS, no malloc
  input 0: class 7, onnx runtime says 7
  ...
8 inputs: 0 values outside tolerance, 0 class mismatches, largest difference from onnx runtime 6.4e-06
-- the Cortex-M's outputs are bit for bit the host's
```

| | Bytes |
|---|---:|
| code and constants (`text`): the weights (241 KB), the kernels and the recorded vectors (25 KB) | 283,192 |
| RAM (`bss`): the model's arena (23 KB) and the demo's input and output buffers (48 KB) | 72,704 |

## The pieces

- [`startup.c`](startup.c): the vector table, a reset handler that turns on the FPU and zeroes `.bss`, and the Arm semihosting
  calls (`bkpt 0xAB`) through which the program prints, writes its outputs to a host file and exits with a status.
- [`main.c`](main.c): reads the embedded vectors, runs `model_run_v`, compares, reports.
- [`vectors.S`](vectors.S): embeds the `.vectors` file with `.incbin`.
- [`link.ld`](link.ld): code and weights in the board's 4 MB at address 0, data, arena and stack in the 4 MB at `0x20000000`.
- [`run_demo.cmake`](run_demo.cmake): compiles the model, builds the image, runs QEMU, builds the host version, compares.

The image links with `-nostdlib` and only `libgcc`, the compiler's support library, for 64-bit integer division. Nothing in the
generated `model.c` calls the C library.

## Running it

Install `arm-none-eabi-gcc` (with newlib's headers, for `<math.h>`) and `qemu-system-arm`: on Ubuntu,
`apt install gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm`; on Windows, `scoop install gcc-arm-none-eabi qemu`. Configure
again so CMake finds them, then:

```bash
ctest --test-dir build -R demo_cortex_m --output-on-failure -V
```

or by hand, with any model of one input and one output and its vectors file:

```bash
cmake -DNULLC=build/nullc -DMODEL=tests/fixtures/pytorch/lenet__legacy.onnx \
      -DVECTORS=tests/fixtures/e2e/pytorch__lenet__legacy.vectors \
      -DARM_GCC=arm-none-eabi-gcc -DQEMU=qemu-system-arm -DCC=gcc \
      -DDRIVER=tests/e2e/run_vectors.c -DOUT=build/demo -P demo/cortex-m/run_demo.cmake
```
