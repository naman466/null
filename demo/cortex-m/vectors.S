/* the recorded inputs and onnx runtime's outputs (a tests/fixtures/e2e
   .vectors file), embedded as they are; VECTORS_FILE is set by the build */
    .section .rodata
    .balign 8
    .global null_vectors
null_vectors:
    .incbin VECTORS_FILE
    .global null_vectors_end
null_vectors_end:
