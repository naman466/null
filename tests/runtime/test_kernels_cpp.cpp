// the runtime header must also work from c++ (constant folding will reuse
// the kernels), so the c test is compiled a second time as c++
#include "test_kernels.c"
