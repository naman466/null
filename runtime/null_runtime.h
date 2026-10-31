/*
 * null runtime: every kernel that generated model.c files can call
 *
 * the runtime lives in parts under runtime/null/, core first (see
 * null/core.h for what it guarantees). generated files embed only the parts
 * they use; code that wants all of it, like the compiler and the runtime's
 * own tests, includes this header. the parts include nothing from each other,
 * so that they can be pasted into a generated file as they are: this order
 * is their dependency order.
 */
#ifndef NULL_RUNTIME_H
#define NULL_RUNTIME_H

#include "null/core.h"
#include "null/elementwise.h"
#include "null/matmul.h"
#include "null/softmax.h"
#include "null/conv.h"
#include "null/pool.h"
#include "null/reduce.h"

#endif /* NULL_RUNTIME_H */
