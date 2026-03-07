#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace null {

// supported data types 
enum class DType : int32_t {
    UNDEFINED = 0,
    FLOAT32   = 1,
    UINT8     = 2,
    INT8      = 3,
    UINT16    = 4,
    INT16     = 5,
    INT32     = 6,
    INT64     = 7,
    BOOL      = 9,
    FLOAT64   = 11,
};

} // namespace null
