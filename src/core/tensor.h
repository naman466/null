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


inline std::string dtype_to_str(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return "float";
        case DType::FLOAT64: return "double";
        case DType::INT32:   return "int32_t";
        case DType::INT64:   return "int64_t";
        case DType::INT8:    return "int8_t";
        case DType::UINT8:   return "uint8_t";
        case DType::INT16:   return "int16_t";
        case DType::UINT16:  return "uint16_t";
        case DType::BOOL:    return "uint8_t";
        default:             return "void";
    }
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::INT32:   return 4;
        case DType::INT64:   return 8;
        case DType::INT8:    return 1;
        case DType::UINT8:   return 1;
        case DType::INT16:   return 2;
        case DType::UINT16:  return 2;
        case DType::BOOL:    return 1;
        default:             return 0;
    }
} // namespace null
