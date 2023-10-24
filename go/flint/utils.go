package flint

// #include <flint/flint.h>
import "C"
import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// See: https://stackoverflow.com/questions/71587996/cannot-use-type-assertion-on-type-parameter-value
//func convertArrayOld[In completeNumeric, Out cNumbers](arr []In) []Out {
//	result := make([]Out, len(arr))
//	for idx, val := range arr {
//		result[idx] = Out(val) //nolint:all
//		//x := any(val).(Out)
//		//result[idx] = x
//	}
//	return result
//}

// convertArray converts Arrays between arbitrary types
// NOTE: does not work with nested arrays yet.
func convertArray[In completeNumeric, Out cNumbers](arr []In) []Out {
	result := make([]Out, len(arr))
	switch any(result[0]).(type) {
	case C.int:
		for idx, val := range arr {
			result[idx] = any(C.int(val)).(Out)
		}
	case C.size_t:
		for idx, val := range arr {
			result[idx] = any(C.size_t(val)).(Out)
		}
	case C.long:
		for idx, val := range arr {
			result[idx] = any(C.long(val)).(Out)
		}
	case C.uint:
		for idx, val := range arr {
			result[idx] = any(C.uint(val)).(Out)
		}
	case C.float:
		for idx, val := range arr {
			result[idx] = any(C.float(val)).(Out)
		}
	case C.double:
		for idx, val := range arr {
			result[idx] = any(C.double(val)).(Out)
		}
	}
	return result
}

//func convertToCNumber[In completeNumeric, Out cNumbers](n In) Out {
//	switch v := any(n).(type) {
//	case int, int8, int16, int32:
//		return C.int(v)
//	case int64:
//		return C.long(v)
//	case uint, uint8, uint16, uint32:
//		return C.uint(v)
//	case float32:
//		return C.float(v)
//	case float64:
//		return C.double(v)
//	default:
//		// Handle this case appropriately; maybe panic or return a zero value
//		panic("unsupported type")
//	}
//}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

func fromCToArray[T completeNumeric](dataPtr unsafe.Pointer, length int, dataType dataType) []T {
	var result = make([]T, length)

	switch dataType {
	case f_INT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_int))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint32(relevantData)
			result[i] = T(x)
		}
	case f_INT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_long))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint64(relevantData)
			result[i] = T(x)
		}

	case f_FLOAT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_float))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint32(relevantData)
			x := math.Float32frombits(bits)
			result[i] = T(x)
		}

	case f_FLOAT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_double))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint64(relevantData)
			x := math.Float64frombits(bits)
			result[i] = T(x)
		}
	default:
		panic("invalid type")
	}

	return result
}

// closestType takes in a Go type and returns a valid [dataType] that most closely matches it.
func closestType[T completeNumeric](x T) dataType {
	switch any(x).(type) {
	case int:
		return f_INT64
	case int8:
		return f_INT32
	case int16:
		return f_INT32
	case int32:
		return f_INT32
	case int64:
		return f_INT64
	case uint:
		return f_INT64
	case uint8:
		return f_INT32
	case uint16:
		return f_INT32
	case uint32:
		return f_INT64
	case float32:
		return f_FLOAT32
	case float64:
		return f_FLOAT64
	default:
		panic("invalid type")
	}
}
