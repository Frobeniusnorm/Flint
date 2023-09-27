package flint

// #include <flint/flint.h>
import "C"
import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// convertArray converts Arrays between arbitrary types
// NOTE: does not work with nested arrays yet.
// See: https://stackoverflow.com/questions/71587996/cannot-use-type-assertion-on-type-parameter-value
// FIXME: func convertArray[In any, Out any](arr []In) []Out {
func convertArray[In completeNumbers, Out completeNumbers | cNumbers](arr []In) []Out {
	result := make([]Out, len(arr))
	for idx, val := range arr {
		result[idx] = Out(val)
		//x := any(val).(Out)
		//result[idx] = x
	}
	return result
}

func convertToCType[T numeric](value T, dataType DataType) any {
	switch dataType {
	case F_INT32:
		return C.int(value)
	case F_INT64:
		return C.long(value)
	case F_FLOAT32:
		return C.float(value)
	case F_FLOAT64:
		return C.double(value)
	default:
		panic("invalid data type")
	}
}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

/*
func arrayFromC[T numeric | uint64 | uint32 | int | uint](length int, dataPtr unsafe.Pointer, dataType DataType) []T {
	var sizeOf int
	switch dataType {
	case F_INT32:
		sizeOf = int(C.sizeof_int)
	case F_INT64:
		sizeOf = int(C.sizeof_long)
	case F_FLOAT32:
		sizeOf = int(C.sizeof_float)
	case F_FLOAT64:
		sizeOf = int(C.sizeof_double)
	default:
		panic("invalid type")
	}

	var result = make([]T, length)
	//voidPtr := (*C.void)(dataPtr)
	for i := 0; i < length; i++ {
		//x := *(voidPtr + C.int(i)) // add sizeof?
		//x := unsafe.Pointer(dataPtr + C.int(i))
		x := dataPtr + i*sizeOf
		result[i] = T(x)
	}
	return result
}
*/

func fromCToArray[T completeNumbers](dataPtr unsafe.Pointer, length int, dataType DataType) []T {
	var result = make([]T, length)

	switch dataType {
	case F_INT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_int))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint32(relevantData)
			result[i] = T(x)
		}
	case F_INT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_long))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint64(relevantData)
			result[i] = T(x)
		}

	case F_FLOAT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_float))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint32(relevantData)
			x := math.Float32frombits(bits)
			result[i] = T(x)
		}

	case F_FLOAT64:
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
