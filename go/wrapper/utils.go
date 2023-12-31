package wrapper

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

func fromCToArray[T completeNumeric](dataPtr unsafe.Pointer, length int, dataType DataType) []T {
	var result = make([]T, length)

	switch dataType {
	case INT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_int))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint32(relevantData)
			result[i] = T(x)
		}
	case INT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_long))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint64(relevantData)
			result[i] = T(x)
		}

	case FLOAT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_float))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint32(relevantData)
			x := math.Float32frombits(bits)
			result[i] = T(x)
		}

	case FLOAT64:
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
func closestType[T completeNumeric](x T) DataType {
	switch any(x).(type) {
	case int:
		return INT64
	case int8:
		return INT32
	case int16:
		return INT32
	case int32:
		return INT32
	case int64:
		return INT64
	case uint:
		return INT64
	case uint8:
		return INT32
	case uint16:
		return INT32
	case uint32:
		return INT64
	case float32:
		return FLOAT32
	case float64:
		return FLOAT64
	default:
		panic("invalid type")
	}
}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

func returnHelper(flintNode *C.FGraphNode, errno error) (GraphNode, error) {
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}
