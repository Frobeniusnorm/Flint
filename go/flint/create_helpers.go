package flint

import (
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"reflect"
	"unsafe"
)

func unsafeConversion[T Numeric](input []any) []T {
	// Attempting the unsafe conversion
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&input))
	header.Len *= int(unsafe.Sizeof(input[0])) / int(unsafe.Sizeof(T(0)))
	header.Cap *= int(unsafe.Sizeof(input[0])) / int(unsafe.Sizeof(T(0)))
	slice := *(*[]T)(unsafe.Pointer(&header))
	return slice
}

func FromNode(node wrapper.GraphNode) Tensor {
	res := Tensor{node: node}
	res.init()
	return res
}

func FromNodeWithErr(node wrapper.GraphNode, err error) Tensor {
	res := FromNode(node)
	res.err = err
	return res
}

func flatten[T Numeric](input any) []T {
	var result []T
	val := reflect.ValueOf(input)
	flattenHelper(val, &result)
	return result
}

func flattenHelper[T Numeric](value reflect.Value, result *[]T) {
	// Recursively flatten the slice
	if value.Kind() == reflect.Slice {
		for i := 0; i < value.Len(); i++ {
			flattenHelper(value.Index(i), result)
		}
		return
	}

	// Ensure the value can be converted to type T and append it
	// This is where the actual type assertion happens
	val := value.Interface()
	if converted, ok := val.(T); ok {
		*result = append(*result, converted)
	}
}
