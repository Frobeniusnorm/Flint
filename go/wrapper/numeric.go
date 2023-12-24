package wrapper

// #include <flint/flint.h>
import "C"

// baseNumeric is a constraint interface representing the supported types of C operations
type baseNumeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// completeNumeric is a constraint interface representing all numeric types that can be used in wrapper!
// NOTE: cant support uint64 as casting it to int64 might cause overflows
type completeNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~float32 | ~float64
}

// cNumbers represents the basic C types available in CGo
// NOTE: size_t should be equivalent to ulong
type cNumbers interface {
	C.int | C.size_t | C.long | C.uint | C.float | C.double
}
