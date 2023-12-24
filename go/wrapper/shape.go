package wrapper

// #include <flint/flint.h>
import "C"
import "fmt"

// Shape represents the size of a flint.
// needs to have one entry for each dimension of flint
type Shape []uint

func (a Shape) String() string {
	return fmt.Sprintf("%v", []uint(a))
}

func (a Shape) Equal(b Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// Stride defines the steps for sliding operations
// needs to have one entry for each dimension of flint
type Stride []int

func (a Stride) String() string {
	return fmt.Sprintf("%v", []int(a))
}

// Axes indicate changes in dimensions (i.e. transpose)
// needs to have one entry for each dimension of flint
// and each entry should not be higher than the number of dimensions
type Axes []int

func (a Axes) String() string {
	return fmt.Sprintf("%v", []int(a))
}
