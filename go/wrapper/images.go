package wrapper

// #include <flint/flint.h>
import "C"
import (
	"unsafe"
)

type ImageFormat int

const (
	PNG ImageFormat = iota
	JPEG
	BMP
)

/*
LoadImage loads an image from the given path.

The image will be stored in floating point data and the shape will be h, w, c
where w is the width, h is the height and c are the channels.

The supported formats include the ones listed in [ImageFormat].

If no image can be found at the [path], LoadImage returns an error wrapping ErrIO
*/
func LoadImage(path string) (GraphNode, error) {
	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))

	flintNode, errno := C.fload_image(unsafePath)
	return returnHelper(flintNode, errno)
}

/*
StoreImage saves the result of node in a given ImageFormat.
The node has to be of type [FLOAT32] or [FLOAT64] datatype and needs to have 3 dimensions.

Errors:
- error wrapping ErrIllegalDimensionality if node has a dimensionality other than 3.
- error wrapping ErrWrongType is thrown if node is not of type [FLOAT32] or [FLOAT64].
- error wrapping ErrOOM if out of memory
- error wrapping ErrIO if path is invalid or not writable
*/
func StoreImage(node GraphNode, path string, format ImageFormat) error {
	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))
	errCode, errno := C.fstore_image(node.ref, unsafePath, C.enum_FImageFormat(format))
	return buildErrorFromErrnoAndErrorType(errno, errorType(errCode))
}
