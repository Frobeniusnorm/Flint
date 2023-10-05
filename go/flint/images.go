package flint

// #include <flint/flint.h>
import "C"
import "unsafe"

type imageFormat int

const (
	PNG imageFormat = iota
	JPEG
	BMP
)

/*
LoadImage loads an image from the given path.
The image will be stored in floating point data and the shape will be h, w, c
where w is the width, h is the height and c are the channels.
Supported formats include png, jpeg, bmp, gif, hdr ... essentially everything stb_image supports.

NOTE for CGo: this function exemplary shows how to properly deal with errno
*/
func LoadImage(path string) (GraphNode, error) {
	resetErrno()

	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))

	var flintNode *C.FGraphNode
	var err error // has underlying syscall.Errno type
	flintNode, err = C.fload_image(unsafePath)
	if err != nil {
		return GraphNode{}, err
	}
	return GraphNode{ref: flintNode}, nil
}

/*
StoreImage saves the result of [node] in a given [imageFormat].
[node] has to be of type [f_FLOAT32] or [f_FLOAT64] datatype and needs to have 3 dimensions.
*/
func StoreImage(node GraphNode, path string, format imageFormat) {
	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))
	C.fstore_image(node.ref, unsafePath, C.enum_FImageFormat(format))
}
