package main

// #cgo CFLAGS: -g
// #include <stdlib.h>
// #include "../flint.h"

/*
int greet(const char *name, int year, char *out);
*/
import "C"
import (
	"unsafe"
)

func main() {
	//test := C.CString("Test")
	//defer C.free(unsafe.Pointer(test))
	//
	//cc := new(C.enum_FLogType)
	//fmt.Println(*cc)
	//fmt.Println(C.F_NO_LOGGING)

	name := C.CString("Gopher")
	defer C.free(unsafe.Pointer(name))

	year := C.int(2018)
	//ptr := C.malloc(C.sizeof_char * 1024)
	//defer C.free(unsafe.Pointer(ptr))
	//
	//size := C.greet(name, year, (*C.char)(ptr))
	//
	//b := C.GoBytes(ptr, size)
	//fmt.Println(string(b))
}
