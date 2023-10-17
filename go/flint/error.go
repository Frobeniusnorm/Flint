package flint

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -lm
#include <flint/flint.h>
#include <errno.h>

// simple go function for the go interface to reset the errno
// WARNING: this MAY break in multi-threaded scenarios
void reset_errno(void) {
	errno = 0;
}
*/
import "C"
import (
	"errors"
	"fmt"
	"io/fs"
	"syscall"
)

// baseError offers a generic error struct in flint
type baseError struct {
	message string
	errno   syscall.Errno
}

//var (
//	ErrWrongType = errors.New
//	ErrIllegalDimension
//	ErrIllegalDimenisonality
//	ErrIncompatibleShapes
//	ErrInvalidSelect
//)

type Error baseError

func (err Error) Error() string {
	return fmt.Sprintf("Flint Error (%s) - %s", err.errno.Error(), err.message)
}

type WrongType baseError

func (err WrongType) Error() string {
	return fmt.Sprintf("Wrong Type (%s) - %s", err.errno.Error(), err.message)
}

type IllegalDimension baseError

func (err IllegalDimension) Error() string {
	return fmt.Sprintf("Illegal Dimension (%s) - %s", err.errno.Error(), err.message)
}

type IllegalDimensionality baseError

func (err IllegalDimensionality) Error() string {
	return fmt.Sprintf("Illegal Dimensionality (%s) - %s", err.errno.Error(), err.message)
}

type IncompatibleShapes baseError

func (err IncompatibleShapes) Error() string {
	return fmt.Sprintf("Incompatible Shapes (%s) - %s", err.errno.Error(), err.message)
}

type InvalidSelect baseError

func (err InvalidSelect) Error() string {
	return fmt.Sprintf("Invalid Select (%s) - %s", err.errno.Error(), err.message)
}

func BuildError(err error) error {
	const (
		NO_ERROR errType = iota
		WRONG_TYPE
		ILLEGAL_DIMENSION
		ILLEGAL_DIMENSIONALITY
		INCOMPATIBLE_SHAPES
		INVALID_SELECT
	)

	errors.Is
	fs.ErrNotExist

	var errno syscall.Errno
	if !errors.As(err, &errno) {
		panic("expected a Errno to build error from!")
	}

	errT := errorType()
	errM := errorMessage()

	switch errT {
	case NO_ERROR:
		return Error{
			message: errM,
			errno:   errno,
		}
	case WRONG_TYPE:
		return WrongType{
			message: errM,
			errno:   errno,
		}
	case ILLEGAL_DIMENSION:
		return IllegalDimension{
			message: errM,
			errno:   errno,
		}
	case ILLEGAL_DIMENSIONALITY:
		return IllegalDimensionality{
			message: errM,
			errno:   errno,
		}
	case INCOMPATIBLE_SHAPES:
		return IncompatibleShapes{
			message: errM,
			errno:   errno,
		}
	case INVALID_SELECT:
		return InvalidSelect{
			message: errM,
			errno:   errno,
		}
	default:
		panic("unhandled error type")
	}
}

type errType uint

// errorType fetches the error type for the last recorded error.
// This should only be queried when an error is certain (e.g. nullptr returned from C)
func errorType() errType {
	return errType(C.fErrorType())
}

// errorMessage fetches the error message for the last recorded error.
// This should only be queried when an error is certain (e.g. nullptr returned from C)
func errorMessage() string {
	// NOTE: it is NOT required to free the char* result!
	return C.GoString(C.fErrorMessage())
}

// resetErrno resets the errno for flintC.
// All functions calls to the C header are dispatched to the same thread.
// Therefore, no issues like race-conditions should occur.
// However, this function should only be used in edge cases and not replace common sense error handling.
func resetErrno() {
	C.reset_errno()
}
