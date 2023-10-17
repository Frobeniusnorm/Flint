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
	"syscall"
)

// Error offers a generic error struct in flint
type Error struct {
	Message string
	Err     error
	Errno   syscall.Errno
}

// TODO: is a custom .Is and .As function needed? as seen in: https://go.dev/blog/go1.13-errors

func (err *Error) Error() string {
	return fmt.Sprintf("%v (%v) - %s", err.Err, err.Errno, err.Message)
}

func (err *Error) Unwrap() error {
	// return err.Err
	return errors.Join(err.Err, err.Errno)
}

// Defines all the possible errors of flint
var (
	ErrNoError               = errors.New("no error")
	ErrWrongType             = errors.New("wrong type")
	ErrIllegalDimension      = errors.New("illegal dimension")
	ErrIllegalDimensionality = errors.New("illegal dimensionality")
	ErrIncompatibleShapes    = errors.New("incompatible shapes")
	ErrInvalidSelect         = errors.New("invalid select")
)

func BuildError(err error) error {
	const (
		NO_ERROR errType = iota
		WRONG_TYPE
		ILLEGAL_DIMENSION
		ILLEGAL_DIMENSIONALITY
		INCOMPATIBLE_SHAPES
		INVALID_SELECT
	)

	var errno syscall.Errno
	if !errors.As(err, &errno) {
		panic("expected a Errno to build error from!")
	}

	errT := errorType()
	errM := errorMessage()

	switch errT {
	case NO_ERROR:
		return &Error{
			Message: errM,
			Err:     ErrNoError,
			Errno:   errno,
		}
	case WRONG_TYPE:
		return &Error{
			Message: errM,
			Err:     ErrWrongType,
			Errno:   errno,
		}
	case ILLEGAL_DIMENSION:
		return &Error{
			Message: errM,
			Err:     ErrIllegalDimension,
			Errno:   errno,
		}
	case ILLEGAL_DIMENSIONALITY:
		return &Error{
			Message: errM,
			Err:     ErrIllegalDimensionality,
			Errno:   errno,
		}
	case INCOMPATIBLE_SHAPES:
		return &Error{
			Message: errM,
			Err:     ErrIncompatibleShapes,
			Errno:   errno,
		}
	case INVALID_SELECT:
		return &Error{
			Message: errM,
			Err:     ErrInvalidSelect,
			Errno:   errno,
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
