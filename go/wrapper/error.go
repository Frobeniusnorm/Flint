package wrapper

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

// Error offers a generic error struct in wrapper
type Error struct {
	Message string
	Err     error
	Errno   syscall.Errno
}

// TODO: is a custom .Is and .As function needed? as seen in: https://go.dev/blog/go1.13-errors

func (err Error) Error() string {
	return fmt.Sprintf("%v (%v) - %s", err.Err, err.Errno, err.Message)
}

func (err Error) Unwrap() error {
	// return err.Err
	return errors.Join(err.Err, err.Errno)
}

// Defines all the possible errors of wrapper
var (
	ErrWrongType             = errors.New("wrong type")
	ErrIllegalDimension      = errors.New("illegal dimension")
	ErrIllegalDimensionality = errors.New("illegal dimensionality")
	ErrIncompatibleShapes    = errors.New("incompatible shapes")
	ErrInvalidSelect         = errors.New("invalid select")
	ErrOpenCL                = errors.New("OpenCL error")
	ErrInternal              = errors.New("internal error of flint")
	ErrOOM                   = errors.New("out of memory")
	ErrIllegalDerive         = errors.New("impossible to derive")
	ErrIO                    = errors.New("io error")
)

type errorCode int

const (
	noError errorCode = iota
	wrongType
	illegalDimension
	illegalDimensionality
	incompatibleShapes
	invalidSelect
	oclError
	internalError
	outOfMemory
	illegalDerive
	ioError
)

// TODO: need to assume that error message and code might not always be set correctly!

func buildErrorFromCode(err error, errCode errorCode) error {
	var errno syscall.Errno
	if err != nil && !errors.As(err, &errno) {
		panic("expected a Errno to build error from!")
	}

	errM := errorMessage()

	switch errCode {
	case noError:
		return nil
	case wrongType:
		return Error{
			Message: errM,
			Err:     ErrWrongType,
			Errno:   errno,
		}
	case illegalDimension:
		return Error{
			Message: errM,
			Err:     ErrIllegalDimension,
			Errno:   errno,
		}
	case illegalDimensionality:
		return Error{
			Message: errM,
			Err:     ErrIllegalDimensionality,
			Errno:   errno,
		}
	case incompatibleShapes:
		return Error{
			Message: errM,
			Err:     ErrIncompatibleShapes,
			Errno:   errno,
		}
	case invalidSelect:
		return Error{
			Message: errM,
			Err:     ErrInvalidSelect,
			Errno:   errno,
		}
	case oclError:
		return Error{
			Message: errM,
			Err:     ErrOpenCL,
			Errno:   errno,
		}
	case internalError:
		return Error{
			Message: errM,
			Err:     ErrInternal,
			Errno:   errno,
		}
	case outOfMemory:
		return Error{
			Message: errM,
			Err:     ErrOOM,
			Errno:   errno,
		}
	case illegalDerive:
		return Error{
			Message: errM,
			Err:     ErrIllegalDerive,
			Errno:   errno,
		}
	case ioError:
		return Error{
			Message: errM,
			Err:     ErrIO,
			Errno:   errno,
		}
	default:
		panic("unhandled error type")
	}
}

func buildError(err error) error {
	errCode := lastError()
	return buildErrorFromCode(err, errCode)
}

// lastError fetches the error type for the last recorded error.
// This should only be queried when an error is certain (e.g. nullptr returned from C)
func lastError() errorCode {
	return errorCode(C.fErrorType())
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
