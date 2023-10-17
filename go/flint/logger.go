package flint

// #include <flint/flint.h>
import "C"
import "unsafe"

type loggingLevel int

const (
	NO_LOGGING loggingLevel = iota
	ERROR
	WARNING
	INFO
	VERBOSE
	DEBUG
)

// SetLoggingLevel sets the threshold for logging. Only message with a level higher or equal than [level] will be printed.
func SetLoggingLevel(level loggingLevel) {
	C.fSetLoggingLevel(C.enum_FLogType(level))
}

// Logging prints the [message] to the flint console in a given [loggingLevel] [level]
func Logging(level loggingLevel, message string) {
	unsafeMessage := C.CString(message)
	defer C.free(unsafe.Pointer(unsafeMessage))
	C.flogging(C.enum_FLogType(level), unsafeMessage)
}
