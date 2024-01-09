package wrapper

// #include <flint/flint.h>
import "C"
import "unsafe"

type loggingLevel int

const (
	LOG_OFF loggingLevel = iota
	LOG_ERROR
	LOG_WARNING
	LOG_INFO
	LOG_VERBOSE
	LOG_DEBUG
)

// SetLoggingLevel sets the threshold for logging. Only message with a level higher or equal than [level] will be printed.
func SetLoggingLevel(level loggingLevel) {
	if level < LOG_OFF || level > LOG_DEBUG {
		panic("invalid logging level")
	}
	C.fSetLoggingLevel(C.enum_FLogType(level))
}

// Logging prints the [message] to the wrapper console in a given [loggingLevel] [level]
func Logging(level loggingLevel, message string) {
	unsafeMessage := C.CString(message)
	defer C.free(unsafe.Pointer(unsafeMessage))
	C.flogging(C.enum_FLogType(level), unsafeMessage)
}
