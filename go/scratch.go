package scratch

import "fmt"

//func convertToCNumber[In completeNumeric, Out cNumbers](n In) Out {
//	switch v := any(n).(type) {
//	case int, int8, int16, int32:
//		return C.int(v)
//	case int64:
//		return C.long(v)
//	case uint, uint8, uint16, uint32:
//		return C.uint(v)
//	case float32:
//		return C.float(v)
//	case float64:
//		return C.double(v)
//	default:
//		// Handle this case appropriately; maybe panic or return a zero value
//		panic("unsupported type")
//	}
//}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

// See: https://stackoverflow.com/questions/71587996/cannot-use-type-assertion-on-type-parameter-value
//func convertArrayOld[In completeNumeric, Out cNumbers](arr []In) []Out {
//	result := make([]Out, len(arr))
//	for idx, val := range arr {
//		result[idx] = Out(val) //nolint:all
//		//x := any(val).(Out)
//		//result[idx] = x
//	}
//	return result
//}
