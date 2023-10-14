package losses

import "fmt"

type Losses interface {
	fmt.Stringer
}

type reduction uint8

const (
	REDUCE_NONE reduction = iota
	REDUCE_MEAN
	REDUCE_SUM
)
