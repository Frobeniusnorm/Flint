package optimize

import "fmt"

// TODO learning rate schedulers

type LRScheduler interface {
	fmt.Stringer
	Step()
}
