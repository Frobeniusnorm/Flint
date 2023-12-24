package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

type Dropout struct {
	probability float32
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		probability: prob,
	}
}

func (d Dropout) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{}
}

func (d Dropout) Forward(x flint_old.Tensor) flint_old.Tensor {
	return x // TODO
}

func (d Dropout) TrainMode() {}

func (d Dropout) EvalMode() {}

func (d Dropout) String() string {
	return fmt.Sprintf("Dropout(prob=%f)", d.probability)
}
