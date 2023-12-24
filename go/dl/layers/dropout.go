package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Dropout struct {
	probability float32
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		probability: prob,
	}
}

func (d Dropout) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{}
}

func (d Dropout) Forward(x flint.Tensor) flint.Tensor {
	return x // TODO
}

func (d Dropout) TrainMode() {}

func (d Dropout) EvalMode() {}

func (d Dropout) String() string {
	return fmt.Sprintf("Dropout(prob=%f)", d.probability)
}
