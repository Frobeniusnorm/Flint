package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
)

type Dropout struct {
	probability float32
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		probability: prob,
	}
}

func (d Dropout) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (d Dropout) Forward(x dl.Tensor) dl.Tensor {
	return x // TODO
}

func (d Dropout) TrainMode() {}

func (d Dropout) EvalMode() {}

func (d Dropout) String() string {
	return fmt.Sprintf("Dropout(prob=%f)", d.probability)
}
