package layers

import "fmt"

type Dropout struct {
	probability float32
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		probability: prob,
	}
}

func (d Dropout) Parameters(recurse bool) []Parameter {
	return []Parameter{}
}

func (d Dropout) Forward(x Tensor) Tensor {
	return x // TODO
}

func (d Dropout) TrainMode() {}

func (d Dropout) EvalMode() {}

func (d Dropout) String() string {
	return fmt.Sprintf("Dropout(prob=%f)", d.probability)
}
