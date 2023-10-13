package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"strings"
)

/*
Sequential is a convenience module for executing a number of layers one after another.
This saves writing the tedious and repetitive forward method
*/
type Sequential struct {
	layers        []Layer
	trainingPhase bool
}

func NewSequential(layers ...Layer) Sequential {
	return Sequential{
		layers: layers,
	}
}

func (seq Sequential) Forward(x dl.Tensor) dl.Tensor {
	var res dl.Tensor
	res = x
	for _, l := range seq.layers {
		res = l.Forward(res)
	}
	return res
}

func (seq Sequential) Parameters(recurse bool) []dl.Parameter {
	if recurse {
		var res []dl.Parameter
		for _, layer := range seq.layers {
			res = append(res, layer.Parameters(recurse)...)
		}
		return res
	} else {
		return []dl.Parameter{}
	}
}

func (seq Sequential) TrainMode() {
	for _, layer := range seq.layers {
		layer.TrainMode()
	}
}

func (seq Sequential) EvalMode() {
	for _, layer := range seq.layers {
		layer.EvalMode()
	}
}

func (seq Sequential) String() string {
	var res strings.Builder
	res.WriteString("Sequential([\n")
	for _, l := range seq.layers {
		res.WriteString(fmt.Sprintf("\t%s,\n", l.String()))
	}
	res.WriteString("])\n")
	return res.String()
}

// TODO: method for extending?