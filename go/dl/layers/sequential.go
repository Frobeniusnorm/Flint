package layers

import (
	"fmt"
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

func (seq Sequential) Forward(x Tensor) Tensor {
	var res Tensor
	res = x
	for _, l := range seq.layers {
		res = l.Forward(res)
	}
	return res
}

func (seq Sequential) Parameters(recurse bool) []Parameter {
	if recurse {
		var res []Parameter
		for _, layer := range seq.layers {
			res = append(res, layer.Parameters(recurse)...)
		}
		return res
	} else {
		return []Parameter{}
	}
}

func (seq Sequential) Train() {
	for _, layer := range seq.layers {
		layer.Train()
	}
}

func (seq Sequential) Eval() {
	for _, layer := range seq.layers {
		layer.Eval()
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
