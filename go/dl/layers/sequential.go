package layers

import (
	"fmt"
	"strings"
)

/*
Sequential is a convenience module for executing a number of trainable layers one after another.
This saves writing the tedious and repetitive forward method
*/
type Sequential struct {
	BaseLayer
	layers []Layer
}

func NewSequential(layers ...Layer) Sequential {
	return Sequential{
		BaseLayer: BaseLayer{
			trainable:  true,
			EnableGrad: true,
		},
		layers: layers,
	}
}

//func NewSequentialFromLayerList(layers LayerList) Sequential {
//	return Sequential{}
//}

func (seq *Sequential) Forward(x Tensor) Tensor {
	var res Tensor
	res = x
	for _, l := range seq.layers {
		res = l.Forward(res)
	}
	return res
}

func (seq *Sequential) String() string {
	var res strings.Builder
	res.WriteString("Sequential([\n")
	for _, l := range seq.layers {
		res.WriteString(fmt.Sprintf("\t%s,\n", l.String()))
	}
	res.WriteString("])\n")
	return res.String()
}

// TODO: method for extending?
