package flint

import "github.com/Frobeniusnorm/Flint/go/wrapper"

func (x Tensor) convertImpl(tt DataType) Tensor {
	flintNode, err := wrapper.Convert(x.node, tt)
	return FromNodeWithErr(flintNode, err)
}
