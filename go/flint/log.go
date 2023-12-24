package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func lnImpl(x Tensor) Tensor {
	node, err := wrapper.Log(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Ln(x Tensor) Tensor {
	return lnImpl(x)
}

func (x Tensor) Ln() Tensor {
	return lnImpl(x)
}

func (x *Tensor) Ln_() {
	*x = lnImpl(*x)
}

func log2Impl(x Tensor) Tensor {
	node, err := wrapper.Log2(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Log2(x Tensor) Tensor {
	return log2Impl(x)
}

func (x Tensor) Log2() Tensor {
	return log2Impl(x)
}

func (x *Tensor) Log2_() {
	*x = log2Impl(*x)
}

func log10Impl(x Tensor) Tensor {
	node, err := wrapper.Log10(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Log10(x Tensor) Tensor {
	return log10Impl(x)
}

func (x Tensor) Log10() Tensor {
	return log10Impl(x)
}

func (x *Tensor) Log10_() {
	*x = log10Impl(*x)
}
