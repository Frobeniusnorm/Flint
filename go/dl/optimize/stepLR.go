package optimize

type StepLR struct {
	optim Optimizer
}

func NewStepLR(optim Optimizer) StepLR {
	return StepLR{
		optim: optim,
	}
}

func (s StepLR) Step() {
	// TODO
}

func (s StepLR) String() string {
	return "StepLR()"
}
