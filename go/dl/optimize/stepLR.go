package optimize

type StepLR struct {
	optim     Optimizer
	stepSize  uint
	gamma     float32
	lastEpoch int
}

func NewStepLR(optim Optimizer, stepSize uint, gamma float32) StepLR {
	return StepLR{
		optim:     optim,
		stepSize:  stepSize,
		gamma:     gamma,
		lastEpoch: -1,
	}
}

func (s StepLR) Step() {
	// TODO
}

func (s StepLR) String() string {
	return "StepLR()"
}
