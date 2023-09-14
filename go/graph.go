package flint

type GraphContext struct {
	loggingLevel   loggingLevel
	eagerExecution bool
	backend        Backend
}

func (ctx GraphContext) Start() {

}

func (ctx GraphContext) End() {
	// TODO: maybe return result
}
