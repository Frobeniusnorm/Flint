package main

import "flint"

func main() {
	flint.Init(flint.BACKEND_ONLY_CPU)
	flint.SetLoggingLevel(flint.DEBUG)
}
