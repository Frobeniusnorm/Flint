package loss

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"testing"
)

const (
	batchSize = 9
	labels    = 4
)

func TestCrossEntropyLoss(t *testing.T) {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	t.Run("test_zero_loss", func(t *testing.T) {
		labelData := make([]float32, batchSize*labels)
		labelNode := flint.CreateGraph(labelData, flint.Shape{batchSize, labels}, flint.F_FLOAT32)
		resNode := CrossEntropyLoss(labelNode, labelNode)
		fmt.Println("label node:", flint.CalculateResult[float32](labelNode))
		res := flint.CalculateResult[float32](resNode)
		fmt.Println(res)
	})

	t.Run("test_high_loss", func(t *testing.T) {
		labelData := make([]float32, batchSize*labels)
		for i := 1; i < batchSize*labels; i += batchSize {
			labelData[i] = 1
		}
		labelNode := flint.CreateGraph(labelData, flint.Shape{batchSize, labels}, flint.F_FLOAT32)
		fmt.Println("label node:", flint.CalculateResult[float32](labelNode))

		inputData := make([]float32, batchSize*labels)
		for i := 2; i < batchSize*labels; i += batchSize {
			inputData[i] = 1
		}
		inputNode := flint.CreateGraph(inputData, flint.Shape{batchSize, labels}, flint.F_FLOAT32)
		fmt.Println("input node:", flint.CalculateResult[float32](inputNode))

		resNode := CrossEntropyLoss(labelNode, inputNode)
		res := flint.CalculateResult[float32](resNode)
		fmt.Println(res)
	})
}
