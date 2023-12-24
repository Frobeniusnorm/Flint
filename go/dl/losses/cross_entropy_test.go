package losses

const (
	batchSize = 9
	labels    = 4
)

//func TestCrossEntropyLoss(t *testing.T) {
//	wrapper.Init(wrapper.BACKEND_BOTH)
//	wrapper.SetLoggingLevel(wrapper.INFO)
//
//	t.Run("test_zero_loss", func(t *testing.T) {
//		labelData := make([]float32, batchSize*labels)
//		labelNode := wrapper.CreateGraph(labelData, wrapper.Shape{batchSize, labels})
//		resNode := CrossEntropyLoss(labelNode, labelNode)
//		fmt.Println("label node:", wrapper.CalculateResult[float32](labelNode))
//		res := wrapper.CalculateResult[float32](resNode)
//		fmt.Println(res)
//	})
//
//	t.Run("test_high_loss", func(t *testing.T) {
//		labelData := make([]float32, batchSize*labels)
//		for i := 1; i < batchSize*labels; i += batchSize {
//			labelData[i] = 1
//		}
//		labelNode := wrapper.CreateGraph(labelData, wrapper.Shape{batchSize, labels})
//		fmt.Println("label node:", wrapper.CalculateResult[float32](labelNode))
//
//		inputData := make([]float32, batchSize*labels)
//		for i := 2; i < batchSize*labels; i += batchSize {
//			inputData[i] = 1
//		}
//		inputNode := wrapper.CreateGraph(inputData, wrapper.Shape{batchSize, labels})
//		fmt.Println("input node:", wrapper.CalculateResult[float32](inputNode))
//
//		resNode := CrossEntropyLoss(labelNode, inputNode)
//		res := wrapper.CalculateResult[float32](resNode)
//		fmt.Println(res)
//	})
//}
