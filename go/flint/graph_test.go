package flint

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateGraph(t *testing.T) {
	t.Run("happy case", func(t *testing.T) {
		node := CreateGraph([]uint16{1, 3, 2, 4}, Shape{2, 2})
		assert.Equal(t, Shape{2, 2}, node.GetShape())
		fmt.Println(CalculateResult[uint8](node))
		// TODO: check result using calculateResult
	})

	t.Run("data and shape mismatch", func(t *testing.T) {
		assert.Panics(t, func() {
			CreateGraph([]uint{1, 2, 3}, Shape{2, 2})
		})
	})
}

func TestCreateScalar(t *testing.T) {

}

func TestCreateGraphConstant(t *testing.T) {

}

func TestCreateGraphRandom(t *testing.T) {

}
