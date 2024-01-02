package wrapper

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateGraph(t *testing.T) {
	t.Run("happy case", func(t *testing.T) {
		node, err := CreateGraphDataInt([]Int{1, 3, 2, 4}, Shape{2, 2})
		assert.NoError(t, err)
		assert.Equal(t, Shape{2, 2}, node.GetShape())
		fmt.Println(CalculateResult[Int](node))
		// TODO: check result using calculateResult
	})

	t.Run("data and shape mismatch", func(t *testing.T) {
		node, err := CreateGraphDataInt([]Int{1, 2, 3}, Shape{2, 2})
		assert.Error(t, err)
		assert.Empty(t, node)
	})
}
