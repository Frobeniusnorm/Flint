package flint

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"path"
	"testing"
)

func TestLoadImage(t *testing.T) {
	node, err := LoadImage("does not exist lol")
	assert.Empty(t, node)
	assert.Error(t, err)
	assert.ErrorContains(t, err, "Flint Error")
	fmt.Println(node, err)
	node, err = LoadImage(path.Join("..", "..", "flint.png"))
	assert.NotEmpty(t, node)
	assert.NoError(t, err)
	fmt.Println(node, err)
}
