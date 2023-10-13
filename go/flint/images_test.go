package flint

import (
	"fmt"
	"testing"
)

func TestLoadImage(t *testing.T) {
	//node, err := LoadImage("does not exist lol")
	//fmt.Println(node, err)
	// TODO: this does nothing except provoke weird errors
	node := CreateScalar(float32(2.4))
	node = Add(node, float32(1))
	node = Even(node)
	fmt.Println(node)
}
