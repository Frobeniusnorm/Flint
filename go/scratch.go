package scratch

// TransposeFull acts like a traditional transpose where the order of all axis are reversed
func TransposeFull(node GraphNode) (GraphNode, error) {
	n := len(node.GetShape())
	ax := make(Axes, n)
	for i := 0; i < n; i++ {
		ax[i] = uint(n - i - 1)
	}
	return Transpose(node, ax)
}

func OneHot(node GraphNode, numClasses uint) (GraphNode, error) {
	id, err := CreateGraphIdentity(numClasses)
	if err != nil {
		return GraphNode{}, err
	}

	id, err = Index(id, node)
	if err != nil {
		return GraphNode{}, err
	}

	return id, nil
}
