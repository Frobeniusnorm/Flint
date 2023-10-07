package dataloader

import (
	"testing"
)

func TestQueue(t *testing.T) {
	q := NewQueue[int](10)
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)

	if q.Length() != 3 {
		t.Errorf("Expected length 3, got %d", q.Length())
	}

	item, ok := q.Peek()
	if !ok || item != 1 {
		t.Errorf("Expected peek item 1, got %v", item)
	}

	item, ok = q.Dequeue()
	if !ok || item != 1 {
		t.Errorf("Expected dequeued item 1, got %v", item)
	}

	item, ok = q.Dequeue()
	if !ok || item != 2 {
		t.Errorf("Expected dequeued item 2, got %v", item)
	}

	item, ok = q.Dequeue()
	if !ok || item != 3 {
		t.Errorf("Expected dequeued item 3, got %v", item)
	}

	if q.Length() != 0 {
		t.Errorf("Expected length 0, got %d", q.Length())
	}
}
