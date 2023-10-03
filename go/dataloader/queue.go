package dataloader

// Queue represents the data structure for a simple FIFO queue.
type Queue[T any] struct {
	items []T
}

// NewQueue returns an initialized queue
func NewQueue[T any](initialCapacity uint) Queue[T] {
	return Queue[T]{
		items: make([]T, 0, initialCapacity),
	}
}

// Enqueue adds an item to the queue.
func (q *Queue[T]) Enqueue(item T) {
	q.items = append(q.items, item)
}

// Dequeue removes and returns the first item from the queue
func (q *Queue[T]) Dequeue() (item T, ok bool) {
	if len(q.items) == 0 {
		return item, false
	}
	item = q.items[0]
	q.items = q.items[1:]
	return item, true
}

func (q *Queue[T]) DequeueAll() (items []T) {
	items = q.items
	q.items = make([]T, 0, cap(items))
	return items
}

// Length returns the number of items in the queue
func (q *Queue[T]) Length() uint {
	return uint(len(q.items))
}

// Peek returns the first item from the queue without removing it
func (q *Queue[T]) Peek() (item T, ok bool) {
	if len(q.items) == 0 {
		return item, false
	} else {
		return q.items[0], true
	}
}
