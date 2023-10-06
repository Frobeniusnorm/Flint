package dataloader

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLinearSampler(t *testing.T) {
	remainingIndices := make([]uint, 3)
	for i := 0; i < 3; i++ {
		remainingIndices[i] = uint(i)
	}
	assert.Len(t, remainingIndices, 3)

	// test basic happy case
	nextIndex, err := linearSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Equal(t, uint(0), nextIndex)
	assert.Len(t, remainingIndices, 2)

	// test the limits (run out of numbers!)
	nextIndex, err = linearSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Equal(t, uint(1), nextIndex)
	assert.Len(t, remainingIndices, 1)

	nextIndex, err = linearSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Equal(t, uint(2), nextIndex)
	assert.Len(t, remainingIndices, 0)

	nextIndex, err = linearSampler(&remainingIndices)
	assert.ErrorIs(t, err, Done)
	assert.Equal(t, uint(0), nextIndex)
	assert.Len(t, remainingIndices, 0)

	nextIndex, err = linearSampler(&remainingIndices)
	assert.ErrorIs(t, err, Done)
	assert.Equal(t, uint(0), nextIndex)
	assert.Len(t, remainingIndices, 0)
}

func TestRandomSampler(t *testing.T) {
	remainingIndices := make([]uint, 3)
	for i := 0; i < 3; i++ {
		remainingIndices[i] = uint(i)
	}
	assert.Len(t, remainingIndices, 3)

	// test basic case
	nextIndex, err := randomSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Less(t, nextIndex, uint(3))
	assert.Len(t, remainingIndices, 2)

	// test the limits
	nextIndex, err = randomSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Less(t, nextIndex, uint(3))
	assert.Len(t, remainingIndices, 1)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.NoError(t, err)
	assert.Less(t, nextIndex, uint(3))
	assert.Len(t, remainingIndices, 0)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.ErrorIs(t, err, Done)
	assert.Equal(t, uint(0), nextIndex)
	assert.Len(t, remainingIndices, 0)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.ErrorIs(t, err, Done)
	assert.Equal(t, uint(0), nextIndex)
	assert.Len(t, remainingIndices, 0)
}

func TestLinearBatchSampler(t *testing.T) {
	t.Run("happy case", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		nextIndices, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
		assert.Len(t, remainingIndices, 1)
	})

	t.Run("dataset size not divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
		assert.Len(t, remainingIndices, 1)
		// should run fine
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 1)
		assert.Equal(t, []uint{2}, nextIndices)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})

	t.Run("dataset size divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 4; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		nextIndices, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
		assert.Len(t, remainingIndices, 2)
		// should run fine
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{2, 3}, nextIndices)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})

	t.Run("dataset size not divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := linearBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
		assert.Len(t, remainingIndices, 1)
		// provoke error
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 1)
	})

	t.Run("dataset size divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 4; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		nextIndices, err := linearBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
		assert.Len(t, remainingIndices, 2)
		// should run fine
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{2, 3}, nextIndices)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})
}

func TestRandomBatchSampler(t *testing.T) {
	t.Run("happy case", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		nextIndices, err := randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 1)
	})

	t.Run("dataset size not divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 1)
		// should run fine
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 1)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})

	t.Run("dataset size divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 4; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		nextIndices, err := randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 2)
		// should run fine
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})

	t.Run("dataset size not divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := randomBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 1)
		// provoke error
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 1)
	})

	t.Run("dataset size divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 4; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		nextIndices, err := randomBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 2)
		// should run fine
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Len(t, remainingIndices, 0)
		// provoke error
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
		assert.Len(t, remainingIndices, 0)
	})
}
