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

	nextIndex, err = linearSampler(&remainingIndices)
	assert.Equal(t, uint(2), nextIndex)
	assert.NoError(t, err)

	nextIndex, err = linearSampler(&remainingIndices)
	assert.Equal(t, uint(0), nextIndex)
	assert.ErrorIs(t, err, Done)

	nextIndex, err = linearSampler(&remainingIndices)
	assert.Equal(t, uint(0), nextIndex)
	assert.ErrorIs(t, err, Done)
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
	assert.Equal(t, uint(1), nextIndex)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.Equal(t, uint(2), nextIndex)
	assert.NoError(t, err)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.Equal(t, uint(0), nextIndex)
	assert.ErrorIs(t, err, Done)

	nextIndex, err = randomSampler(&remainingIndices)
	assert.Equal(t, uint(0), nextIndex)
	assert.ErrorIs(t, err, Done)
}

func TestLinearBatchSampler(t *testing.T) {
	t.Run("happy case", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		nextIndices, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.Len(t, remainingIndices, 1)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Equal(t, []uint{0, 1}, nextIndices)
	})

	t.Run("dataset size not divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		_, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// provoke error
		_, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		_, err := linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// another happy execution
		_, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// provoke error
		_, err = linearBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size not divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := linearBatchSampler(&remainingIndices, 2, true)
		assert.Len(t, nextIndices, 2)
		assert.NoError(t, err)
		// another happy (but short) execution
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.Len(t, nextIndices, 1)
		assert.NoError(t, err)
		// provoke error
		nextIndices, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		_, err := linearBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		// another happy execution
		_, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		// provoke error
		_, err = linearBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
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
		assert.Len(t, remainingIndices, 1)
		assert.NoError(t, err)
		assert.Len(t, nextIndices, 2)
		assert.Less(t, nextIndices[0], uint(3))
		assert.Less(t, nextIndices[1], uint(3))
	})

	t.Run("dataset size not divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		_, err := randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// provoke error
		_, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size divisible by batch size and no dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 4)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 4)

		// happy execution
		_, err := randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// another happy execution
		_, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.NoError(t, err)
		// provoke error
		_, err = randomBatchSampler(&remainingIndices, 2, false)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size not divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		nextIndices, err := randomBatchSampler(&remainingIndices, 2, true)
		assert.Len(t, nextIndices, 2)
		assert.NoError(t, err)
		// another happy (but short) execution
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.Len(t, nextIndices, 1)
		assert.NoError(t, err)
		// provoke error
		nextIndices, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
	})

	t.Run("dataset size divisible by batch size and dropLast", func(t *testing.T) {
		remainingIndices := make([]uint, 3)
		for i := 0; i < 3; i++ {
			remainingIndices[i] = uint(i)
		}
		assert.Len(t, remainingIndices, 3)

		// happy execution
		_, err := randomBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		// another happy execution
		_, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.NoError(t, err)
		// provoke error
		_, err = randomBatchSampler(&remainingIndices, 2, true)
		assert.ErrorIs(t, err, Done)
	})
}
