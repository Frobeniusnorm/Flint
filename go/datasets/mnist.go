package datasets

import (
	"errors"
	"fmt"
	"io"
	"os"
)

type MnistDataset struct {
	Dataset
	size uint
}

const (
	trainImagesFile = "train-images-idx3-ubyte"
	trainLabelsFile = "train-labels-idx1-ubyte"
	testImagesFile  = "t10k-images-idx3-ubyte"
	testLabelsFile  = "t10k-labels-idx1-ubyte"

	labelsFileMagic int32 = 0x00000801
	imagesFileMagic int32 = 0x00000803
)

// readWord reads 4 bytes and converts to big endian integer
func readWord(reader io.Reader) (int32, error) {
	const wordLen int = 4 // 4 byte
	buf := make([]byte, wordLen)
	n, err := reader.Read(buf)
	if err != nil {
		return 0, err
	}
	if n != wordLen {
		return 0, errors.New("could not read full word for some reason")
	}

	var v int32 = 0
	for _, x := range buf {
		v = v*256 + int32(x)
	}
	return v, nil
}

// TODO: option to download mnist

// NewMnistDataset returns the train and test dataset of MNIST
func NewMnistDataset(path string) (MnistDataset, MnistDataset, error) {
	//train, err := loadMnistImages(path)
	//if err != nil {
	//	return MnistDataset{}, MnistDataset{}, err
	//}
	//test, err := loadTestMnist(path)
	//if err != nil {
	//	return MnistDataset{}, MnistDataset{}, err
	//}
	//return train, test, nil
	return MnistDataset{}, MnistDataset{}, nil
}

func loadMnistLabels(path string) (struct {
	size uint32
	data []byte
}, error) {
	result := struct {
		size uint32
		data []byte
	}{}

	file, err := os.Open(path)
	if err != nil {
		return result, err
	}
	defer file.Close()

	// magic number
	magic, err := readWord(file)
	if err != nil {
		return result, err
	}
	if magic != labelsFileMagic {
		return result, errors.New("magic number does not match")
	}

	// number of labels
	size, err := readWord(file)
	if err != nil {
		return result, err
	}

	// label data
	data := make([]byte, size)
	sizeRead, err := file.Read(data)
	if err != nil {
		return result, err
	}
	if int32(sizeRead) != size {
		return result, errors.New(fmt.Sprintf("number of read labels (%d) does not match expected number of labels: %d", sizeRead, size))
	}

	result.size = uint32(size)
	result.data = data
	return result, nil
}

func loadMnistImages(path string) (struct {
	size   uint32
	height uint32
	width  uint32
	data   []byte
}, error) {
	result := struct {
		size   uint32
		height uint32
		width  uint32
		data   []byte
	}{}

	file, err := os.Open(path)
	if err != nil {
		return result, err
	}
	defer file.Close()

	// magic number
	magic, err := readWord(file)
	if err != nil {
		return result, err
	}
	if magic != imagesFileMagic {
		return result, errors.New("magic number does not match")
	}

	// number of images
	size, err := readWord(file)
	if err != nil {
		return result, err
	}

	// number of rows
	height, err := readWord(file)
	if err != nil {
		return result, err
	}

	// number of columns
	width, err := readWord(file)
	if err != nil {
		return result, err
	}

	totalByteSize := size * height * width
	data := make([]byte, totalByteSize)

	sizeRead, err := file.Read(data)
	if err != nil {
		return result, err
	}
	if int32(sizeRead) != totalByteSize {
		return result, errors.New(fmt.Sprintf("number of read images does not match expected number of labels: %d", size))
	}

	result.size = uint32(size)
	result.width = uint32(width)
	result.height = uint32(height)
	result.data = data
	return result, nil
}

// Count returns the number of items in the dataset
// also used to implement the IndexableDataset interface
func (d *MnistDataset) Count() uint {
	return d.size
}

// Get returns a DatasetEntry by index
// also used to implement the IndexableDataset interface
func (d *MnistDataset) Get(index uint) DatasetEntry {
	return d.data[index]
}
