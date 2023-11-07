package datasets

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"io"
	"log"
	"os"
	"path"
	"strings"
)

type MnistDataset struct {
	baseDataset
	path string
	size uint
	data []MnistDatasetEntry
}

type MnistDatasetEntry struct {
	Label tensor.Tensor
	Data  tensor.Tensor
}

const (
	trainImagesFile = "train-images.idx3-ubyte"
	trainLabelsFile = "train-labels.idx1-ubyte"
	testImagesFile  = "t10k-images.idx3-ubyte"
	testLabelsFile  = "t10k-labels.idx1-ubyte"

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
func NewMnistDataset(baseDir string) (MnistDataset, MnistDataset, error) {
	trainDataset, err := loadMnistDataset(path.Join(baseDir, trainImagesFile), path.Join(baseDir, trainLabelsFile))
	if err != nil {
		return MnistDataset{}, MnistDataset{}, err
	}

	testDataset, err := loadMnistDataset(path.Join(baseDir, testImagesFile), path.Join(baseDir, testLabelsFile))
	if err != nil {
		return MnistDataset{}, MnistDataset{}, err
	}

	return trainDataset, testDataset, nil
}

func loadMnistDataset(imagePath string, labelPath string) (MnistDataset, error) {
	images, err := loadMnistImages(imagePath)
	if err != nil {
		return MnistDataset{}, err
	}

	labels, err := loadMnistLabels(labelPath)
	if err != nil {
		return MnistDataset{}, err
	}

	if images.size != labels.size {
		return MnistDataset{}, errors.New(fmt.Sprintf("number of labels (%d) and number of images (%d) does not match!", labels.size, images.size))
	}
	datasetSize := labels.size

	data := make([]MnistDatasetEntry, datasetSize)
	for i := uint32(0); i < datasetSize; i++ {
		imageByteSize := images.width * images.height
		imageOffset := i * imageByteSize
		imageData := images.data[imageOffset : imageOffset+imageByteSize]

		image := tensor.FromNodeWithErr(flint.CreateGraph(imageData, flint.Shape{uint(images.height), uint(images.width)}))
		class := labels.data[i]
		label := tensor.Scalar(int32(class))
		data[i] = MnistDatasetEntry{
			Label: label,
			Data:  image,
		}
	}

	name := "mnist"
	if strings.HasPrefix(path.Base(imagePath), "train") {
		name = name + "-train"
	} else {
		name = name + "-test"
	}
	dataset := MnistDataset{
		baseDataset: baseDataset{Name: name},
		path:        path.Dir(imagePath),
		size:        uint(datasetSize),
		data:        data,
	}
	return dataset, nil
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
	defer func(file *os.File) {
		_ = file.Close()
	}(file)

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
	defer func(file *os.File) {
		_ = file.Close()
	}(file)

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
func (d MnistDataset) Count() uint {
	return d.size
}

// Get returns a MnistDatasetEntry by index
func (d MnistDataset) Get(index uint) MnistDatasetEntry {
	return d.data[index]
}

func (d MnistDataset) Collate(items []MnistDatasetEntry) MnistDatasetEntry {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}
	labels := make([]tensor.Tensor, len(items))
	images := make([]tensor.Tensor, len(items))
	for idx, val := range items {
		labels[idx] = val.Label
		images[idx] = val.Data
	}
	res := MnistDatasetEntry{
		Label: TrivialCollate(labels),
		Data:  TrivialCollate(images),
	}
	res.Label = res.Label.Flatten()
	return res
}

func (d MnistDataset) String() string {
	return fmt.Sprintf("MnistDataset (%s) with %d items", d.Name, d.Count())
}
