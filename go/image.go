package flint

// TODO: maybe use "image" library from go instead of flint image lib
// see: https://go.dev/tour/methods/24

type imageFormat int

const (
	PNG imageFormat = iota
	JPEG
	BMP
)

func LoadImage(path string) Tensor {
	return 0
}

func StoreImage(Tensor, path string, format imageFormat) {
}
