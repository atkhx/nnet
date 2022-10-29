package mnist

const (
	ImageWidth  = 28
	ImageHeight = 28
	ImageSize   = ImageWidth * ImageHeight

	TrainSetImagesCount = 60000
	TestSetImagesCount  = 10000

	TrainImagesFileName = "train-images-idx3-ubyte"
	TrainLabelsFileName = "train-labels-idx1-ubyte"

	TestImagesFileName = "t10k-images-idx3-ubyte"
	TestLabelsFileName = "t10k-labels-idx1-ubyte"
)
