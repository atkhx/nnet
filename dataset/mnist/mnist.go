package mnist

const (
	ImageWidth  = 28
	ImageHeight = 28
	ImageDepth  = 1
	ImageSize   = ImageWidth * ImageHeight * ImageDepth

	TrainSetImagesCount = 60000
	TestSetImagesCount  = 10000

	TrainImagesFileName = "train-images-idx3-ubyte"
	TrainLabelsFileName = "train-labels-idx1-ubyte"

	TestImagesFileName = "t10k-images-idx3-ubyte"
	TestLabelsFileName = "t10k-labels-idx1-ubyte"
)

var labels = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
