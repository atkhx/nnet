.PHONY: train
train:
	go run cmd/train/main.go -d ./data/ -c ./config.json

.PHONY: test
test:
	go run cmd/test/main.go -d ./data/ -c ./config.json

.PHONY: dataset
dataset:
	cd data
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --fail --output ./data/train-images-idx3-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --fail --output ./data/train-labels-idx1-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --fail --output ./data/t10k-images-idx3-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --fail --output ./data/t10k-labels-idx1-ubyte.gz
	gunzip ./data/*.gz

