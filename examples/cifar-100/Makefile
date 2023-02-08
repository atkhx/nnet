.PHONY: train
train:
	go run cmd/train/main.go -d ./data/ -c ./config.json

.PHONY: test
test:
	go run cmd/test/main.go -d ./data/ -c ./config.json

.PHONY: dataset
dataset:
	curl http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz --fail --output ./cifar100.tar.gz || exit 1
	tar xzvf ./cifar100.tar.gz
	mv ./cifar-100-binary/train.bin ./data/cifar100-train-data.bin
	mv ./cifar-100-binary/test.bin ./data/cifar100-test-data.bin
	rm -rf ./cifar-100-binary
	rm -rf ./cifar100.tar.gz

