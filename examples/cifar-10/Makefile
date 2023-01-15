.PHONY: train
train:
	go run cmd/train/main.go -d ./data/ -c ./config.json

.PHONY: test
test:
	go run cmd/test/main.go -d ./data/ -c ./config.json

.PHONY: dataset
dataset:
	curl http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz --fail --output ./cifar-10-batches-bin.tar.gz || exit 1
	tar xzvf ./cifar-10-batches-bin.tar.gz
	rm -rf ./data_tmp ./data
	mv cifar-10-batches-bin data_tmp
	cat ./data_tmp/data_batch_1.bin > ./data_tmp/train_bach.bin
	cat ./data_tmp/data_batch_2.bin >> ./data_tmp/train_bach.bin
	cat ./data_tmp/data_batch_3.bin >> ./data_tmp/train_bach.bin
	cat ./data_tmp/data_batch_4.bin >> ./data_tmp/train_bach.bin
	cat ./data_tmp/data_batch_5.bin >> ./data_tmp/train_bach.bin
	mkdir data
	mv ./data_tmp/train_bach.bin ./data
	mv ./data_tmp/test_batch.bin ./data
	rm -rf ./data_tmp
	rm -rf ./cifar-10-batches-bin.tar.gz
