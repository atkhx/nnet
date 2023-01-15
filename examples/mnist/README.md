# MNIST Dataset Example

Original dataset you could find on http://yann.lecun.com/exdb/mnist/

## How to play

### Prepare dataset

Download dataset files to `./data` path

```bash
make dataset
```

### Prepare your neural network config

You could find some default configuration for convolutional neural network in `./pkg/net.go`.

**Important!** If you decide to change network configuration you must delete the previously trained `./config.json` file.

Otherwise, you will see nice panic in output.

### Train your neural network 

Run the training process with command `make train`.

It will make one iteration by all training-set images (60k images).

```bash
make train
```

On the script interruption or finishing neural network configuration will be saved into `./config.json` file.

Next launches will start training with stored in `./config.json` file weights values.

### Test your trained neural network

Run the testing process with command `make test`.

Script will load previously trained network configuration from `./config.json` file.

It will make one iteration by all testing-set images (10k) and show some statistics in output.

```bash
make test
```
