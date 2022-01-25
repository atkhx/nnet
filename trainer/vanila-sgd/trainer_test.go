package vanila_sgd

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/trainer/vanila-sgd/mocks"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTrainer_Activate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	inputs := data.NewVector(1, 0)
	target := data.NewVector(1)
	netOutput := data.NewVector(0.3)
	netDeltas := data.NewVector(0.7)

	loss := mocks.NewMockLoss(ctrl)
	loss.EXPECT().GetDeltas(target, netOutput).Return(netDeltas)

	net := mocks.NewMockNet(ctrl)
	net.EXPECT().Activate(inputs).Return(netOutput)
	net.EXPECT().Backprop(netDeltas)

	trainer := New(net, loss, 0.12)

	assert.EqualValues(t, netOutput, trainer.Activate(inputs, target))
}

func TestTrainer_UpdateWeights(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	learningRate := 0.1

	inputs := data.NewVector(1, 0)
	target := data.NewVector(1)

	netOutput := data.NewVector(0.3)
	netDeltas := data.NewVector(0.7)

	loss := mocks.NewMockLoss(ctrl)
	loss.EXPECT().GetDeltas(target, netOutput).Return(netDeltas)

	net := mocks.NewMockNet(ctrl)
	net.EXPECT().Activate(inputs).Return(netOutput)
	net.EXPECT().Backprop(netDeltas)
	net.EXPECT().GetLayersCount().Return(1).AnyTimes()

	layer := mocks.NewMockTrainableLayer(ctrl)
	net.EXPECT().GetLayer(0).Return(layer)

	layerWeights := data.NewVector(0.11, 0.22, 0.33)
	layerWeightsGradients := data.NewVector(0.1, 0.2, 0.3)

	layerBiases := data.NewVector(0.1, 0.2, 0.3)
	layerBiasesGradients := data.NewVector(0.3, 0.4, 0.5)

	layer.EXPECT().GetWeightsWithGradient().Return(layerWeights, layerWeightsGradients)
	layer.EXPECT().GetBiasesWithGradient().Return(layerBiases, layerBiasesGradients)

	trainer := New(net, loss, learningRate)
	trainer.Activate(inputs, target)
	trainer.UpdateWeights()

	expectedWeights := data.NewVector(0.11-learningRate*0.1, 0.22-learningRate*0.2, 0.33-learningRate*0.3)
	expectedBiases := data.NewVector(0.1-learningRate*0.3, 0.2-learningRate*0.4, 0.3-learningRate*0.5)

	assert.EqualValues(t, expectedWeights, layerWeights, "updated weights failed")
	assert.EqualValues(t, expectedBiases, layerBiases, "updated biases failed")

	expectedWeightsGradients := data.NewVector(0.1, 0.2, 0.3)
	expectedBiasesGradients := data.NewVector(0.3, 0.4, 0.5)

	assert.EqualValues(t, expectedWeightsGradients, layerWeightsGradients, "weight gradients changed")
	assert.EqualValues(t, expectedBiasesGradients, layerBiasesGradients, "biases gradients changed")
}
