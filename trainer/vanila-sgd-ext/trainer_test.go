package vanila_sgd_ext

import (
	"github.com/atkhx/nnet/data"
	"github.com/atkhx/nnet/trainer/vanila-sgd-ext/mocks"
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

	trainer := New(net, loss, 0.12, 0.07, 0.01)

	assert.EqualValues(t, netOutput, trainer.Activate(inputs, target))
}

func TestTrainer_UpdateWeights(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	inputs := data.NewVector(1, 0, 1)
	target := data.NewVector(1, 0, 0)

	netOutput := data.NewVector(0.3, 0, 0)
	netDeltas := data.NewVector(0.7, 0, 0)

	loss := mocks.NewMockLoss(ctrl)
	loss.EXPECT().GetDeltas(target, netOutput).Return(netDeltas)

	net := mocks.NewMockNet(ctrl)
	net.EXPECT().Activate(inputs).Return(netOutput)
	//net.EXPECT().GetOutputDeltas(target, netOutput).Return(netDeltas)
	net.EXPECT().Backprop(netDeltas)
	net.EXPECT().GetLayersCount().Return(1).AnyTimes()

	layer := mocks.NewMockTrainableLayer(ctrl)
	net.EXPECT().GetLayer(0).Return(layer).AnyTimes()

	layerWeights := data.NewVector(0.1, 0.2, 0.3)
	layerWeightsGradients := data.NewVector(0.1, 0.2, 0.3)

	layerBiases := data.NewVector(0.5, 0.5, 0.7)
	layerBiasesGradients := data.NewVector(0.3, 0.4, 0.5)

	layer.EXPECT().GetWeightsWithGradient().Return(layerWeights, layerWeightsGradients).AnyTimes()
	layer.EXPECT().GetBiasesWithGradient().Return(layerBiases, layerBiasesGradients).AnyTimes()

	trainer := New(net, loss, 0.12, 0.07, 0.01)
	trainer.Activate(inputs, target)
	trainer.UpdateWeights()

	assert.EqualValues(t, data.NewVector(0.08700000000000001, 0.17400000000000002, 0.261), layerWeights)
	assert.EqualValues(t, data.NewVector(0.459, 0.447, 0.633), layerBiases)

	assert.EqualValues(t, data.NewVector(0.1, 0.2, 0.3), layerWeightsGradients)
	assert.EqualValues(t, data.NewVector(0.3, 0.4, 0.5), layerBiasesGradients)
}
