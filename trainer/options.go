package trainer

import "github.com/atkhx/nnet/trainer/methods"

func applyOptions(trainer *trainer, options ...Option) {
	for _, opt := range options {
		opt(trainer)
	}
}

var defaults = []Option{
	WithMethod(methods.Adadelta(Ro, Eps)),
	WithBatchSize(1),
}

type Option func(trainer *trainer)

func WithMethod(method Method) Option {
	return func(trainer *trainer) {
		trainer.method = method
	}
}

func WithL1Decay(l1decay float64) Option {
	return func(trainer *trainer) {
		trainer.l1Decay = l1decay
	}
}

func WithL2Decay(l2decay float64) Option {
	return func(trainer *trainer) {
		trainer.l2Decay = l2decay
	}
}

func WithBatchSize(batchSize int) Option {
	return func(trainer *trainer) {
		trainer.batchSize = batchSize
	}
}
