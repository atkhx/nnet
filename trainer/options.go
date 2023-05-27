package trainer

import "github.com/atkhx/nnet/trainer/methods"

func applyOptions(trainer *Trainer, options ...Option) {
	for _, opt := range options {
		opt(trainer)
	}
}

var defaults = []Option{
	WithMethod(methods.Adadelta(Ro, Eps)),
}

type Option func(trainer *Trainer)

func WithMethod(method Method) Option {
	return func(trainer *Trainer) {
		trainer.method = method
	}
}

func WithL1Decay(l1decay float64) Option {
	return func(trainer *Trainer) {
		trainer.l1Decay = l1decay
	}
}

func WithL2Decay(l2decay float64) Option {
	return func(trainer *Trainer) {
		trainer.l2Decay = l2decay
	}
}
