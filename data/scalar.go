package data

import (
	"fmt"
	"math"
)

func NewScalar(v float64) *Scalar {
	return &Scalar{Data: v}
}

func NewScalarResult(v float64, from *Source) *Scalar {
	return &Scalar{Data: v, From: from}
}

type Scalar struct {
	Data           float64
	Grad           float64
	From           *Source
	backwardCalled bool
}

func (a *Scalar) Add(b *Scalar) (out *Scalar) {
	return NewScalarResult(a.Data+b.Data, NewSource(func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}, a, b))
}

func (a *Scalar) Sub(b *Scalar) (out *Scalar) {
	return NewScalarResult(a.Data-b.Data, NewSource(func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}, a, b))
}

func (a *Scalar) Mul(b *Scalar) (out *Scalar) {
	return NewScalarResult(a.Data*b.Data, NewSource(func() {
		a.Grad += out.Grad * b.Data
		b.Grad += out.Grad * a.Data
	}, a, b))
}

func (a *Scalar) Pow(b *Scalar) (out *Scalar) {
	return NewScalarResult(math.Pow(a.Data, b.Data), NewSource(func() {
		a.Grad += 2.0 * math.Pow(a.Data, b.Data-1) * out.Grad
	}, a, b))
}

func (a *Scalar) Tanh() (out *Scalar) {
	return NewScalarResult(math.Tanh(a.Data), NewSource(func() {
		a.Grad += out.Grad * (1 - out.Data*out.Data)
	}, a))
}

func (a *Scalar) backward() {
	if a.backwardCalled {
		return
	}

	a.backwardCalled = true

	if a.From != nil {
		if a.From.Callback != nil {
			a.From.Callback()
		}

		for _, prev := range a.From.Parents {
			prev.backward()
		}
	}
}

func (a *Scalar) Backward() {
	a.Grad = 1
	a.backward()
}

func (a *Scalar) resetGrad() {
	a.Grad = 0
	a.backwardCalled = false

	if a.From == nil {
		return
	}

	for _, prev := range a.From.Parents {
		prev.resetGrad()
	}
}

func (a *Scalar) ResetGrad() {
	a.Grad = 0
	a.resetGrad()
}

func (a *Scalar) String() string {
	if a.From == nil {
		return fmt.Sprintf("%v", a.Data)
	}
	return fmt.Sprintf("%v:(%v)", a.Data, a.From)
}
