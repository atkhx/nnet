package num

import "math"

type Nodes []*Data

func New(size int) *Data {
	return &Data{
		data: make(Float64s, size),
		grad: make(Float64s, size),
	}
}

func Wrap(data, grad Float64s) *Data {
	return &Data{
		data: data,
		grad: grad,
	}
}

type Data struct {
	data Float64s
	grad Float64s

	srcNodes Nodes
	calcGrad func()
}

func (d *Data) AddTo(out *Data, b *Data) {
	out.data.CopyFrom(d.data)
	out.data.RepeatAdd(b.data)

	out.srcNodes = Nodes{d, b}
	out.calcGrad = func() {
		d.grad.RepeatAdd(out.grad)
		b.grad.RepeatAdd(out.grad)
	}
}

func (d *Data) DotTo(out *Data, b *Data, batchSize int) {
	iSize := len(d.data) / batchSize
	oSize := len(out.data) / batchSize

	for i := 0; i < batchSize; i++ {
		inputs := d.data[i*iSize : (i+1)*iSize]
		output := out.data[i*oSize : (i+1)*oSize]

		for o := 0; o < oSize; o++ {
			weights := b.data[o*iSize : (o+1)*iSize]
			output[o] = Dot(inputs, weights)
		}
	}

	out.srcNodes = Nodes{d, b}
	out.calcGrad = func() {
		for i := 0; i < batchSize; i++ {
			inputs := d.data[i*iSize : (i+1)*iSize]
			iGrads := d.grad[i*iSize : (i+1)*iSize]
			oGrads := out.grad[i*oSize : (i+1)*oSize]

			for o, delta := range oGrads {
				weights := b.data[o*iSize : (o+1)*iSize]
				iGrads.AddWeighted(weights, delta)

				wGrads := b.grad[o*iSize : (o+1)*iSize]
				wGrads.AddWeighted(inputs, delta)
			}
		}
	}
}

func (d *Data) ReLuTo(out *Data) {
	for i, v := range d.data {
		if v > 0 {
			out.data[i] = v
		} else {
			out.data[i] = 0
		}
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.data {
			if v > 0 {
				d.grad[i] += out.grad[i]
			}
		}
	}
}

func (d *Data) SigmoidTo(out *Data) {
	for i, v := range d.data {
		out.data[i] = 1.0 / (1.0 + math.Exp(-v))
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.data {
			d.grad[i] += out.grad[i] * v * (1 - v)
		}
	}
}

func (d *Data) TanhTo(out *Data) {
	for i, v := range d.data {
		out.data[i] = math.Tanh(v)
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.data {
			d.grad[i] += out.grad[i] * (1 - v*v)
		}
	}
}

func (d *Data) GetEmbeddedTo(out *Data, featuresCount int, idx []int) {
	for i, pos := range idx {
		features := d.data[pos*featuresCount : (pos+1)*featuresCount]
		outBuffer := out.data[i*featuresCount : (i+1)*featuresCount]

		copy(outBuffer, features)
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, pos := range idx {
			wGrads := d.grad[pos*featuresCount : (pos+1)*featuresCount]
			wGrads.Add(out.grad[i*featuresCount : (i+1)*featuresCount])
		}
	}
}

func (d *Data) CalcGrad() {
	if d.calcGrad != nil {
		d.calcGrad()
	}

	for _, node := range d.srcNodes {
		node.CalcGrad()
	}
}

func (d *Data) ResetGrad() {
	d.grad.Fill(0)

	for _, node := range d.srcNodes {
		node.ResetGrad()
	}
}
