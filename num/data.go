package num

type Nodes []*Data

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

func (d *Data) CalcGrad() {
	if d.calcGrad != nil {
		d.calcGrad()
	}

	for _, node := range d.srcNodes {
		node.CalcGrad()
	}
}
