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

func (d *Data) CalcGrad() {
	if d.calcGrad != nil {
		d.calcGrad()
	}

	for _, node := range d.srcNodes {
		node.CalcGrad()
	}
}
