package num

import (
	"fmt"
	"math"
)

type Nodes []*Data

func New(size int) *Data {
	return &Data{
		Data: make(Float64s, size),
		grad: make(Float64s, size),
	}
}

func Wrap(data, grad Float64s) *Data {
	return &Data{
		Data: data,
		grad: grad,
	}
}

type Data struct {
	Data Float64s
	grad Float64s

	srcNodes Nodes
	calcGrad func()
}

func (d *Data) AddTo(out *Data, b *Data) {
	out.Data.CopyFrom(d.Data)
	out.Data.RepeatAdd(b.Data)

	out.srcNodes = Nodes{d, b}
	out.calcGrad = func() {
		d.grad.RepeatAdd(out.grad)
		b.grad.RepeatAdd(out.grad)
	}
}

func (d *Data) MulScalar(k float64) (out *Data) {
	out = &Data{
		Data:     d.Data.Copy(),
		grad:     d.grad.CopyZero(),
		srcNodes: Nodes{d},
	}

	out.Data.MulScalar(k)

	out.calcGrad = func() {
		oGrad := out.grad.Copy()
		oGrad.MulScalar(k)
		d.grad.Add(oGrad)
	}

	return out
}

func (d *Data) DotTo(out *Data, b *Data, batchSize int) {
	iSize := len(d.Data) / batchSize
	oSize := len(out.Data) / batchSize

	for i := 0; i < batchSize; i++ {
		inputs := d.Data[i*iSize : (i+1)*iSize]
		output := out.Data[i*oSize : (i+1)*oSize]

		for o := 0; o < oSize; o++ {
			weights := b.Data[o*iSize : (o+1)*iSize]
			output[o] = Dot(inputs, weights)
		}
	}

	out.srcNodes = Nodes{d, b}
	out.calcGrad = func() {
		for i := 0; i < batchSize; i++ {
			inputs := d.Data[i*iSize : (i+1)*iSize]
			iGrads := d.grad[i*iSize : (i+1)*iSize]
			oGrads := out.grad[i*oSize : (i+1)*oSize]

			for o, delta := range oGrads {
				weights := b.Data[o*iSize : (o+1)*iSize]
				iGrads.AddWeighted(weights, delta)

				wGrads := b.grad[o*iSize : (o+1)*iSize]
				wGrads.AddWeighted(inputs, delta)
			}
		}
	}
}

func (d *Data) Softmax(bSize int) (out *Data) {
	out = &Data{
		Data:     d.Data.Copy(),
		grad:     d.grad.CopyZero(),
		srcNodes: Nodes{d},
	}

	softmax := out.Data
	chunkSize := len(softmax) / bSize

	for i := 0; i < len(softmax); i += chunkSize {
		softmax[i : i+chunkSize].Softmax()
	}

	out.calcGrad = func() {
		for b := 0; b < len(softmax); b += chunkSize {
			oGrad := out.grad[b : b+chunkSize]
			dGrad := d.grad[b : b+chunkSize]
			softmax := softmax[b : b+chunkSize]

			for i := 0; i < len(oGrad); i++ {
				g := oGrad[i]
				for j := 0; j < len(oGrad); j++ {
					if i == j {
						dGrad[j] += g * softmax[i] * (1 - softmax[i])
					} else {
						dGrad[j] += -g * softmax[i] * softmax[j]
					}
				}
			}
		}
	}
	return out
}

func (d *Data) ReLuTo(out *Data) {
	for i, v := range d.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = 0
		}
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.Data {
			if v > 0 {
				d.grad[i] += out.grad[i]
			}
		}
	}
}

func (d *Data) SigmoidTo(out *Data) {
	for i, v := range d.Data {
		out.Data[i] = 1.0 / (1.0 + math.Exp(-v))
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.Data {
			d.grad[i] += out.grad[i] * v * (1 - v)
		}
	}
}

func (d *Data) TanhTo(out *Data) {
	for i, v := range d.Data {
		out.Data[i] = math.Tanh(v)
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for i, v := range out.Data {
			d.grad[i] += out.grad[i] * (1 - v*v)
		}
	}
}

func (d *Data) GetEmbeddedTo(out *Data, featuresCount int, idx []int) {
	for i, pos := range idx {
		features := d.Data[pos*featuresCount : (pos+1)*featuresCount]
		outBuffer := out.Data[i*featuresCount : (i+1)*featuresCount]

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

func (d *Data) CrossEntropy(target Float64s, bSize int) (out *Data) {
	softmax := d.Data.Copy()
	chunkSize := len(softmax) / bSize

	for i := 0; i < len(softmax); i += chunkSize {
		softmax[i : i+chunkSize].Softmax()
	}

	actual := softmax.Copy()
	for i, t := range target {
		actual[i] = -t * math.Log(actual[i])
	}

	loss := actual.Sum() / float64(bSize)

	out = New(1)
	out.Data[0] = loss
	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		k := 1.0 / float64(bSize)
		for i, t := range target {
			d.grad[i] = k * (softmax[i] - t)
		}
	}

	return
}

func (d *Data) Regression(target Float64s, bSize int) (out *Data) {
	loss := 0.0
	for i, t := range target {
		loss += math.Pow(d.Data[i]-t, 2)
	}
	loss = 0.5 * loss / float64(bSize)

	out = New(1)
	out.Data[0] = loss
	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		oGrads := d.Data.Copy()
		oGrads.AddWeighted(target, -1.0)
		oGrads.MulScalar(1.0 / float64(bSize))

		d.grad.Add(oGrads)
	}

	return
}

func (d *Data) Tril(bSize int, zeroVal float64) (out *Data) {
	iSize := d.Len() / bSize
	sSize := int(math.Sqrt(float64(iSize)))

	out = New(d.Len())
	out.Data.Fill(zeroVal)

	for b := 0; b < bSize; b++ {
		// Data - square matrix
		data := out.Data[b*iSize : (b+1)*iSize]
		dData := d.Data[b*iSize : (b+1)*iSize]
		for r := 0; r < sSize; r++ {
			for c := 0; c <= r; c++ {
				data[r*sSize+c] = dData[r*sSize+c]
			}
		}
	}

	out.srcNodes = Nodes{d}
	out.calcGrad = func() {
		for b := 0; b < bSize; b++ {
			// Data - square matrix
			grad := d.grad[b*iSize : (b+1)*iSize]
			oGrad := out.grad[b*iSize : (b+1)*iSize]
			for r := 0; r < sSize; r++ {
				for c := 0; c <= r; c++ {
					grad[r*sSize+c] = oGrad[r*sSize+c]
				}
			}
		}
	}

	return out
}

func (d *Data) Print(bSize int) {
	iSize := d.Len() / bSize
	sSize := int(math.Sqrt(float64(iSize)))

	for b := 0; b < bSize; b++ {
		// Data - square matrix
		data := d.Data[b*iSize : (b+1)*iSize]
		for r := 0; r < sSize; r++ {
			for c := 0; c < sSize; c++ {
				fmt.Printf("%.4f ", data[r*sSize+c])
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

//
//type Dims struct {
//	W, H, D int
//}
//
//func (d *Data) MatrixMultiply(
//	b *Data,
//	srcColumns int,
//	batchSize int,
//) (out *Data) {
//	r := NewVolume(b.W, obj.H, obj.D)
//
//	bT := b.Transpose()
//	bT.ScanRows(func(weightIndex, _ int, bFloats []float64) {
//		obj.ScanRows(func(inputIndex, z int, aFloats []float64) {
//			r.Set(weightIndex, inputIndex, z, Dot(aFloats, bFloats))
//		})
//	})
//}

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

func (d *Data) Len() int {
	return len(d.Data)
}

func (d *Data) GetData() Float64s {
	return d.Data
}

func (d *Data) ForUpdate() [2]Float64s {
	return [2]Float64s{d.Data, d.grad}
}
