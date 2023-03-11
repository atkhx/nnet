package data

import (
	"fmt"
	"math"
)

func NewRandomMinMax(colsCount, rowsCount, chanCount int, min, max float64) *Data {
	return NewData(colsCount, rowsCount, chanCount, MakeRandomMinMax(colsCount*rowsCount*chanCount, min, max))
}

func NewRandom(colsCount, rowsCount, chanCount int) *Data {
	return NewData(colsCount, rowsCount, chanCount, MakeRandom(colsCount*rowsCount*chanCount))
}

func NewData(w, h, d int, data []float64) *Data {
	if len(data) != w*h*d {
		panic(fmt.Sprintf("invalid dimensions: data length %d, w*h*d %d", len(data), w*h*d))
	}

	return &Data{
		Data: WrapVolume(w, h, d, data),
		Grad: NewVolume(w, h, d),
	}
}

type Data struct {
	Data *Volume
	Grad *Volume

	parents []*Data

	backwardFn     func()
	backwardCalled bool
}

func (m *Data) SetParentsAndBackwardFn(parents []*Data, backwardFn func()) {
	m.parents = parents
	m.backwardFn = backwardFn
}

func (m *Data) Generate(data *Volume, backwardFn func(), parents ...*Data) (outMatrix *Data) {
	return m.generate(data, backwardFn, parents...)
}

func (m *Data) generate(data *Volume, backwardFn func(), parents ...*Data) (outMatrix *Data) {
	return &Data{Data: data, backwardFn: backwardFn, parents: append([]*Data{m}, parents...)}
}

func (m *Data) GetDims() []int {
	return m.Data.GetDims()
}

func (m *Data) Transpose() (outMatrix *Data) {
	return m.generate(m.Data.Transpose(), func() {
		m.Grad.Add(outMatrix.Grad.Transpose())
	})
}

func (m *Data) Flat() (outMatrix *Data) {
	return m.generate(m.Data.Reshape(m.Data.Len(), 1, 1), func() {
		m.Grad.Add(outMatrix.Grad)
	})
}

func (m *Data) Tanh() (outMatrix *Data) {
	return m.generate(m.Data.Copy().Tanh(), func() {
		for i, v := range outMatrix.Data.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * (1 - v*v)
		}
	})
}

func (m *Data) Sigmoid() (outMatrix *Data) {
	return m.generate(m.Data.Copy().Sigmoid(), func() {
		for i, v := range outMatrix.Data.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * v * (1 - v)
		}
	})
}

func (m *Data) Relu() (outMatrix *Data) {
	return m.generate(m.Data.Copy().Relu(), func() {
		for i, v := range outMatrix.Data.Data {
			if v > 0 {
				m.Grad.Data[i] += outMatrix.Grad.Data[i]
			}
		}
	})
}

func (m *Data) Sum() (outMatrix *Data) {
	return m.generate(m.Data.Sum(), func() {
		m.Grad.AddScalar(outMatrix.Grad.Data[0])
	})
}

func (m *Data) Mean() (outMatrix *Data) {
	return m.generate(m.Data.Mean(), func() {
		m.Grad.AddScalar(outMatrix.Grad.Data[0])
	})
}

func (m *Data) Exp() (outMatrix *Data) {
	return m.generate(m.Data.Copy().Exp(), func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * outMatrix.Data.Data[i]
		}
	})
}

func (m *Data) Log() (outMatrix *Data) {
	return m.generate(m.Data.Copy().Log(), func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] / m.Data.Data[i]
		}
	})
}

func (m *Data) Mul(f float64) (outMatrix *Data) {
	return m.generate(m.Data.Copy().Mul(f), func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * f
		}
	})
}

func (m *Data) AddRowVector(b *Data) (outMatrix *Data) {
	if b.Data.H > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Data.H))
	}

	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Add(b.Data)
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
		outMatrix.Grad.ScanRowsVolume(func(y, z int, f *Volume) {
			b.Grad.Add(f)
		})
	}, b)
}

func (m *Data) MatrixMultiply(b *Data) (outMatrix *Data) {
	return m.generate(m.Data.MatrixMultiply(b.Data), func() {
		oG := NewData(
			outMatrix.Data.W,
			outMatrix.Data.H,
			outMatrix.Data.D,
			outMatrix.Grad.Data,
		)

		m.Grad.Add(oG.MatrixMultiply(b.Transpose()).Data)
		b.Grad.Add(m.Transpose().MatrixMultiply(oG).Data)
	}, b)
}

func (m *Data) Conv(
	imageWidth, imageHeight, filterSize, padding, stride int,
	filters *Data,
	biases *Data,
) (outMatrix *Data) {
	imagesCount := m.Data.D
	filtersCount := filters.Data.D
	channels := m.Data.H

	outImageWidth, outImageHeight := CalcConvOutputSize(
		imageWidth, imageHeight,
		filterSize, filterSize,
		padding, stride,
	)
	outputSquare := outImageWidth * outImageHeight

	outputData := make([]float64, outputSquare*imagesCount*filtersCount)

	offset := 0
	for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
		image := m.Data.GetRows(imageIndex)

		for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
			filter := filters.Data.GetRows(filterIndex)
			featureMap := outputData[offset : offset+outputSquare]

			ConvTo(
				imageWidth, imageHeight, image.Data,
				filterSize, filterSize, filter.Data,
				outImageWidth, outImageHeight, featureMap,
				channels,
				padding,
			)

			AddScalarTo(featureMap, biases.Data.Data[filterIndex])
			offset += outputSquare
		}
	}

	return m.generate(WrapVolume(outputSquare, filtersCount, imagesCount, outputData), func() {
		inputs := m

		outputGrad := outMatrix.Grad
		outputGradPad := WrapVolume(AddPadding(outputGrad.Data, outImageWidth, outImageHeight, filtersCount*imagesCount, 2-padding))

		filtersRot := filters.Data.
			Reshape(filterSize, filterSize, filtersCount*channels).
			Rotate180().
			Reshape(filters.Data.W, filters.Data.H, filters.Data.D)

		outputGrad = outputGrad.Reshape(outImageWidth, outImageHeight, filtersCount*imagesCount)

		for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
			iGrads := inputs.Grad.GetRows(imageIndex)
			inputs := inputs.Data.GetRows(imageIndex)

			for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
				deltas := outputGrad.GetRows(imageIndex*filtersCount + filterIndex)
				wGrads := filters.Grad.GetRows(filterIndex)

				biases.Grad.Data[filterIndex] += deltas.Sum().Data[0]

				// dW = I x Dy
				ConvLayersTo(
					filterSize, filterSize, wGrads.Data,
					imageWidth, imageHeight, channels, inputs.Data,
					outImageWidth, outImageHeight, 1, deltas.Data,
					0,
				)

				deltasPad := outputGradPad.GetRows(imageIndex*filtersCount + filterIndex)
				filterRot := filtersRot.GetRows(filterIndex)

				// dI = DyPad x Wrot180
				ConvLayersTo(
					imageWidth, imageHeight, iGrads.Data,
					deltasPad.W, deltasPad.H, 1, deltasPad.Data,
					filterSize, filterSize, channels, filterRot.Data,
					0,
				)
			}
		}

	}, m, filters, biases)
}

func (m *Data) Softmax() (outMatrix *Data) {
	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Softmax()
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
	})
}

func (m *Data) Regression(targets *Data) (outMatrix *Data) {
	if !m.Data.IsDimensionsEqual(targets) {
		panic(fmt.Sprintf(
			"invalid targets dimensions: expected %v, actual %v",
			m.GetDims(),
			targets.GetDims(),
		))
	}

	r := 0.0
	for i, t := range targets.Data.Data {
		r += math.Pow(m.Data.Data[i]-t, 2)
	}
	r *= 0.5

	return m.generate(WrapVolume(1, 1, 1, []float64{r}), func() {
		for i, t := range targets.Data.Data {
			m.Grad.Data[i] += m.Data.Data[i] - t
		}
	})
}

const minimalNonZeroFloat = 0.000000000000000000001

func (m *Data) Classification(targets *Data) (outMatrix *Data) {
	if !m.Data.IsDimensionsEqual(targets) {
		panic(fmt.Sprintf(
			"invalid targets dimensions: expected %v, actual %v",
			m.GetDims(),
			targets.GetDims(),
		))
	}

	out := NewVolume(1, m.Data.H, m.Data.D)
	targets.Data.ScanRows(func(y, z int, row []float64) {
		for i, t := range row {
			if t == 1 {
				o := m.Data.At(i, y, z)
				if o <= 0 {
					o = minimalNonZeroFloat
				}

				out.Set(0, y, z, -math.Log(o))
				break
			}
		}
	})

	return m.generate(out, func() {
		outMatrix.Grad.ScanRows(func(y, z int, f []float64) {
			for x := 0; x < m.Data.W; x++ {
				m.Grad.Set(x, y, z, f[0]*(m.Data.At(x, y, z)-targets.Data.At(x, y, z)))
			}
		})
	})
}

func (m *Data) ResetGrad() {
	m.resetGrad()
}

func (m *Data) resetGrad() {
	if m.Grad == nil {
		m.Grad = NewVolume(m.Data.W, m.Data.H, m.Data.D)
	}

	m.Grad.Fill(0)
	m.backwardCalled = false
	for _, prev := range m.parents {
		prev.resetGrad()
	}
}

func (m *Data) Backward() {
	m.Grad.Fill(1)
	m.backward()
}

func (m *Data) backward() {
	if m.backwardCalled {
		return
	}

	m.backwardCalled = true
	if m.backwardFn != nil {
		m.backwardFn()
	}

	for _, prev := range m.parents {
		prev.backward()
	}
}
