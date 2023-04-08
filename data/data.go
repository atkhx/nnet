package data

import (
	"fmt"
	"math"
)

func NewRandom(colsCount, rowsCount, chanCount int) *Data {
	return WrapData(colsCount, rowsCount, chanCount, MakeRandom(colsCount*rowsCount*chanCount))
}

func WrapData(w, h, d int, data []float64) *Data {
	if len(data) != w*h*d {
		panic(fmt.Sprintf("invalid dimensions: data length %d, w*h*d %d", len(data), w*h*d))
	}

	return &Data{
		Data: WrapVolume(w, h, d, data),
		Grad: NewVolume(w, h, d),
	}
}

func NewData(w, h, d int) *Data {
	return &Data{
		Data: NewVolume(w, h, d),
		Grad: NewVolume(w, h, d),
	}
}

type Data struct {
	Data *Volume
	Grad *Volume

	parents []*Data

	title          string
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
		//return m.generate(m.Data.Relu(), func() {
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
		k := 1.0 / float64(m.Data.Len())
		m.Grad.AddScalar(outMatrix.Grad.Data[0] * k)
	})
}

func (m *Data) Std() (outMatrix *Data) {
	return m.generate(m.Data.Std(), func() {
		g := outMatrix.Grad.Data[0]
		k := 1.0 / float64(m.Data.Len()-1)

		out := outMatrix.Data.Data[0]
		mean := m.Data.Mean().Data[0]

		for offset := range m.Grad.Data {
			m.Grad.Data[offset] += g * (1.0 / out) * (m.Data.Data[offset] - mean) * k
		}
	})
}

func (m *Data) ColMean() (outMatrix *Data) {
	k := 1.0 / float64(m.Data.H)

	out := NewVolume(m.Data.W, 1, m.Data.D)
	m.Data.Scan(func(x, y, z int, offset int, v float64) {
		out.PointAdd(x, 0, z, k*v)
	})

	return m.generate(out, func() {
		m.Grad.Scan(func(x, y, z int, offset int, v float64) {
			m.Grad.Data[offset] += outMatrix.Grad.At(x, 0, z) * k
		})
	})
}

func (m *Data) RowMean() (outMatrix *Data) {
	k := 1.0 / float64(m.Data.W)

	out := NewVolume(1, m.Data.H, m.Data.D)
	m.Data.Scan(func(x, y, z int, offset int, v float64) {
		out.PointAdd(0, y, z, k*v)
	})

	return m.generate(out, func() {
		m.Grad.Scan(func(x, y, z int, offset int, v float64) {
			m.Grad.Data[offset] += outMatrix.Grad.At(0, y, z) * k
		})
	})
}

func (m *Data) RowVariance() (outMatrix *Data) {
	rowMean := m.RowMean()
	rowVars := NewVolume(1, rowMean.Data.H, rowMean.Data.D)

	k := 1.0 / float64(m.Data.W-1)

	m.Data.Scan(func(x, y, z int, offset int, v float64) {
		rowVars.PointAdd(0, y, z, math.Pow(v-rowMean.Data.At(0, y, z), 2)*k)
	})

	return m.generate(rowVars, func() {
		m.Data.Scan(func(x, y, z int, offset int, v float64) {
			G := outMatrix.Grad.At(0, y, z)
			M := rowMean.Data.At(0, y, z)

			m.Grad.Data[offset] += G * 2.0 * (v - M) * k
		})
	})
}

func (m *Data) ColStd() (outMatrix *Data) {
	colMean := m.ColMean()
	colVariance := NewVolume(colMean.Data.W, 1, colMean.Data.D)
	colStdDev := NewVolume(colMean.Data.W, 1, colMean.Data.D)

	k := 1.0 / float64(m.Data.H-1)
	e := 0.000000001

	m.Data.Scan(func(x, y, z int, offset int, v float64) {
		colVariance.PointAdd(x, 0, z, math.Pow(v-colMean.Data.At(x, 0, z), 2)*k)
	})

	colVariance.Scan(func(_, _, _ int, offset int, v float64) {
		colStdDev.Data[offset] = math.Sqrt(v + e)
	})

	return m.generate(colStdDev, func() {
		colStdDevGrad := NewVolume(colMean.Data.W, 1, colMean.Data.D)
		outMatrix.Data.Scan(func(x, y, z int, offset int, v float64) {
			g := outMatrix.Grad.Data[offset]
			o := outMatrix.Data.Data[offset]

			colStdDevGrad.Data[offset] = g * (1.0 / (2.0 * o))
		})

		m.Data.Scan(func(x, y, z int, offset int, v float64) {
			g := colStdDevGrad.At(x, 0, z)
			M := colMean.Data.At(x, 0, z)

			m.Grad.Data[offset] += g * 2.0 * (v - M)
		})
	})
}

func (m *Data) Pow(pow float64) (outMatrix *Data) {
	out := m.Data.Copy()
	out.Scan(func(x, y, z int, offset int, v float64) {
		out.Data[offset] = math.Pow(v, pow)
	})

	return m.generate(out, func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * pow * math.Pow(m.Data.Data[i], pow-1)
		}
	})
}

func (m *Data) Sqrt() (outMatrix *Data) {
	out := m.Data.Copy()
	out.Scan(func(_, _, _ int, offset int, v float64) {
		out.Data[offset] = math.Sqrt(v)
	})

	return m.generate(out, func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * 0.5 * math.Pow(m.Data.Data[i], -0.5)
		}
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

func (m *Data) MulScalar(f float64) (outMatrix *Data) {
	return m.generate(m.Data.Copy().MulScalar(f), func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] * f
		}
	})
}

func (m *Data) DivScalar(f float64) (outMatrix *Data) {
	return m.generate(m.Data.Copy().DivScalar(f), func() {
		for i := range m.Grad.Data {
			m.Grad.Data[i] += outMatrix.Grad.Data[i] / f
		}
	})
}

func (m *Data) AddColVector(b *Data) (outMatrix *Data) {
	if b.Data.W > 1 {
		panic(fmt.Sprintf("bColsCount > 1: %d", b.Data.W))
	}

	out := m.Data.Copy()
	out.Scan(func(x, y, z int, offset int, v float64) {
		out.Data[offset] += b.Data.At(0, y, z)
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
		outMatrix.Grad.Scan(func(x, y, z int, offset int, v float64) {
			b.Grad.PointAdd(0, y, z, v)
		})
	}, b)
}

func (m *Data) AddRowVector(b *Data) (outMatrix *Data) {
	if b.Data.H > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Data.H))
	}

	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Add(b.Data.GetRows(z))
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
		outMatrix.Grad.ScanRowsVolume(func(y, z int, f *Volume) {
			b.Grad.GetRows(z).Add(f)
		})
	}, b)
}

func (m *Data) SubColVector(b *Data) (outMatrix *Data) {
	if b.Data.W > 1 {
		panic(fmt.Sprintf("bColsCount > 1: %d", b.Data.W))
	}

	out := m.Data.Copy()
	out.Scan(func(x, y, z int, offset int, v float64) {
		out.Data[offset] -= b.Data.At(0, y, z)
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
		outMatrix.Grad.Scan(func(x, y, z int, offset int, v float64) {
			b.Grad.PointAdd(0, y, z, -v)
		})
	}, b)
}

func (m *Data) SubRowVector(b *Data) (outMatrix *Data) {
	if b.Data.H > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Data.H))
	}

	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Sub(b.Data.GetRows(z))
	})

	return m.generate(out, func() {
		m.Grad.Add(outMatrix.Grad)
		outMatrix.Grad.ScanRowsVolume(func(y, z int, f *Volume) {
			b.Grad.GetRows(z).Sub(f)
		})
	}, b)
}

func (m *Data) MulColVector(b *Data) (outMatrix *Data) {
	if b.Data.W > 1 {
		panic(fmt.Sprintf("bColsCount > 1: %d", b.Data.W))
	}

	out := m.Data.Copy()
	out.Scan(func(x, y, z int, offset int, v float64) {
		out.Data[offset] *= b.Data.At(0, y, z)
	})

	return m.generate(out, func() {
		m.Data.Scan(func(x, y, z int, offset int, O float64) {
			G := outMatrix.Grad.Data[offset]

			m.Grad.Data[offset] += G * b.Data.At(0, y, z)
			b.Grad.PointAdd(0, y, z, G*O)
		})
	}, b)
}

func (m *Data) MulRowVector(b *Data) (outMatrix *Data) {
	if b.Data.H > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Data.H))
	}

	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Mul(b.Data.GetRows(z))
	})

	return m.generate(out, func() {
		bDataRow := b.Data.Data

		m.Grad.ScanRows(func(y, z int, iGradRow []float64) {
			bGradRow := b.Grad.GetRows(z).Data
			offset := z*m.Data.H*m.Data.W + y*m.Data.W

			oGradRow := outMatrix.Grad.Data[offset : offset+m.Data.W]
			iDataRow := m.Data.Data[offset : offset+m.Data.W]

			for x, g := range oGradRow {
				iGradRow[x] += g * bDataRow[x]
				bGradRow[x] += g * iDataRow[x]
			}
		})
	}, b)
}

func (m *Data) DivColVector(b *Data) (outMatrix *Data) {
	if b.Data.W > 1 {
		panic(fmt.Sprintf("bColsCount > 1: %d", b.Data.W))
	}

	out := m.Data.Copy()
	out.Scan(func(x, y, z int, offset int, v float64) {
		out.Data[offset] /= b.Data.At(0, y, z)
	})

	return m.generate(out, func() {
		outMatrix.Grad.Scan(func(x, y, z int, offset int, g float64) {
			bValue := b.Data.At(0, y, z)
			bSquare := bValue * bValue
			if bSquare == 0 {
				bSquare = 0.0000000001
			}

			m.Grad.Data[offset] += g / b.Data.At(0, y, z)
			b.Grad.PointAdd(0, y, z, g*m.Data.Data[offset]*(-1.0/(bValue*bValue)))
		})
	}, b)
}

func (m *Data) DivRowVector(b *Data) (outMatrix *Data) {
	if b.Data.H > 1 {
		panic(fmt.Sprintf("bRowsCount > 1: %d", b.Data.H))
	}

	out := m.Data.Copy()
	out.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Div(b.Data.GetRows(z))
	})

	return m.generate(out, func() {
		bDataRow := b.Data.Data
		//m.Grad.Fill(0.0)
		m.Grad.ScanRows(func(y, z int, iGradRow []float64) {
			bGradRow := b.Grad.GetRows(z).Data
			offset := z*m.Data.H*m.Data.W + y*m.Data.W
			oGradRow := outMatrix.Grad.Data[offset : offset+m.Data.W]
			iDataRow := m.Data.Data[offset : offset+m.Data.W]

			for x, g := range oGradRow {
				iGradRow[x] += g / bDataRow[x]
				bGradRow[x] += g * iDataRow[x] * (-1.0 / (bDataRow[x] * bDataRow[x]))
			}
		})
	}, b)
}

func (m *Data) MatrixMultiply(b *Data) (outMatrix *Data) {
	return m.generate(m.Data.MatrixMultiply(b.Data), func() {
		oG := WrapData(
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
	imageWidth, imageHeight, channels,
	filterSize, padding, stride int,
	filters *Data,
	biases *Data,
) (outMatrix *Data) {
	imagesCount := m.Data.D
	filtersCount := filters.Data.D

	outImageWidth, outImageHeight := CalcConvOutputSize(
		imageWidth, imageHeight,
		filterSize, filterSize,
		padding, stride,
	)

	outputSquare := outImageWidth * outImageHeight
	outputData := make([]float64, outputSquare*imagesCount*filtersCount)

	return m.ConvTo(
		outImageWidth, outImageHeight, outputData,
		imageWidth, imageHeight, channels, filterSize, padding, stride,
		filters,
		biases,
	)
}

func (m *Data) ConvTo(
	outImageWidth, outImageHeight int, outputData []float64,
	imageWidth, imageHeight, channels, filterSize, padding, stride int,
	filters *Data,
	biases *Data,
) (outMatrix *Data) {
	imagesCount := m.Data.D
	filtersCount := filters.Data.D
	outputSquare := outImageWidth * outImageHeight

	offset := 0
	for imageIndex := 0; imageIndex < imagesCount; imageIndex++ {
		image := m.Data.GetRows(imageIndex)

		for filterIndex := 0; filterIndex < filtersCount; filterIndex++ {
			featureMap := outputData[offset : offset+outputSquare]
			// Fill(featureMap, biases.Data.Data[filterIndex])
			ConvTo(
				outImageWidth, outImageHeight, featureMap,
				imageWidth, imageHeight, image.Data,
				filterSize, filterSize, filters.Data.GetRows(filterIndex).Data,
				channels,
				padding,
			)
			offset += outputSquare
		}
	}

	return m.generate(WrapVolume(outputSquare, filtersCount, imagesCount, outputData), func() {
		inputs := m

		outputGrad := outMatrix.Grad

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
				// biases.Grad.Data[filterIndex] += deltas.Sum().Data[0]

				filtersGrad := filters.Grad.GetRows(filterIndex).Data
				filtersRotGrad := filtersRot.GetRows(filterIndex).Data

				// fWH[D] = iWH[D] x Dy[1]
				// dW     = I      x Dy
				Convolve2dBatchTo(
					filterSize, filterSize, filtersGrad,
					imageWidth, imageHeight, channels, inputs.Data,
					outImageWidth, outImageHeight, 1, deltas.Data,
					padding,
				)

				// iWH[D] = oWoH[1] x fWH[D]
				// dI     = DyPad   x Wrot180
				Convolve2dBatchTo(
					imageWidth, imageHeight, iGrads.Data,
					deltas.W, deltas.H, 1, deltas.Data,
					filterSize, filterSize, channels, filtersRotGrad,
					padding,
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
		out.ScanRowsVolume(func(y, z int, f *Volume) {
			for i := 0; i < f.Len(); i++ {
				g := outMatrix.Grad.At(i, y, z)
				for j := 0; j < f.Len(); j++ {
					if i == j {
						m.Grad.PointAdd(j, y, z, g*f.Data[i]*(1-f.Data[i]))
					} else {
						m.Grad.PointAdd(j, y, z, -g*f.Data[i]*f.Data[j])
					}
				}
			}
		})
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
			out.PointAdd(0, y, z, -t*math.Log(m.Data.At(i, y, z)))
		}
	})

	return m.generate(out, func() {
		outMatrix.Grad.ScanRows(func(y, z int, f []float64) {
			for x := 0; x < m.Data.W; x++ {
				t := targets.Data.At(x, y, z)
				o := m.Data.At(x, y, z)

				m.Grad.PointAdd(x, y, z, -f[0]*t/o)
			}
		})
	})
}

func (m *Data) CrossEntropy(targets *Data) (outMatrix *Data) {
	if !m.Data.IsDimensionsEqual(targets) {
		panic(fmt.Sprintf(
			"invalid targets dimensions: expected %v, actual %v",
			m.GetDims(),
			targets.GetDims(),
		))
	}

	softmax := m.Data.Copy()
	softmax.ScanRowsVolume(func(y, z int, f *Volume) {
		f.Softmax()
	})

	//fmt.Println("softmax", softmax.Data)
	logLikelihood := NewVolume(1, m.Data.H, m.Data.D)

	targets.Data.ScanRows(func(y, z int, row []float64) {
		for i, t := range row {
			logLikelihood.PointAdd(0, y, z, -t*math.Log(softmax.At(i, y, z)))
		}
	})
	//fmt.Println("logLikelihood", logLikelihood.Data)

	return m.generate(logLikelihood, func() {
		outMatrix.Grad.ScanRows(func(y, z int, f []float64) {
			for x := 0; x < m.Data.W; x++ {
				m.Grad.PointAdd(x, y, z, f[0]*(softmax.At(x, y, z)-targets.Data.At(x, y, z)))
			}
		})
	})
}

func (m *Data) Backward() {
	var collect func(data *Data)

	visited := map[*Data]bool{}
	parents := []*Data{}

	collect = func(data *Data) {
		if _, ok := visited[data]; ok {
			return
		}

		visited[data] = true
		for _, c := range data.parents {
			collect(c)
		}

		// reset grads
		if data.Grad == nil {
			data.Grad = NewVolume(data.Data.W, data.Data.H, data.Data.D)
		} else {
			data.Grad.Fill(0)
		}

		parents = append(parents, data)
	}

	collect(m)
	m.Grad.Fill(1)

	for i := len(parents); i > 0; i-- {
		if parents[i-1].backwardFn != nil {
			parents[i-1].backwardFn()
		}
	}
}
