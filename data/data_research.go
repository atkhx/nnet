package data

import "math"

func (m *Data) ResearchStd() (outMatrix *Data) {
	mean := m.Data.Mean().Data[0]

	k := 1.0 / float64(m.Data.Len()-1)
	out := 0.0
	for _, v := range m.Data.Data {
		out += math.Pow(v-mean, 2)
	}
	out *= k

	return m.generate(WrapVolume(1, 1, 1, []float64{math.Sqrt(out)}), func() {
		g := outMatrix.Grad.Data[0]

		m.Grad.Scan(func(_, _, _ int, offset int, v float64) {
			m.Grad.Data[offset] += g * 0.5 * math.Pow(out, -0.5) * 2 * (m.Data.Data[offset] - mean) * k
		})
	})
}
