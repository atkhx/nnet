package num

func (input *Data) Mul(bData *Data) *Data {
	config := BroadCast(input, bData)
	output := New(config.oDims, input, bData)
	output.calcData = func() {
		config.BroadCast(func(ax, bx, offset int) {
			output.Data[offset] = input.Data[ax] * bData.Data[bx]
		})
	}
	output.calcGrad = func() {
		config.BroadCast(func(ax, bx, offset int) {
			input.Grad[ax] += output.Grad[offset] * bData.Data[bx]
			bData.Grad[bx] += output.Grad[offset] * input.Data[ax]
		})
	}
	return output
}
