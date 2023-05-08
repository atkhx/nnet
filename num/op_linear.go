package num

func (input *Data) Linear(weights *Data, outputDims Dims) *Data {
	output := New(outputDims, input, weights)
	//
	//weightsLength := len(weights.Data)
	//
	//output.calcData = func() {
	//	for o := range output.Data {
	//		v := 0.0
	//
	//		output.Data[o] = v
	//	}
	//}

	return output
}
