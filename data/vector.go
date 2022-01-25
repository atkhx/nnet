package data

func NewVector(data ...float64) *Data {
	res := new(Data)
	res.InitVector(len(data))

	copy(res.Data, data)
	return res
}

func NewVectors(data ...[]float64) []*Data {
	var res []*Data
	for _, v := range data {
		res = append(res, NewVector(v...))
	}
	return res
}
