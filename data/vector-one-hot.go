package data

import (
	"github.com/pkg/errors"
)

var (
	ErrorVectorIndexToLow  = errors.New("vector index to low")
	ErrorVectorCountToLow  = errors.New("vector count to low")
	ErrorVectorIndexToHigh = errors.New("vector index to high")
)

func NewOneHotVector(index, count int) (*Data, error) {
	if index < 0 {
		return nil, ErrorVectorIndexToLow
	}
	if count < 1 {
		return nil, ErrorVectorCountToLow
	}
	if index >= count {
		return nil, ErrorVectorIndexToHigh
	}

	res := &Data{}
	res.InitVector(count)
	res.Data[index] = 1
	return res, nil
}

func NewOneHotVectors(count int) ([]*Data, error) {
	if count < 1 {
		return nil, ErrorVectorCountToLow
	}

	res := make([]*Data, count)
	for i := 0; i < count; i++ {
		r, err := NewOneHotVector(i, count)
		if err != nil {
			return nil, err
		}
		res[i] = r
	}
	return res, nil
}

func MustCompileOneHotVectors(count int) []*Data {
	r, err := NewOneHotVectors(count)
	if err != nil {
		panic(err)
	}
	return r
}
