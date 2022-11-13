package data

import (
	"fmt"

	"github.com/pkg/errors"
)

var (
	ErrorVectorIndexToLow  = errors.New("vector index to low")
	ErrorVectorCountToLow  = errors.New("vector count to low")
	ErrorVectorIndexToHigh = errors.New("vector index to high")
)

func NewOneHotVector(index, count int) (*Data, error) {
	if index < 0 {
		return nil, errors.Wrap(ErrorVectorIndexToLow, fmt.Sprintf("%d less than zero", index))
	}
	if count < 1 {
		return nil, errors.Wrap(ErrorVectorCountToLow, fmt.Sprintf("%d less than one", count))
	}
	if index >= count {
		return nil, errors.Wrap(ErrorVectorIndexToHigh, fmt.Sprintf("index: %d, count: %d", index, count))
	}

	res := NewVector(count)
	res.Data[index] = 1
	return res, nil
}

func NewOneHotVectors(count int) ([]*Data, error) {
	if count < 1 {
		return nil, errors.Wrap(ErrorVectorCountToLow, fmt.Sprintf("%d less than one", count))
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
