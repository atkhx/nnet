package data

import (
	"fmt"
	"strings"
)

type Operand interface {
	backward()
	resetGrad()
}

type Operands []Operand

func NewSource(callback func(), operands ...Operand) *Source {
	return &Source{
		Callback: callback,
		Parents:  operands,
	}
}

type Source struct {
	Parents  Operands
	Callback func()
}

func (s *Source) String() string {
	if s == nil {
		return ""
	}

	return fmt.Sprintf("%v", s.Parents)
}

func (s Operands) String() string {
	if len(s) == 0 {
		return ""
	}

	lines := []string{}
	for _, v := range s {
		lines = append(lines, fmt.Sprintf("%v", v))
	}

	return strings.Join(lines, ", ")
}
