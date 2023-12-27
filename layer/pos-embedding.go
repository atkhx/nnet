package layer

import (
	"github.com/atkhx/nnet"
	"github.com/atkhx/nnet/num"
)

func NewAddRow(
	rowData *num.Data,
) *AddRow {
	return &AddRow{
		rowData: rowData,
	}
}

type AddRow struct {
	rowData *num.Data
}

func (l *AddRow) Compile(device nnet.Device, inputs *num.Data) *num.Data {
	return device.AddRow(
		inputs,
		l.rowData,
		l.rowData.Dims.Size(),
	)
}
