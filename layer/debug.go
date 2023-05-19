package layer

import (
	"fmt"
	"os"

	"github.com/atkhx/nnet/num"
)

func NewDebug(stopCnt int) *Debug {
	return &Debug{stopCnt: stopCnt}
}

type Debug struct {
	outputObj *num.Data
	counter   int
	stopCnt   int
}

func (l *Debug) Compile(inputs *num.Data) *num.Data {
	l.outputObj = inputs
	return l.outputObj
}

func (l *Debug) Forward() {

}

func (l *Debug) Backward() {
	l.counter++
	if l.counter >= l.stopCnt {
		//fmt.Println(l.outputObj.StringData())
		fmt.Println(l.outputObj.StringGrad())
		os.Exit(1)
	}
}
