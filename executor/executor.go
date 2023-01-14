package executor

const maxRoutinesActive = 32

var (
	execFn func(int)

	iChan chan int
	oChan chan int
)

func init() {
	iChan = make(chan int, maxRoutinesActive)
	oChan = make(chan int, maxRoutinesActive)
	for i := 0; i < maxRoutinesActive; i++ {
		go func() {
			for idx := range iChan {
				execFn(idx)
				oChan <- idx
			}
		}()
	}
}

func RunParallel(count int, fn func(n int)) {
	execFn = fn
	go func() {
		for i := 0; i < count; i++ {
			iChan <- i
		}
	}()

	for i := 0; i < count; i++ {
		<-oChan
	}
}
