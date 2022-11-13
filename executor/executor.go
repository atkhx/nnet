package executor

import (
	"sync"
)

var (
	mu = sync.Mutex{}
	wg = sync.WaitGroup{}
)

func RunParallel(count int, fn func(n int)) {
	//for n := 0; n < count; n++ {
	//	fn(n)
	//}
	mu.Lock()
	wg.Add(count)
	for n := 0; n < count; n++ {
		go func(n int) {
			fn(n)
			wg.Done()
		}(n)
	}
	wg.Wait()
	mu.Unlock()
}
