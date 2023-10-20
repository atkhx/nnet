# open blas
CGO_CFLAGS="-I/opt/homebrew/opt/openblas/include"
CGO_LDFLAGS="-L/opt/homebrew/opt/openblas/lib -lopenblas"

#CGO_CFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX13.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/ -DACCELERATE_NEW_LAPACK=1"
#CGO_LDFLAGS="-lcblas"

.PHONY: pprof
pprof:
	go tool pprof http://localhost:6060/debug/pprof/profile

.PHONY: train
train:
	env CGO_CFLAGS=$(CGO_CFLAGS) CGO_LDFLAGS=$(CGO_LDFLAGS) go run gpt/cmd/train/main.go

