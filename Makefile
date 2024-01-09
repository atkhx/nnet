.PHONY: pprof
pprof:
	go tool pprof http://localhost:6060/debug/pprof/profile

.PHONY: train
train:
	go run gpt/cmd/train/main.go


.PHONY: test
test:
	go run gpt/cmd/test/main.go

