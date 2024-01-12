.PHONY: pprof
pprof:
	go tool pprof http://localhost:6060/debug/pprof/profile

.PHONY: gpt-train
gpt-train:
	go run gpt/cmd/train/main.go

.PHONY: gpt-test
gpt-test:
	go run gpt/cmd/test/main.go

