.PHONY: fmt
fmt:
	@gofmt -l -w $(shell find . -type f -name '*.go' -not -path "./vendor/*")

.PHONY: lint
lint:
	golangci-lint cache clean
	golangci-lint run ./... -v --timeout 120s

.PHONY: imports
imports:
	@goimports -w -local github.com/atkhx/nnet $(shell find . -type f -name '*.go' -not -path "./vendor/*")
