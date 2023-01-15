.PHONY: fmt
fmt:
	@gofmt -l -w $(shell find . -type f -name '*.go' -not -path "./vendor/*")

.PHONY: lint
lint:
	golangci-lint cache clean
	golangci-lint run ./... -v --timeout 120s

.PHONY: test-amd64
test-amd64:
	go test --tags=amd64 ./...

.PHONY: test-noasm
test-noasm:
	go test --tags=noasm ./...

.PHONY: vendors
vendors:
	go mod download

.PHONY: bench
bench:
	go test ./... -bench . -run ^$ | grep -E 'ns/op' && \

.PHONY: bench
bench-noasm:
	go test --tags=noasm ./... -bench . -run ^$ | grep -E 'ns/op'
