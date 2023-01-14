.PHONY: test
test:
	go test ./...

.PHONY: test-noasm
test-noasm:
	go test --tags=noasm ./...

.PHONY: bench
bench:
	go test ./... -bench . -run ^$ | grep -E 'ns/op' && \

.PHONY: bench
bench-noasm:
	go test --tags=noasm ./... -bench . -run ^$ | grep -E 'ns/op'

.PHONY: lint
lint:
	golangci-lint cache clean
	golangci-lint run ./... -v --timeout 120s