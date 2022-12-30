.PHONY: test bench
test:
	go test ./... && go test --tags=noasm ./...

bench:
	clear && \
	echo "--- bench ---" && \
	go test ./... -bench . -run ^$ | grep -E 'ns/op' && \
	echo "--- bench noasm ---" && \
	go test --tags=noasm ./... -bench . -run ^$ | grep -E 'ns/op'

