GO_PACKAGES := ./...
SERVICES := leadership-mcp development-mcp
BINARY_DIRECTORY := $(HOME)/bin

.PHONY: build test test-cover test-race clean fmt vet lint run-leadership run-development install help

build:
	mkdir -p $(BINARY_DIRECTORY)
	@for service in $(SERVICES); do \
		go build -o $(BINARY_DIRECTORY)/$$service ./cmd/$$service ; \
	done

test:
	go test -v $(GO_PACKAGES)

test-cover:
	go test -coverprofile=coverage.out $(GO_PACKAGES)
	go tool cover -html=coverage.out

test-race:
	go test -race $(GO_PACKAGES)

clean:
	rm -f coverage.out
	@for service in $(SERVICES); do \
		rm -f $(BINARY_DIRECTORY)/$$service ; \
	done

fmt:
	gofmt -s -w .

vet:
	go vet $(GO_PACKAGES)

lint:
	golangci-lint run ./...

run-leadership:
	go run ./cmd/leadership-mcp

run-development:
	go run ./cmd/development-mcp

install:
	go mod tidy

help:
	@echo "Targets:"
	@echo "  build            - Build all MCP service binaries into $(BINARY_DIRECTORY)"
	@echo "  test             - Run the unit test suite"
	@echo "  test-cover       - Run tests with coverage report"
	@echo "  test-race        - Run tests with the race detector"
	@echo "  clean            - Remove build and coverage artefacts"
	@echo "  fmt              - Format Go sources with gofmt"
	@echo "  vet              - Run go vet across the module"
	@echo "  lint             - Execute golangci-lint"
	@echo "  run-leadership   - Run the leadership MCP entrypoint"
	@echo "  run-development  - Run the development MCP entrypoint"
	@echo "  install          - Synchronise module dependencies"
	@echo "  help             - Display this help message"

