// Package command provides helpers for executing shell commands under strict whitelists.
package command

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

// ErrExecutorEmptyWhitelist indicates that the executor was initialised without allowed commands.
var ErrExecutorEmptyWhitelist = errors.New("command executor whitelist cannot be empty")

// ErrCommandNotWhitelisted indicates that a command is not permitted by the whitelist.
var ErrCommandNotWhitelisted = errors.New("command executor command not whitelisted")

// ErrCommandExecutionFailed indicates that executing the command returned a failure.
var ErrCommandExecutionFailed = errors.New("command executor command failed")

type timeSource interface {
	Now() time.Time
}

type systemTimeSource struct{}

func (source systemTimeSource) Now() time.Time {
	return time.Now().UTC()
}

// Executor enforces a whitelist and records auditing metadata for executed commands.
type Executor struct {
	allowedCommands map[string]struct{}
	clock           timeSource
}

// NewExecutor constructs a command executor using the supplied whitelist.
func NewExecutor(allowedCommands []string) (*Executor, error) {
	if len(allowedCommands) == 0 {
		return nil, ErrExecutorEmptyWhitelist
	}

	commandLookup := make(map[string]struct{}, len(allowedCommands))
	for _, commandPath := range allowedCommands {
		trimmedCommand := strings.TrimSpace(commandPath)
		if trimmedCommand == "" {
			continue
		}

		commandLookup[trimmedCommand] = struct{}{}
	}

	if len(commandLookup) == 0 {
		return nil, ErrExecutorEmptyWhitelist
	}

	return &Executor{
		allowedCommands: commandLookup,
		clock:           systemTimeSource{},
	}, nil
}

// Execute runs the supplied command with arguments under the configured whitelist.
func (executor *Executor) Execute(
	ctx context.Context,
	commandPath string,
	arguments []string,
	invokedBy string,
	description string,
) (model.CommandExecutionRecord, error) {
	if _, exists := executor.allowedCommands[commandPath]; !exists {
		return model.CommandExecutionRecord{}, fmt.Errorf("%w: %s", ErrCommandNotWhitelisted, commandPath)
	}

	execution := exec.CommandContext(ctx, commandPath, arguments...)

	var (
		stdoutBuffer bytes.Buffer
		stderrBuffer bytes.Buffer
	)

	execution.Stdout = &stdoutBuffer
	execution.Stderr = &stderrBuffer

	runError := execution.Run()

	exitCode := 0

	if runError != nil {
		var exitError *exec.ExitError
		if errors.As(runError, &exitError) {
			exitCode = exitError.ExitCode()
		} else {
			exitCode = -1
		}
	}

	record := model.CommandExecutionRecord{
		Command:     commandPath,
		Arguments:   append([]string(nil), arguments...),
		Stdout:      stdoutBuffer.String(),
		Stderr:      stderrBuffer.String(),
		ExitCode:    exitCode,
		InvokedBy:   invokedBy,
		RecordedAt:  executor.clock.Now(),
		Description: description,
	}

	if runError != nil {
		return record, fmt.Errorf("%w: %w", ErrCommandExecutionFailed, runError)
	}

	return record, nil
}
