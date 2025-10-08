// Executor tests verify whitelist enforcement and auditing behaviour.
package command_test

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/internal/command"
)

// locateShellExecutable discovers the absolute path to a POSIX-compliant shell.
func locateShellExecutable(tb testing.TB) string {
	tb.Helper()

	var (
		searchError     error
		shellExecutable string
	)

	for _, candidate := range []string{"bash", "sh"} {
		shellExecutable, searchError = exec.LookPath(candidate)
		if searchError == nil {
			return shellExecutable
		}
	}

	tb.Fatalf("failed to locate shell executable: %v", searchError)

	return ""
}

func TestNewExecutorRejectsEmptyWhitelist(t *testing.T) {
	t.Parallel()

	_, executorError := command.NewExecutor(nil)
	if executorError == nil {
		t.Fatalf("expected error when whitelist is empty")
	}
}

func TestExecuteRejectsCommandNotWhitelisted(t *testing.T) {
	t.Parallel()

	allowedCommand := locateShellExecutable(t)

	executorInstance, constructorError := command.NewExecutor([]string{allowedCommand})
	if constructorError != nil {
		t.Fatalf("unexpected constructor error: %v", constructorError)
	}

	executionContext, executionCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer executionCancel()

	_, executionError := executorInstance.Execute(
		executionContext,
		filepath.Join("/", "usr", "bin", "env"),
		nil,
		"tester",
		"env call",
	)
	if executionError == nil {
		t.Fatalf("expected whitelist error but received nil")
	}

	if !strings.Contains(executionError.Error(), command.ErrCommandNotWhitelisted.Error()) {
		t.Fatalf("expected error to contain %q but received %v", command.ErrCommandNotWhitelisted, executionError)
	}
}

func TestExecuteAllowedCommandSucceeds(t *testing.T) {
	t.Parallel()

	allowedCommand := locateShellExecutable(t)

	executorInstance, constructorError := command.NewExecutor([]string{allowedCommand})
	if constructorError != nil {
		t.Fatalf("unexpected constructor error: %v", constructorError)
	}

	executionContext, executionCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer executionCancel()

	arguments := []string{"-c", "printf 'out' && >&2 printf 'err'"}

	record, executionError := executorInstance.Execute(
		executionContext,
		allowedCommand,
		arguments,
		"tester",
		"successful command",
	)
	if executionError != nil {
		t.Fatalf("unexpected execution error: %v", executionError)
	}

	if record.Command != allowedCommand {
		t.Fatalf("expected command %q but received %q", allowedCommand, record.Command)
	}

	if strings.TrimSpace(record.Stdout) != "out" {
		t.Fatalf("expected stdout 'out' but received %q", record.Stdout)
	}

	if strings.TrimSpace(record.Stderr) != "err" {
		t.Fatalf("expected stderr 'err' but received %q", record.Stderr)
	}

	if record.ExitCode != 0 {
		t.Fatalf("expected exit code 0 but received %d", record.ExitCode)
	}

	if record.InvokedBy != "tester" {
		t.Fatalf("expected invoked by tester but received %q", record.InvokedBy)
	}

	if record.Description != "successful command" {
		t.Fatalf("expected description %q but received %q", "successful command", record.Description)
	}

	if record.RecordedAt.IsZero() {
		t.Fatalf("expected non-zero recorded timestamp")
	}
}

func TestExecuteFailedCommandCapturesOutput(t *testing.T) {
	t.Parallel()

	allowedCommand := locateShellExecutable(t)

	executorInstance, constructorError := command.NewExecutor([]string{allowedCommand})
	if constructorError != nil {
		t.Fatalf("unexpected constructor error: %v", constructorError)
	}

	executionContext, executionCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer executionCancel()

	arguments := []string{"-c", "printf 'partial' && >&2 printf 'error' && exit 12"}

	record, executionError := executorInstance.Execute(
		executionContext,
		allowedCommand,
		arguments,
		"tester",
		"failing command",
	)
	if executionError == nil {
		t.Fatalf("expected failure error but received nil")
	}

	if !strings.Contains(executionError.Error(), command.ErrCommandExecutionFailed.Error()) {
		t.Fatalf("expected execution error to contain %q but received %v", command.ErrCommandExecutionFailed, executionError)
	}

	if strings.TrimSpace(record.Stdout) != "partial" {
		t.Fatalf("expected stdout 'partial' but received %q", record.Stdout)
	}

	if strings.TrimSpace(record.Stderr) != "error" {
		t.Fatalf("expected stderr 'error' but received %q", record.Stderr)
	}

	if record.ExitCode != 12 {
		t.Fatalf("expected exit code 12 but received %d", record.ExitCode)
	}

	if record.RecordedAt.IsZero() {
		t.Fatalf("expected non-zero recorded timestamp")
	}
}
