// Package config_test verifies configuration loading using deterministic providers.
package config_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/internal/config"
)

type mapEnvironmentProvider struct {
	values map[string]string
}

func (provider mapEnvironmentProvider) Get(key string) string {
	return provider.values[key]
}

func TestLoadFromEnvSuccess(t *testing.T) {
	t.Parallel()

	envValues := map[string]string{
		"MCP_NATS_URL":                        "nats://localhost:4222",
		"MCP_NATS_STREAM":                     "mcp_stream",
		"MCP_NATS_TASK_REQUEST_SUBJECT":       "leadership.task.request",
		"MCP_NATS_ASSIGNMENT_SUBJECT":         "leadership.task.assignment",
		"MCP_NATS_WORK_READY_SUBJECT":         "team.development.work.ready",
		"MCP_NATS_BLOCKED_SUBJECT":            "team.development.blocked",
		"MCP_NATS_VOTE_SUBJECT":               "team.development.vote",
		"MCP_QDRANT_URL":                      "https://qdrant.local",
		"MCP_QDRANT_API_KEY":                  "test-key",
		"MCP_QDRANT_TIMEOUT_SECONDS":          "20",
		"MCP_QDRANT_WORK_PACKET_COLLECTION":   "work_packets",
		"MCP_QDRANT_TEAM_CHAT_PREFIX":         "team_chat_",
		"MCP_QDRANT_ARTEFACT_PREFIX":          "artefacts_",
		"MCP_LEADERSHIP_CHAT_COLLECTION":      "team_chat_leadership",
		"MCP_LEADERSHIP_ARTEFACT_COLLECTION":  "artefacts_leadership",
		"MCP_LEADERSHIP_COMMANDS":             "echo,ls",
		"MCP_DEVELOPMENT_CHAT_COLLECTION":     "team_chat_development",
		"MCP_DEVELOPMENT_ARTEFACT_COLLECTION": "artefacts_development",
		"MCP_DEVELOPMENT_COMMANDS":            "go test,go build",
		"MCP_MCP_HOST":                        "127.0.0.1",
		"MCP_MCP_PORT":                        "8080",
		"MCP_RATE_LIMIT_PER_MINUTE":           "30",
		"MCP_DRY_RUN":                         "false",
	}

	cfg, loadErr := config.LoadWithProvider(mapEnvironmentProvider{values: envValues})
	require.NoError(t, loadErr)

	require.Equal(t, "nats://localhost:4222", cfg.NATS.URL)
	require.Equal(t, "mcp_stream", cfg.NATS.Stream)
	require.Equal(t, "leadership.task.request", cfg.NATS.TaskRequestSubject)
	require.Equal(t, "leadership.task.assignment", cfg.NATS.TaskAssignmentSubject)
	require.Equal(t, "team.development.work.ready", cfg.NATS.WorkReadySubject)
	require.Equal(t, "team.development.blocked", cfg.NATS.BlockedSubject)
	require.Equal(t, "team.development.vote", cfg.NATS.VoteSubject)

	require.Equal(t, "https://qdrant.local", cfg.Qdrant.URL)
	require.Equal(t, "test-key", cfg.Qdrant.APIKey)
	require.Equal(t, 20, cfg.Qdrant.TimeoutSeconds)
	require.Equal(t, "work_packets", cfg.Qdrant.Collections.WorkPackets)
	require.Equal(t, "team_chat_", cfg.Qdrant.Collections.TeamChatPrefix)
	require.Equal(t, "artefacts_", cfg.Qdrant.Collections.ArtefactPrefix)

	require.Equal(t, []string{"echo", "ls"}, cfg.Teams.Leadership.CommandWhitelist)
	require.Equal(t, "team_chat_leadership", cfg.Teams.Leadership.ChatCollection)
	require.Equal(t, "artefacts_leadership", cfg.Teams.Leadership.ArtefactCollection)

	require.Equal(t, []string{"go test", "go build"}, cfg.Teams.Development.CommandWhitelist)
	require.Equal(t, "team_chat_development", cfg.Teams.Development.ChatCollection)
	require.Equal(t, "artefacts_development", cfg.Teams.Development.ArtefactCollection)

	require.Equal(t, "127.0.0.1", cfg.MCP.Host)
	require.Equal(t, 8080, cfg.MCP.Port)
	require.Equal(t, 30, cfg.MCP.RateLimitPerMinute)
	require.False(t, cfg.MCP.DryRun)
}

func TestLoadMissingRequiredEnvFails(t *testing.T) {
	t.Parallel()

	envValues := map[string]string{
		"MCP_NATS_URL":       "nats://localhost:4222",
		"MCP_QDRANT_URL":     "https://qdrant.local",
		"MCP_QDRANT_API_KEY": "test-key",
	}

	_, loadErr := config.LoadWithProvider(mapEnvironmentProvider{values: envValues})
	require.Error(t, loadErr)
}
