// Validation tests for chat message domain primitives.

package model_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

func TestTeamChatMessageValidateSuccess(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Now(),
		Metadata:  map[string]string{"thread": "planning"},
	}

	validateErr := chatMessage.Validate()
	require.NoError(t, validateErr)
}

func TestTeamChatMessageValidateMissingMessageIDFails(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Now(),
		Metadata:  nil,
	}

	validateErr := chatMessage.Validate()
	require.ErrorIs(t, validateErr, model.ErrTeamChatMessageMissingID)
}

func TestTeamChatMessageValidateMissingTeamFails(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      "",
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Now(),
		Metadata:  nil,
	}

	validateErr := chatMessage.Validate()
	require.ErrorIs(t, validateErr, model.ErrTeamChatMessageMissingTeam)
}

func TestTeamChatMessageValidateMissingSenderFails(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Now(),
		Metadata:  nil,
	}

	validateErr := chatMessage.Validate()
	require.ErrorIs(t, validateErr, model.ErrTeamChatMessageMissingSender)
}

func TestTeamChatMessageValidateMissingContentFails(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "",
		CreatedAt: time.Now(),
		Metadata:  nil,
	}

	validateErr := chatMessage.Validate()
	require.ErrorIs(t, validateErr, model.ErrTeamChatMessageMissingContent)
}

func TestTeamChatMessageValidateMissingTimestampFails(t *testing.T) {
	t.Parallel()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Time{},
		Metadata:  nil,
	}

	validateErr := chatMessage.Validate()
	require.ErrorIs(t, validateErr, model.ErrTeamChatMessageMissingTimestamp)
}
