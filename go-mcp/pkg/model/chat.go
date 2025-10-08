// Chat domain entities shared between MCP services.

package model

import (
	"errors"
	"strings"
	"time"
)

// ChatMessageRole indicates the responsibility of the sender within the conversation.
type ChatMessageRole string

// Supported chat message roles used during agent deliberations.
const (
	ChatMessageRoleCoordinator ChatMessageRole = "coordinator"
	ChatMessageRoleSpecialist  ChatMessageRole = "specialist"
	ChatMessageRoleObserver    ChatMessageRole = "observer"
	ChatMessageRoleSystem      ChatMessageRole = "system"
)

var (
	// ErrTeamChatMessageMissingID indicates that the chat message lacks a primary identifier.
	ErrTeamChatMessageMissingID = errors.New("team chat message id cannot be empty")
	// ErrTeamChatMessageMissingTeam indicates that the team field is empty.
	ErrTeamChatMessageMissingTeam = errors.New("team chat message team cannot be empty")
	// ErrTeamChatMessageMissingSender indicates that the sender field is empty.
	ErrTeamChatMessageMissingSender = errors.New("team chat message sender cannot be empty")
	// ErrTeamChatMessageMissingContent indicates that the content field is blank.
	ErrTeamChatMessageMissingContent = errors.New("team chat message content cannot be empty")
	// ErrTeamChatMessageMissingTimestamp indicates that the message timestamp is zero.
	ErrTeamChatMessageMissingTimestamp = errors.New("team chat message timestamp cannot be empty")
)

// TeamChatMessage captures a single conversational entry recorded for a team.
type TeamChatMessage struct {
	MessageID string
	Team      TeamName
	Sender    string
	Role      ChatMessageRole
	Content   string
	CreatedAt time.Time
	Metadata  map[string]string
}

// Validate ensures the chat message contains the minimum required fields.
func (message TeamChatMessage) Validate() error {
	if strings.TrimSpace(message.MessageID) == "" {
		return ErrTeamChatMessageMissingID
	}

	if strings.TrimSpace(string(message.Team)) == "" {
		return ErrTeamChatMessageMissingTeam
	}

	if strings.TrimSpace(message.Sender) == "" {
		return ErrTeamChatMessageMissingSender
	}

	if strings.TrimSpace(message.Content) == "" {
		return ErrTeamChatMessageMissingContent
	}

	if message.CreatedAt.IsZero() {
		return ErrTeamChatMessageMissingTimestamp
	}

	return nil
}
