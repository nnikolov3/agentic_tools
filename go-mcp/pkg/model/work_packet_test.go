// Reusable domain model validation tests for the MCP services.
// This file exercises the validation logic for core entities.

package model_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

func TestWorkPacketValidateSuccess(t *testing.T) {
	t.Parallel()

	workPacket := model.WorkPacket{
		ID:          "packet-123",
		Objective:   "Implement config loader",
		Summary:     "Adds configuration parsing with validation",
		CreatedAt:   time.Now(),
		CreatedBy:   "leadership-coordinator",
		TargetTeam:  model.TeamNameDevelopment,
		CurrentStep: model.WorkPacketStepAssignment,
		Tags:        []string{"config"},
		Metadata:    map[string]string{"epic": "foundations"},
	}

	validateErr := workPacket.Validate()
	require.NoError(t, validateErr)
}

func TestWorkPacketValidateMissingIDFails(t *testing.T) {
	t.Parallel()

	workPacket := model.WorkPacket{
		ID:          "",
		Objective:   "Implement config loader",
		Summary:     "Adds configuration parsing with validation",
		CreatedAt:   time.Now(),
		CreatedBy:   "leadership-coordinator",
		TargetTeam:  model.TeamNameDevelopment,
		CurrentStep: model.WorkPacketStepAssignment,
		Tags:        nil,
		Metadata:    nil,
	}

	validateErr := workPacket.Validate()
	require.ErrorIs(t, validateErr, model.ErrWorkPacketMissingID)
}

func TestWorkPacketValidateMissingObjectiveFails(t *testing.T) {
	t.Parallel()

	workPacket := model.WorkPacket{
		ID:          "packet-123",
		Objective:   "",
		Summary:     "Adds configuration parsing with validation",
		CreatedAt:   time.Now(),
		CreatedBy:   "leadership-coordinator",
		TargetTeam:  model.TeamNameDevelopment,
		CurrentStep: model.WorkPacketStepAssignment,
		Tags:        nil,
		Metadata:    nil,
	}

	validateErr := workPacket.Validate()
	require.ErrorIs(t, validateErr, model.ErrWorkPacketMissingObjective)
}

func TestWorkPacketValidateMissingTargetTeamFails(t *testing.T) {
	t.Parallel()

	workPacket := model.WorkPacket{
		ID:          "packet-123",
		Objective:   "Implement config loader",
		Summary:     "Adds configuration parsing with validation",
		CreatedAt:   time.Now(),
		CreatedBy:   "leadership-coordinator",
		TargetTeam:  "",
		CurrentStep: model.WorkPacketStepAssignment,
		Tags:        nil,
		Metadata:    nil,
	}

	validateErr := workPacket.Validate()
	require.ErrorIs(t, validateErr, model.ErrWorkPacketMissingTargetTeam)
}

func TestCommandExecutionRecordValidateSuccess(t *testing.T) {
	t.Parallel()

	commandRecord := model.CommandExecutionRecord{
		Command:     "go test",
		Arguments:   []string{"./..."},
		Stdout:      "ok",
		Stderr:      "",
		ExitCode:    0,
		InvokedBy:   "development-coordinator",
		RecordedAt:  time.Now(),
		Description: "Run unit tests for the repository",
	}

	validateErr := commandRecord.Validate()
	require.NoError(t, validateErr)
}

func TestCommandExecutionRecordValidateMissingCommandFails(t *testing.T) {
	t.Parallel()

	commandRecord := model.CommandExecutionRecord{
		Command:     "",
		Arguments:   []string{"./..."},
		Stdout:      "ok",
		Stderr:      "",
		ExitCode:    0,
		InvokedBy:   "development-coordinator",
		RecordedAt:  time.Now(),
		Description: "Run unit tests for the repository",
	}

	validateErr := commandRecord.Validate()
	require.ErrorIs(t, validateErr, model.ErrCommandExecutionMissingCommand)
}

func TestCommandExecutionRecordValidateMissingInvokerFails(t *testing.T) {
	t.Parallel()

	commandRecord := model.CommandExecutionRecord{
		Command:     "go test",
		Arguments:   []string{"./..."},
		Stdout:      "ok",
		Stderr:      "",
		ExitCode:    0,
		InvokedBy:   "",
		RecordedAt:  time.Now(),
		Description: "Run unit tests for the repository",
	}

	validateErr := commandRecord.Validate()
	require.ErrorIs(t, validateErr, model.ErrCommandExecutionMissingInvoker)
}

func TestCommandExecutionRecordValidateZeroTimestampFails(t *testing.T) {
	t.Parallel()

	commandRecord := model.CommandExecutionRecord{
		Command:     "go test",
		Arguments:   []string{"./..."},
		Stdout:      "ok",
		Stderr:      "",
		ExitCode:    0,
		InvokedBy:   "development-coordinator",
		RecordedAt:  time.Time{},
		Description: "Run unit tests for the repository",
	}

	validateErr := commandRecord.Validate()
	require.ErrorIs(t, validateErr, model.ErrCommandExecutionMissingTimestamp)
}

func TestAgentVoteValidateSuccess(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "agent-alpha",
		Decision:      model.VoteDecisionApprove,
		Confidence:    92,
		Justification: "Implementation matches design expectations.",
		CastedAt:      time.Now(),
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.NoError(t, validateErr)
}

func TestAgentVoteValidateMissingAgentIDFails(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "",
		Decision:      model.VoteDecisionApprove,
		Confidence:    92,
		Justification: "Implementation matches design expectations.",
		CastedAt:      time.Now(),
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.ErrorIs(t, validateErr, model.ErrAgentVoteMissingAgentID)
}

func TestAgentVoteValidateInvalidDecisionFails(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "agent-alpha",
		Decision:      model.VoteDecision("maybe"),
		Confidence:    92,
		Justification: "Implementation matches design expectations.",
		CastedAt:      time.Now(),
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.ErrorIs(t, validateErr, model.ErrAgentVoteInvalidDecision)
}

func TestAgentVoteValidateInvalidConfidenceFails(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "agent-alpha",
		Decision:      model.VoteDecisionApprove,
		Confidence:    140,
		Justification: "Implementation matches design expectations.",
		CastedAt:      time.Now(),
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.ErrorIs(t, validateErr, model.ErrAgentVoteInvalidConfidence)
}

func TestAgentVoteValidateNegativeConfidenceFails(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "agent-alpha",
		Decision:      model.VoteDecisionApprove,
		Confidence:    -10,
		Justification: "Confidence cannot be negative.",
		CastedAt:      time.Now(),
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.ErrorIs(t, validateErr, model.ErrAgentVoteInvalidConfidence)
}

func TestAgentVoteValidateMissingTimestampFails(t *testing.T) {
	t.Parallel()

	vote := model.AgentVote{
		AgentID:       "agent-alpha",
		Decision:      model.VoteDecisionApprove,
		Confidence:    75,
		Justification: "Timestamp is required to order votes.",
		CastedAt:      time.Time{},
		Metadata:      nil,
	}

	validateErr := vote.Validate()
	require.ErrorIs(t, validateErr, model.ErrAgentVoteMissingTimestamp)
}
