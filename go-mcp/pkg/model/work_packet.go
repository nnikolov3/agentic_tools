// Work packet domain primitives shared across MCP services.

package model

import (
	"errors"
	"strings"
	"time"
)

// TeamName identifies a collaborating team within the MCP ecosystem.
type TeamName string

// Supported team identifiers for the MVP implementation.
const (
	TeamNameLeadership  TeamName = "leadership"
	TeamNameDevelopment TeamName = "development"
)

// WorkPacketStep captures the lifecycle stage of a work packet.
type WorkPacketStep string

// Known work packet lifecycle stages.
const (
	WorkPacketStepAssignment WorkPacketStep = "assignment"
	WorkPacketStepExecution  WorkPacketStep = "execution"
	WorkPacketStepReview     WorkPacketStep = "review"
	WorkPacketStepCompleted  WorkPacketStep = "completed"
	WorkPacketStepBlocked    WorkPacketStep = "blocked"
)

var (
	// ErrWorkPacketMissingID indicates that an identifier was not supplied.
	ErrWorkPacketMissingID = errors.New("work packet id cannot be empty")
	// ErrWorkPacketMissingObjective indicates that the objective field is blank.
	ErrWorkPacketMissingObjective = errors.New("work packet objective cannot be empty")
	// ErrWorkPacketMissingTargetTeam indicates that the target team is unspecified.
	ErrWorkPacketMissingTargetTeam = errors.New("work packet target team cannot be empty")
)

// WorkPacket describes a unit of work passed between teams.
type WorkPacket struct {
	ID          string
	Objective   string
	Summary     string
	CreatedAt   time.Time
	CreatedBy   string
	TargetTeam  TeamName
	CurrentStep WorkPacketStep
	Tags        []string
	Metadata    map[string]string
}

// Validate ensures the work packet carries the minimum data required for routing.
func (workPacket WorkPacket) Validate() error {
	if strings.TrimSpace(workPacket.ID) == "" {
		return ErrWorkPacketMissingID
	}

	if strings.TrimSpace(workPacket.Objective) == "" {
		return ErrWorkPacketMissingObjective
	}

	if strings.TrimSpace(string(workPacket.TargetTeam)) == "" {
		return ErrWorkPacketMissingTargetTeam
	}

	return nil
}

var (
	// ErrCommandExecutionMissingCommand indicates that the command value is blank.
	ErrCommandExecutionMissingCommand = errors.New("command execution command cannot be empty")
	// ErrCommandExecutionMissingInvoker indicates that the invoker identifier was not supplied.
	ErrCommandExecutionMissingInvoker = errors.New("command execution invoked by cannot be empty")
	// ErrCommandExecutionMissingTimestamp indicates that no timestamp was attached to the record.
	ErrCommandExecutionMissingTimestamp = errors.New("command execution timestamp cannot be empty")
)

// CommandExecutionRecord captures a single shell command issued by an MCP agent.
type CommandExecutionRecord struct {
	Command     string
	Arguments   []string
	Stdout      string
	Stderr      string
	ExitCode    int
	InvokedBy   string
	RecordedAt  time.Time
	Description string
}

// Validate checks that the command record contains the essential auditing fields.
func (record CommandExecutionRecord) Validate() error {
	if strings.TrimSpace(record.Command) == "" {
		return ErrCommandExecutionMissingCommand
	}

	if strings.TrimSpace(record.InvokedBy) == "" {
		return ErrCommandExecutionMissingInvoker
	}

	if record.RecordedAt.IsZero() {
		return ErrCommandExecutionMissingTimestamp
	}

	return nil
}

// VoteDecision represents the discrete choice an agent can make during voting.
type VoteDecision string

// Supported vote decisions for MVP majority voting.
const (
	VoteDecisionApprove            VoteDecision = "approve"
	VoteDecisionApproveWithChanges VoteDecision = "approve_with_changes"
	VoteDecisionReject             VoteDecision = "reject"
)

var (
	// ErrAgentVoteMissingAgentID indicates that the vote has no author identifier.
	ErrAgentVoteMissingAgentID = errors.New("agent vote agent id cannot be empty")
	// ErrAgentVoteInvalidDecision indicates that the vote decision is not supported.
	ErrAgentVoteInvalidDecision = errors.New("agent vote decision is invalid")
	// ErrAgentVoteInvalidConfidence indicates that the confidence value is outside 0-100.
	ErrAgentVoteInvalidConfidence = errors.New("agent vote confidence must be between 0 and 100")
	// ErrAgentVoteMissingTimestamp indicates that the vote timestamp is not populated.
	ErrAgentVoteMissingTimestamp = errors.New("agent vote timestamp cannot be empty")
)

// AgentVote stores an individual agent's feedback on a deliverable.
type AgentVote struct {
	AgentID       string
	Decision      VoteDecision
	Confidence    int
	Justification string
	CastedAt      time.Time
	Metadata      map[string]string
}

// Validate checks whether the vote satisfies structural constraints.
func (vote AgentVote) Validate() error {
	if strings.TrimSpace(vote.AgentID) == "" {
		return ErrAgentVoteMissingAgentID
	}

	if vote.Decision != VoteDecisionApprove &&
		vote.Decision != VoteDecisionApproveWithChanges &&
		vote.Decision != VoteDecisionReject {
		return ErrAgentVoteInvalidDecision
	}

	if vote.Confidence < 0 || vote.Confidence > 100 {
		return ErrAgentVoteInvalidConfidence
	}

	if vote.CastedAt.IsZero() {
		return ErrAgentVoteMissingTimestamp
	}

	return nil
}
