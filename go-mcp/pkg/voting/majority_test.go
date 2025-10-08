// Package voting validates the consensus evaluation logic built atop the domain vote models.
package voting_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/voting"
)

func TestNewMajorityEvaluatorRejectsInvalidThreshold(t *testing.T) {
	t.Parallel()

	_, evaluatorErr := voting.NewMajorityEvaluator(1.2)
	require.ErrorIs(t, evaluatorErr, voting.ErrMajorityEvaluatorInvalidThreshold)
}

func TestEvaluateSimpleMajorityApproves(t *testing.T) {
	t.Parallel()

	evaluator, evaluatorErr := voting.NewMajorityEvaluator(voting.SimpleMajorityThreshold)
	require.NoError(t, evaluatorErr)

	votes := []model.AgentVote{
		{
			AgentID:       "agent-alpha",
			Decision:      model.VoteDecisionApprove,
			Confidence:    90,
			Justification: "Solution matches expectations.",
			CastedAt:      time.Date(2024, 10, 12, 15, 0, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-bravo",
			Decision:      model.VoteDecisionApprove,
			Confidence:    80,
			Justification: "Tests pass locally.",
			CastedAt:      time.Date(2024, 10, 12, 15, 1, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-charlie",
			Decision:      model.VoteDecisionReject,
			Confidence:    60,
			Justification: "Code style issue to resolve.",
			CastedAt:      time.Date(2024, 10, 12, 15, 2, 0, 0, time.UTC),
			Metadata:      nil,
		},
	}

	result, evaluateErr := evaluator.Evaluate(votes)
	require.NoError(t, evaluateErr)
	require.True(t, result.Approved)
	require.InDelta(t, 0.666, result.ApprovalRatio, 0.001)
	require.Len(t, result.Approvals, 2)
	require.Len(t, result.Rejections, 1)
	require.Empty(t, result.ApprovalsWithChanges)
}

func TestEvaluateMajorityRejectsWhenThresholdNotMet(t *testing.T) {
	t.Parallel()

	evaluator, evaluatorErr := voting.NewMajorityEvaluator(voting.SimpleMajorityThreshold)
	require.NoError(t, evaluatorErr)

	votes := []model.AgentVote{
		{
			AgentID:       "agent-alpha",
			Decision:      model.VoteDecisionApprove,
			Confidence:    80,
			Justification: "Core functionality verified.",
			CastedAt:      time.Date(2024, 10, 12, 16, 0, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-bravo",
			Decision:      model.VoteDecisionApproveWithChanges,
			Confidence:    70,
			Justification: "Minor documentation issues.",
			CastedAt:      time.Date(2024, 10, 12, 16, 1, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-charlie",
			Decision:      model.VoteDecisionReject,
			Confidence:    75,
			Justification: "Missing integration tests.",
			CastedAt:      time.Date(2024, 10, 12, 16, 2, 0, 0, time.UTC),
			Metadata:      nil,
		},
	}

	result, evaluateErr := evaluator.Evaluate(votes)
	require.NoError(t, evaluateErr)
	require.False(t, result.Approved)
	require.InDelta(t, 0.333, result.ApprovalRatio, 0.001)
	require.Len(t, result.Approvals, 1)
	require.Len(t, result.ApprovalsWithChanges, 1)
	require.Len(t, result.Rejections, 1)
}

func TestEvaluateMajorityUsesCustomThreshold(t *testing.T) {
	t.Parallel()

	evaluator, evaluatorErr := voting.NewMajorityEvaluator(0.75)
	require.NoError(t, evaluatorErr)

	votes := []model.AgentVote{
		{
			AgentID:       "agent-alpha",
			Decision:      model.VoteDecisionApprove,
			Confidence:    95,
			Justification: "All acceptance criteria satisfied.",
			CastedAt:      time.Date(2024, 10, 12, 17, 0, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-bravo",
			Decision:      model.VoteDecisionApprove,
			Confidence:    88,
			Justification: "Linting and tests are green.",
			CastedAt:      time.Date(2024, 10, 12, 17, 1, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-charlie",
			Decision:      model.VoteDecisionApprove,
			Confidence:    70,
			Justification: "Verified deployment checklist.",
			CastedAt:      time.Date(2024, 10, 12, 17, 2, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-delta",
			Decision:      model.VoteDecisionReject,
			Confidence:    65,
			Justification: "Concern about error handling in edge cases.",
			CastedAt:      time.Date(2024, 10, 12, 17, 3, 0, 0, time.UTC),
			Metadata:      nil,
		},
	}

	result, evaluateErr := evaluator.Evaluate(votes)
	require.NoError(t, evaluateErr)
	require.True(t, result.Approved)
	require.InDelta(t, 0.75, result.ApprovalRatio, 0.001)
	require.Len(t, result.Approvals, 3)
	require.Len(t, result.Rejections, 1)
}

func TestEvaluateMajorityNoVotesFails(t *testing.T) {
	t.Parallel()

	evaluator, evaluatorErr := voting.NewMajorityEvaluator(voting.SimpleMajorityThreshold)
	require.NoError(t, evaluatorErr)

	_, evaluateErr := evaluator.Evaluate(nil)
	require.ErrorIs(t, evaluateErr, voting.ErrMajorityEvaluatorNoVotes)
}

func TestEvaluateMajorityInvalidVoteFails(t *testing.T) {
	t.Parallel()

	evaluator, evaluatorErr := voting.NewMajorityEvaluator(voting.SimpleMajorityThreshold)
	require.NoError(t, evaluatorErr)

	votes := []model.AgentVote{
		{
			AgentID:       "agent-alpha",
			Decision:      model.VoteDecisionApprove,
			Confidence:    90,
			Justification: "All requirements satisfied.",
			CastedAt:      time.Date(2024, 10, 12, 15, 0, 0, 0, time.UTC),
			Metadata:      nil,
		},
		{
			AgentID:       "agent-bravo",
			Decision:      model.VoteDecisionApprove,
			Confidence:    85,
			Justification: "Everything looks good.",
			CastedAt:      time.Time{},
			Metadata:      nil,
		},
	}

	_, evaluateErr := evaluator.Evaluate(votes)
	require.ErrorIs(t, evaluateErr, voting.ErrMajorityEvaluatorInvalidVote)
}
