// Package voting implements consensus calculations for MCP agent teams.
package voting

import (
	"errors"
	"fmt"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

// SimpleMajorityThreshold represents the default approval ratio (50%) for majority voting.
const SimpleMajorityThreshold = 0.5

var (
	// ErrMajorityEvaluatorInvalidThreshold indicates that the supplied threshold is outside (0,1).
	ErrMajorityEvaluatorInvalidThreshold = errors.New("majority evaluator threshold must be between 0 and 1")
	// ErrMajorityEvaluatorNoVotes indicates that an evaluation was attempted without any votes.
	ErrMajorityEvaluatorNoVotes = errors.New("majority evaluator requires at least one vote")
	// ErrMajorityEvaluatorInvalidVote indicates that a vote failed structural validation.
	ErrMajorityEvaluatorInvalidVote = errors.New("majority evaluator encountered an invalid vote")
)

// MajorityEvaluator calculates whether a vote collection satisfies the configured threshold.
type MajorityEvaluator struct {
	threshold float64
}

// NewMajorityEvaluator configures a majority evaluator for the supplied approval threshold.
func NewMajorityEvaluator(threshold float64) (*MajorityEvaluator, error) {
	if threshold <= 0 || threshold >= 1 {
		return nil, ErrMajorityEvaluatorInvalidThreshold
	}

	return &MajorityEvaluator{threshold: threshold}, nil
}

// MajorityResult summarises the outcome of a voting round.
type MajorityResult struct {
	Approved             bool
	Threshold            float64
	ApprovalRatio        float64
	TotalVotes           int
	Approvals            []model.AgentVote
	ApprovalsWithChanges []model.AgentVote
	Rejections           []model.AgentVote
}

// Evaluate assesses the supplied votes and determines whether the approval threshold was met.
func (evaluator *MajorityEvaluator) Evaluate(votes []model.AgentVote) (MajorityResult, error) {
	if len(votes) == 0 {
		return MajorityResult{}, ErrMajorityEvaluatorNoVotes
	}

	approvals := make([]model.AgentVote, 0, len(votes))
	approvalsWithChanges := make([]model.AgentVote, 0, len(votes))
	rejections := make([]model.AgentVote, 0, len(votes))

	for _, vote := range votes {
		validateErr := vote.Validate()
		if validateErr != nil {
			return MajorityResult{}, fmt.Errorf("%w: %w", ErrMajorityEvaluatorInvalidVote, validateErr)
		}

		switch vote.Decision {
		case model.VoteDecisionApprove:
			approvals = append(approvals, vote)
		case model.VoteDecisionApproveWithChanges:
			approvalsWithChanges = append(approvalsWithChanges, vote)
		case model.VoteDecisionReject:
			rejections = append(rejections, vote)
		default:
			return MajorityResult{}, fmt.Errorf("%w: %s", ErrMajorityEvaluatorInvalidVote, "unsupported vote decision")
		}
	}

	totalVotes := len(votes)
	approvalRatio := float64(len(approvals)) / float64(totalVotes)

	result := MajorityResult{
		Approved:             approvalRatio >= evaluator.threshold,
		Threshold:            evaluator.threshold,
		ApprovalRatio:        approvalRatio,
		TotalVotes:           totalVotes,
		Approvals:            approvals,
		ApprovalsWithChanges: approvalsWithChanges,
		Rejections:           rejections,
	}

	return result, nil
}
