// Package config loads and validates environment-driven configuration for the MCP services.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config captures the complete runtime configuration shared by the MCP services.
type Config struct {
	NATS   NATSConfig
	Qdrant QdrantConfig
	Teams  TeamConfigs
	MCP    MCPConfig
}

// NATSConfig contains connection and subject information for messaging.
type NATSConfig struct {
	URL                   string
	Stream                string
	TaskRequestSubject    string
	TaskAssignmentSubject string
	WorkReadySubject      string
	BlockedSubject        string
	VoteSubject           string
}

// QdrantConfig represents Qdrant connectivity and collection settings.
type QdrantConfig struct {
	URL            string
	APIKey         string
	TimeoutSeconds int
	Collections    QdrantCollections
}

// QdrantCollections provides logical collection names used at runtime.
type QdrantCollections struct {
	WorkPackets    string
	TeamChatPrefix string
	ArtefactPrefix string
}

// TeamConfigs groups configuration for all participating teams.
type TeamConfigs struct {
	Leadership  TeamConfig
	Development TeamConfig
}

// TeamConfig defines per-team settings such as chat collection and whitelisted commands.
type TeamConfig struct {
	ChatCollection     string
	ArtefactCollection string
	CommandWhitelist   []string
	VoteThreshold      float64
}

// MCPConfig stores metadata for the MCP interface exposed by leadership.
type MCPConfig struct {
	Host               string
	Port               int
	RateLimitPerMinute int
	DryRun             bool
}

var (
	// ErrMissingEnvironmentVariable indicates that a required environment variable was not set.
	ErrMissingEnvironmentVariable = errors.New("missing required environment variable")
	// ErrEmptyCommandList signals that a command whitelist resolved to an empty list.
	ErrEmptyCommandList = errors.New("command whitelist cannot be empty")
	// ErrInvalidInteger marks environment variables that failed integer parsing.
	ErrInvalidInteger = errors.New("invalid integer environment variable")
	// ErrInvalidBoolean marks environment variables that failed boolean parsing.
	ErrInvalidBoolean = errors.New("invalid boolean environment variable")
)

const (
	defaultQdrantTimeoutSeconds = 30
	defaultVoteThreshold        = 0.51
)

// Load retrieves configuration from environment variables and performs validation.
func Load() (*Config, error) {
	return load(osEnvironmentProvider{})
}

// EnvironmentProvider supplies configuration values keyed by environment variable names.
type EnvironmentProvider interface {
	Get(key string) string
}

type osEnvironmentProvider struct{}

func (provider osEnvironmentProvider) Get(key string) string {
	return os.Getenv(key)
}

// LoadWithProvider retrieves configuration using the supplied provider, primarily for testing.
func LoadWithProvider(provider EnvironmentProvider) (*Config, error) {
	return load(provider)
}

func load(provider EnvironmentProvider) (*Config, error) {
	natsCfg, natsErr := loadNATSConfig(provider)
	if natsErr != nil {
		return nil, natsErr
	}

	qdrantCfg, qdrantErr := loadQdrantConfig(provider)
	if qdrantErr != nil {
		return nil, qdrantErr
	}

	teamCfg, teamErr := loadTeamConfigs(provider)
	if teamErr != nil {
		return nil, teamErr
	}

	mcpCfg, mcpErr := loadMCPConfig(provider)
	if mcpErr != nil {
		return nil, mcpErr
	}

	return &Config{
		NATS:   natsCfg,
		Qdrant: qdrantCfg,
		Teams:  teamCfg,
		MCP:    mcpCfg,
	}, nil
}

func loadNATSConfig(provider EnvironmentProvider) (NATSConfig, error) {
	var cfg NATSConfig

	urlValue, urlErr := requireStringEnv(provider, "MCP_NATS_URL")
	if urlErr != nil {
		return cfg, urlErr
	}

	streamValue, streamErr := requireStringEnv(provider, "MCP_NATS_STREAM")
	if streamErr != nil {
		return cfg, streamErr
	}

	taskRequestValue, taskRequestErr := requireStringEnv(provider, "MCP_NATS_TASK_REQUEST_SUBJECT")
	if taskRequestErr != nil {
		return cfg, taskRequestErr
	}

	taskAssignmentValue, taskAssignmentErr := requireStringEnv(provider, "MCP_NATS_ASSIGNMENT_SUBJECT")
	if taskAssignmentErr != nil {
		return cfg, taskAssignmentErr
	}

	workReadyValue, workReadyErr := requireStringEnv(provider, "MCP_NATS_WORK_READY_SUBJECT")
	if workReadyErr != nil {
		return cfg, workReadyErr
	}

	blockedValue, blockedErr := requireStringEnv(provider, "MCP_NATS_BLOCKED_SUBJECT")
	if blockedErr != nil {
		return cfg, blockedErr
	}

	voteValue, voteErr := requireStringEnv(provider, "MCP_NATS_VOTE_SUBJECT")
	if voteErr != nil {
		return cfg, voteErr
	}

	cfg = NATSConfig{
		URL:                   urlValue,
		Stream:                streamValue,
		TaskRequestSubject:    taskRequestValue,
		TaskAssignmentSubject: taskAssignmentValue,
		WorkReadySubject:      workReadyValue,
		BlockedSubject:        blockedValue,
		VoteSubject:           voteValue,
	}

	return cfg, nil
}

func loadQdrantConfig(provider EnvironmentProvider) (QdrantConfig, error) {
	var cfg QdrantConfig

	urlValue, urlErr := requireStringEnv(provider, "MCP_QDRANT_URL")
	if urlErr != nil {
		return cfg, urlErr
	}

	apiKeyValue, apiKeyErr := requireStringEnv(provider, "MCP_QDRANT_API_KEY")
	if apiKeyErr != nil {
		return cfg, apiKeyErr
	}

	timeoutSeconds, timeoutErr := parseIntWithDefault(provider, "MCP_QDRANT_TIMEOUT_SECONDS", defaultQdrantTimeoutSeconds)
	if timeoutErr != nil {
		return cfg, timeoutErr
	}

	workPackets, workPacketsErr := requireStringEnv(provider, "MCP_QDRANT_WORK_PACKET_COLLECTION")
	if workPacketsErr != nil {
		return cfg, workPacketsErr
	}

	chatPrefix, chatPrefixErr := requireStringEnv(provider, "MCP_QDRANT_TEAM_CHAT_PREFIX")
	if chatPrefixErr != nil {
		return cfg, chatPrefixErr
	}

	artefactPrefix, artefactPrefixErr := requireStringEnv(provider, "MCP_QDRANT_ARTEFACT_PREFIX")
	if artefactPrefixErr != nil {
		return cfg, artefactPrefixErr
	}

	cfg = QdrantConfig{
		URL:            urlValue,
		APIKey:         apiKeyValue,
		TimeoutSeconds: timeoutSeconds,
		Collections: QdrantCollections{
			WorkPackets:    workPackets,
			TeamChatPrefix: chatPrefix,
			ArtefactPrefix: artefactPrefix,
		},
	}

	return cfg, nil
}

func loadTeamConfigs(provider EnvironmentProvider) (TeamConfigs, error) {
	leadership, leadershipErr := loadTeamConfig(provider, "LEADERSHIP")
	if leadershipErr != nil {
		return TeamConfigs{}, leadershipErr
	}

	development, developmentErr := loadTeamConfig(provider, "DEVELOPMENT")
	if developmentErr != nil {
		return TeamConfigs{}, developmentErr
	}

	return TeamConfigs{
		Leadership:  leadership,
		Development: development,
	}, nil
}

func loadTeamConfig(provider EnvironmentProvider, team string) (TeamConfig, error) {
	var cfg TeamConfig

	chatCollection, chatErr := requireStringEnv(provider, "MCP_"+team+"_CHAT_COLLECTION")
	if chatErr != nil {
		return cfg, chatErr
	}

	artefactCollection, artefactErr := requireStringEnv(provider, "MCP_"+team+"_ARTEFACT_COLLECTION")
	if artefactErr != nil {
		return cfg, artefactErr
	}

	commandWhitelist, commandErr := requireCommandList(provider, "MCP_"+team+"_COMMANDS")
	if commandErr != nil {
		return cfg, commandErr
	}

	cfg = TeamConfig{
		ChatCollection:     chatCollection,
		ArtefactCollection: artefactCollection,
		CommandWhitelist:   commandWhitelist,
		VoteThreshold:      defaultVoteThreshold,
	}

	return cfg, nil
}

func loadMCPConfig(provider EnvironmentProvider) (MCPConfig, error) {
	var cfg MCPConfig

	hostValue, hostErr := requireStringEnv(provider, "MCP_MCP_HOST")
	if hostErr != nil {
		return cfg, hostErr
	}

	portValue, portErr := parseRequiredInt(provider, "MCP_MCP_PORT")
	if portErr != nil {
		return cfg, portErr
	}

	rateLimitValue, rateLimitErr := parseRequiredInt(provider, "MCP_RATE_LIMIT_PER_MINUTE")
	if rateLimitErr != nil {
		return cfg, rateLimitErr
	}

	dryRunValue, dryRunErr := parseBoolWithDefault(provider, "MCP_DRY_RUN", false)
	if dryRunErr != nil {
		return cfg, dryRunErr
	}

	cfg = MCPConfig{
		Host:               hostValue,
		Port:               portValue,
		RateLimitPerMinute: rateLimitValue,
		DryRun:             dryRunValue,
	}

	return cfg, nil
}

func requireStringEnv(provider EnvironmentProvider, env string) (string, error) {
	value := strings.TrimSpace(provider.Get(env))
	if value == "" {
		return "", fmt.Errorf("%w: %s", ErrMissingEnvironmentVariable, env)
	}

	return value, nil
}

func requireCommandList(provider EnvironmentProvider, env string) ([]string, error) {
	rawValue := provider.Get(env)
	if strings.TrimSpace(rawValue) == "" {
		return nil, fmt.Errorf("%w: %s", ErrEmptyCommandList, env)
	}

	segments := strings.Split(rawValue, ",")

	commands := make([]string, 0, len(segments))
	for _, segment := range segments {
		trimmed := strings.TrimSpace(segment)
		if trimmed == "" {
			continue
		}

		commands = append(commands, trimmed)
	}

	if len(commands) == 0 {
		return nil, fmt.Errorf("%w: %s", ErrEmptyCommandList, env)
	}

	return commands, nil
}

func parseIntWithDefault(provider EnvironmentProvider, env string, defaultValue int) (int, error) {
	value := strings.TrimSpace(provider.Get(env))
	if value == "" {
		return defaultValue, nil
	}

	parsed, parseErr := strconv.Atoi(value)
	if parseErr != nil {
		return 0, fmt.Errorf("%w: %s", ErrInvalidInteger, env)
	}

	return parsed, nil
}

func parseRequiredInt(provider EnvironmentProvider, env string) (int, error) {
	value, valueErr := requireStringEnv(provider, env)
	if valueErr != nil {
		return 0, valueErr
	}

	parsed, parseErr := strconv.Atoi(value)
	if parseErr != nil {
		return 0, fmt.Errorf("%w: %s", ErrInvalidInteger, env)
	}

	return parsed, nil
}

func parseBoolWithDefault(provider EnvironmentProvider, env string, defaultValue bool) (bool, error) {
	value := strings.TrimSpace(provider.Get(env))
	if value == "" {
		return defaultValue, nil
	}

	switch strings.ToLower(value) {
	case "true", "1", "yes":
		return true, nil
	case "false", "0", "no":
		return false, nil
	default:
		return false, fmt.Errorf("%w: %s", ErrInvalidBoolean, env)
	}
}
