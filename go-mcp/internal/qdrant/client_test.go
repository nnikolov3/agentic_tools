// Integration-style tests for the Qdrant HTTP client adapter.
package qdrant_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/internal/qdrant"
	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

func TestNewClientRejectsEmptyBaseURL(t *testing.T) {
	t.Parallel()

	_, clientErr := qdrant.NewClient("", "secret-token", time.Second)
	require.ErrorIs(t, clientErr, qdrant.ErrClientInvalidBaseURL)
}

func TestEnsureCollectionIssuesPutRequest(t *testing.T) {
	t.Parallel()

	testServer, captureChannel := startEnsureCollectionServer()
	defer testServer.Close()

	client, clientErr := qdrant.NewClient(testServer.URL, "secret-token", time.Second)
	require.NoError(t, clientErr)

	executionContext, cancelFunc := context.WithTimeout(context.Background(), time.Second)
	defer cancelFunc()

	ensureErr := client.EnsureCollection(executionContext, "team_chat_development")
	require.NoError(t, ensureErr)

	captured := waitForEnsureCollectionCapture(t, captureChannel)
	require.Equal(t, http.MethodPut, captured.Method)
	require.Equal(t, "/collections/team_chat_development", captured.Path)
	require.Equal(t, "secret-token", captured.APIKey)
	require.NoError(t, captured.BodyCloseErr)
}

func TestEnsureCollectionHandlesNonSuccessStatus(t *testing.T) {
	t.Parallel()

	testServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	defer testServer.Close()

	client, clientErr := qdrant.NewClient(testServer.URL, "secret-token", time.Second)
	require.NoError(t, clientErr)

	executionContext, cancelFunc := context.WithTimeout(context.Background(), time.Second)
	defer cancelFunc()

	ensureErr := client.EnsureCollection(executionContext, "team_chat_development")
	require.ErrorIs(t, ensureErr, qdrant.ErrClientRequestFailed)
}

func TestStoreTeamChatMessageIssuesPostRequest(t *testing.T) {
	t.Parallel()

	testServer, captureChannel := startUpsertServer()
	defer testServer.Close()

	client, clientErr := qdrant.NewClient(testServer.URL, "secret-token", time.Second)
	require.NoError(t, clientErr)

	executionContext, cancelFunc := context.WithTimeout(context.Background(), time.Second)
	defer cancelFunc()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Unix(1_700_000_000, 0).UTC(),
		Metadata:  map[string]string{"thread": "planning"},
	}

	storeErr := client.StoreTeamChatMessage(executionContext, "team_chat_development", chatMessage)
	require.NoError(t, storeErr)

	captured := waitForUpsertCapture(t, captureChannel)
	require.Equal(t, http.MethodPost, captured.Method)
	require.Equal(t, "/collections/team_chat_development/points", captured.Path)
	require.Equal(t, "secret-token", captured.APIKey)
	require.NoError(t, captured.DecodeErr)
	require.NoError(t, captured.BodyCloseErr)
	require.Len(t, captured.Body.Points, 1)
	require.Equal(t, "message-123", captured.Body.Points[0].ID)
	require.Equal(t, "Work packet ready for review.", captured.Body.Points[0].Payload["content"])
}

func TestStoreTeamChatMessageHandlesNonSuccessStatus(t *testing.T) {
	t.Parallel()

	testServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusBadRequest)
	}))
	defer testServer.Close()

	client, clientErr := qdrant.NewClient(testServer.URL, "secret-token", time.Second)
	require.NoError(t, clientErr)

	executionContext, cancelFunc := context.WithTimeout(context.Background(), time.Second)
	defer cancelFunc()

	chatMessage := model.TeamChatMessage{
		MessageID: "message-123",
		Team:      model.TeamNameDevelopment,
		Sender:    "agent-alpha",
		Role:      model.ChatMessageRoleCoordinator,
		Content:   "Work packet ready for review.",
		CreatedAt: time.Now(),
		Metadata:  nil,
	}

	storeErr := client.StoreTeamChatMessage(executionContext, "team_chat_development", chatMessage)
	require.ErrorIs(t, storeErr, qdrant.ErrClientRequestFailed)
}

type ensureCollectionCapture struct {
	Method       string
	Path         string
	APIKey       string
	BodyCloseErr error
}

type upsertCapture struct {
	Method       string
	Path         string
	APIKey       string
	Body         qdrantUpsertRequest
	DecodeErr    error
	BodyCloseErr error
}

type qdrantUpsertRequest struct {
	Points []qdrantUpsertPoint `json:"points"`
}

type qdrantUpsertPoint struct {
	ID      string                 `json:"id"`
	Payload map[string]interface{} `json:"payload"`
}

func startEnsureCollectionServer() (*httptest.Server, <-chan ensureCollectionCapture) {
	captureChannel := make(chan ensureCollectionCapture, 1)

	testServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		closeErr := request.Body.Close()

		captureChannel <- ensureCollectionCapture{
			Method:       request.Method,
			Path:         request.URL.EscapedPath(),
			APIKey:       request.Header.Get("Api-Key"),
			BodyCloseErr: closeErr,
		}

		writer.WriteHeader(http.StatusOK)
	}))

	return testServer, captureChannel
}

func startUpsertServer() (*httptest.Server, <-chan upsertCapture) {
	captureChannel := make(chan upsertCapture, 1)

	testServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var upsertBody qdrantUpsertRequest

		decodeErr := json.NewDecoder(request.Body).Decode(&upsertBody)
		closeErr := request.Body.Close()

		captureChannel <- upsertCapture{
			Method:       request.Method,
			Path:         request.URL.EscapedPath(),
			APIKey:       request.Header.Get("Api-Key"),
			Body:         upsertBody,
			DecodeErr:    decodeErr,
			BodyCloseErr: closeErr,
		}

		if decodeErr != nil || closeErr != nil {
			writer.WriteHeader(http.StatusInternalServerError)

			return
		}

		writer.WriteHeader(http.StatusOK)
	}))

	return testServer, captureChannel
}

func waitForEnsureCollectionCapture(
	t *testing.T,
	captureChannel <-chan ensureCollectionCapture,
) ensureCollectionCapture {
	t.Helper()

	select {
	case captured := <-captureChannel:
		return captured
	case <-time.After(2 * time.Second):
		t.Fatalf("no ensure collection request captured")
	}

	return ensureCollectionCapture{
		Method:       "",
		Path:         "",
		APIKey:       "",
		BodyCloseErr: nil,
	}
}

func waitForUpsertCapture(t *testing.T, captureChannel <-chan upsertCapture) upsertCapture {
	t.Helper()

	select {
	case captured := <-captureChannel:
		return captured
	case <-time.After(2 * time.Second):
		t.Fatalf("no upsert request captured")
	}

	return upsertCapture{
		Method:       "",
		Path:         "",
		APIKey:       "",
		Body:         qdrantUpsertRequest{Points: nil},
		DecodeErr:    nil,
		BodyCloseErr: nil,
	}
}
