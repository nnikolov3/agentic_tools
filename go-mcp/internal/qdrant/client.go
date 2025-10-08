// Package qdrant provides a thin HTTP adapter around the Qdrant REST API.
package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/nnikolov3/multi-agent-mcp/go-mcp/pkg/model"
)

const (
	defaultVectorSize     = 1536
	defaultVectorDistance = "Cosine"
)

var (
	// ErrClientInvalidBaseURL indicates that the supplied base URL is empty or malformed.
	ErrClientInvalidBaseURL = errors.New("qdrant client base url is invalid")
	// ErrClientRequestFailed indicates that an HTTP request could not be completed successfully.
	ErrClientRequestFailed = errors.New("qdrant client request failed")
	// ErrClientInvalidChatMessage indicates that the provided chat message failed validation.
	ErrClientInvalidChatMessage = errors.New("qdrant client chat message is invalid")
)

// Client provides typed helpers for interacting with Qdrant.
type Client struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

// NewClient constructs a Qdrant HTTP client using the provided configuration.
func NewClient(baseURL string, apiKey string, timeout time.Duration) (*Client, error) {
	trimmedBaseURL := strings.TrimSpace(baseURL)
	if trimmedBaseURL == "" {
		return nil, ErrClientInvalidBaseURL
	}

	parsedURL, parseErr := url.Parse(trimmedBaseURL)
	if parseErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientInvalidBaseURL, parseErr)
	}

	normalisedBaseURL := strings.TrimRight(parsedURL.String(), "/")

	httpClientInstance := new(http.Client)
	httpClientInstance.Timeout = timeout

	return &Client{
		httpClient: httpClientInstance,
		baseURL:    normalisedBaseURL,
		apiKey:     apiKey,
	}, nil
}

// EnsureCollection creates or updates a Qdrant collection using a default vector configuration.
func (client *Client) EnsureCollection(ctx context.Context, collectionName string) error {
	requestBody := collectionSchemaRequest{
		Vectors: vectorConfiguration{
			Size:     defaultVectorSize,
			Distance: defaultVectorDistance,
		},
	}

	collectionPath := "/collections/" + url.PathEscape(collectionName)

	return client.execute(ctx, http.MethodPut, collectionPath, requestBody)
}

// StoreTeamChatMessage persists a chat message in the specified collection.
func (client *Client) StoreTeamChatMessage(
	ctx context.Context,
	collectionName string,
	chatMessage model.TeamChatMessage,
) error {
	validationErr := chatMessage.Validate()
	if validationErr != nil {
		return fmt.Errorf("%w: %w", ErrClientInvalidChatMessage, validationErr)
	}

	payload := map[string]interface{}{
		"message_id": chatMessage.MessageID,
		"team":       string(chatMessage.Team),
		"sender":     chatMessage.Sender,
		"role":       string(chatMessage.Role),
		"content":    chatMessage.Content,
		"created_at": chatMessage.CreatedAt.Format(time.RFC3339Nano),
	}

	if len(chatMessage.Metadata) > 0 {
		payload["metadata"] = chatMessage.Metadata
	}

	requestBody := upsertPointsRequest{
		Points: []upsertPoint{
			{
				ID:      chatMessage.MessageID,
				Payload: payload,
			},
		},
	}

	pointsPath := fmt.Sprintf("/collections/%s/points", url.PathEscape(collectionName))

	return client.execute(ctx, http.MethodPost, pointsPath, requestBody)
}

func (client *Client) execute(ctx context.Context, method string, path string, requestBody interface{}) error {
	bodyReader, encodeErr := encodeRequestBody(requestBody)
	if encodeErr != nil {
		return encodeErr
	}

	request, requestErr := client.buildRequest(ctx, method, path, bodyReader)
	if requestErr != nil {
		return requestErr
	}

	response, responseErr := client.doRequest(request)
	if responseErr != nil {
		return responseErr
	}

	statusErr := validateResponseStatus(response)
	if statusErr != nil {
		drainErr := consumeResponseBody(response)
		if drainErr != nil {
			return drainErr
		}

		return statusErr
	}

	consumeErr := consumeResponseBody(response)
	if consumeErr != nil {
		return consumeErr
	}

	return nil
}

func encodeRequestBody(requestBody interface{}) (io.Reader, error) {
	if requestBody == nil {
		return bytes.NewBuffer(nil), nil
	}

	bodyBuffer := &bytes.Buffer{}
	encoder := json.NewEncoder(bodyBuffer)

	encodeErr := encoder.Encode(requestBody)
	if encodeErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientRequestFailed, encodeErr)
	}

	return bodyBuffer, nil
}

func (client *Client) buildRequest(
	ctx context.Context,
	method string,
	path string,
	body io.Reader,
) (*http.Request, error) {
	fullURL, joinErr := url.JoinPath(client.baseURL, path)
	if joinErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientRequestFailed, joinErr)
	}

	request, newRequestErr := http.NewRequestWithContext(ctx, method, fullURL, body)
	if newRequestErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientRequestFailed, newRequestErr)
	}

	client.applyHeaders(request)

	return request, nil
}

func (client *Client) applyHeaders(request *http.Request) {
	request.Header.Set("Content-Type", "application/json")

	if strings.TrimSpace(client.apiKey) != "" {
		request.Header.Set("Api-Key", client.apiKey)
	}
}

func (client *Client) doRequest(request *http.Request) (*http.Response, error) {
	response, httpErr := client.httpClient.Do(request)
	if httpErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientRequestFailed, httpErr)
	}

	return response, nil
}

func validateResponseStatus(response *http.Response) error {
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusBadRequest {
		return fmt.Errorf("%w: status %s", ErrClientRequestFailed, response.Status)
	}

	return nil
}

func consumeResponseBody(response *http.Response) error {
	_, discardErr := io.Copy(io.Discard, response.Body)
	if discardErr != nil {
		return fmt.Errorf("%w: %w", ErrClientRequestFailed, discardErr)
	}

	closeErr := response.Body.Close()
	if closeErr != nil {
		return fmt.Errorf("%w: %w", ErrClientRequestFailed, closeErr)
	}

	return nil
}

type vectorConfiguration struct {
	Size     int    `json:"size"`
	Distance string `json:"distance"`
}

type collectionSchemaRequest struct {
	Vectors vectorConfiguration `json:"vectors"`
}

type upsertPointsRequest struct {
	Points []upsertPoint `json:"points"`
}

type upsertPoint struct {
	ID      string                 `json:"id"`
	Payload map[string]interface{} `json:"payload"`
}
