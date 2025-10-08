// Package nats provides convenience wrappers around the nats.go client tailored for the MCP services.
package nats

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
)

const defaultFlushTimeout = 5 * time.Second

var (
	// ErrClientMissingHandler indicates that a subscription attempted to register a nil handler.
	ErrClientMissingHandler = errors.New("nats client handler cannot be nil")
	// ErrClientInvalidURL indicates that the provided server URL was empty.
	ErrClientInvalidURL = errors.New("nats client server url cannot be empty")
	// ErrClientConnectionFailed indicates that establishing the underlying connection failed.
	ErrClientConnectionFailed = errors.New("nats client failed to connect to server")
	// ErrClientSubscriptionFailed indicates that creating a subscription failed.
	ErrClientSubscriptionFailed = errors.New("nats client failed to subscribe to subject")
	// ErrClientSubscriptionCloseFailed indicates that closing a subscription failed.
	ErrClientSubscriptionCloseFailed = errors.New("nats client failed to close subscription")
	// ErrClientPublishFailed indicates that sending a message to the broker failed.
	ErrClientPublishFailed = errors.New("nats client failed to publish message")
	// ErrClientClosed indicates that an operation was attempted on a closed client.
	ErrClientClosed = errors.New("nats client connection is closed")
)

// Message represents a lightweight wrapper over a NATS message payload.
type Message struct {
	Subject string
	Data    []byte
	Headers nats.Header
}

// Handler defines the signature for subscription callbacks.
type Handler func(context.Context, Message) error

// Client manages the lifecycle of a single NATS connection.
type Client struct {
	connection   *nats.Conn
	connectionMu sync.RWMutex
}

// Connect establishes a NATS connection using the provided server URL and timeout.
func Connect(serverURL string, timeout time.Duration) (*Client, error) {
	if serverURL == "" {
		return nil, ErrClientInvalidURL
	}

	connectOptions := []nats.Option{
		nats.Timeout(timeout),
		nats.Name("multi-agent-mcp"),
	}

	connection, connectionErr := nats.Connect(serverURL, connectOptions...)
	if connectionErr != nil {
		return nil, fmt.Errorf("%w: %w", ErrClientConnectionFailed, connectionErr)
	}

	return &Client{
		connection:   connection,
		connectionMu: sync.RWMutex{},
	}, nil
}

// Publish sends the provided payload to the given subject and flushes the connection to guarantee delivery.
func (client *Client) Publish(ctx context.Context, subject string, payload []byte) error {
	activeConnection, connectionErr := client.getActiveConnection()
	if connectionErr != nil {
		return connectionErr
	}

	if ctx.Err() != nil {
		return fmt.Errorf("%w: %w", ErrClientPublishFailed, ctx.Err())
	}

	publishErr := activeConnection.Publish(subject, payload)
	if publishErr != nil {
		return fmt.Errorf("%w: %w", ErrClientPublishFailed, publishErr)
	}

	flushCtx, flushCancel := context.WithTimeout(ctx, defaultFlushTimeout)
	defer flushCancel()

	flushErr := activeConnection.FlushWithContext(flushCtx)
	if flushErr != nil {
		return fmt.Errorf("%w: %w", ErrClientPublishFailed, flushErr)
	}

	return nil
}

// Subscribe registers a handler for the supplied subject. The subscription is cancelled when the context ends.
func (client *Client) Subscribe(ctx context.Context, subject string, handler Handler) (*Subscription, error) {
	if handler == nil {
		return nil, ErrClientMissingHandler
	}

	activeConnection, connectionErr := client.getActiveConnection()
	if connectionErr != nil {
		return nil, connectionErr
	}

	internalCtx, internalCancel := context.WithCancel(context.Background())

	natsSubscription, subscribeErr := activeConnection.Subscribe(subject, func(message *nats.Msg) {
		select {
		case <-ctx.Done():
			return
		case <-internalCtx.Done():
			return
		default:
		}

		messageCopy := Message{
			Subject: message.Subject,
			Data:    append([]byte(nil), message.Data...),
			Headers: message.Header,
		}

		handlerErr := handler(ctx, messageCopy)
		if handlerErr != nil {
			return
		}
	})
	if subscribeErr != nil {
		internalCancel()

		return nil, fmt.Errorf("%w: %w", ErrClientSubscriptionFailed, subscribeErr)
	}

	go func() {
		select {
		case <-ctx.Done():
			internalCancel()
		case <-internalCtx.Done():
		}
	}()

	return &Subscription{
		subscription: natsSubscription,
		cancel:       internalCancel,
		closeOnce:    sync.Once{},
	}, nil
}

// Close terminates the underlying NATS connection.
func (client *Client) Close() error {
	client.connectionMu.Lock()
	defer client.connectionMu.Unlock()

	if client.connection == nil {
		return nil
	}

	if client.connection.IsClosed() {
		client.connection = nil

		return nil
	}

	client.connection.Close()
	client.connection = nil

	return nil
}

// Subscription represents an active NATS subscription.
type Subscription struct {
	subscription *nats.Subscription
	cancel       context.CancelFunc
	closeOnce    sync.Once
}

// Close drains the subscription and releases associated resources.
func (subscription *Subscription) Close() error {
	var closeErr error

	subscription.closeOnce.Do(func() {
		subscription.cancel()

		closeErr = subscription.subscription.Drain()
		if closeErr != nil {
			closeErr = fmt.Errorf("%w: %w", ErrClientSubscriptionCloseFailed, closeErr)
		}
	})

	return closeErr
}

func (client *Client) getActiveConnection() (*nats.Conn, error) {
	client.connectionMu.RLock()
	defer client.connectionMu.RUnlock()

	if client.connection == nil || client.connection.IsClosed() {
		return nil, ErrClientClosed
	}

	return client.connection, nil
}
