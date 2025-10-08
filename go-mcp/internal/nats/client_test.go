// Package nats_test exercises the messaging adapter against an embedded NATS server.
package nats_test

import (
	"context"
	"sync"
	"testing"
	"time"

	natsserver "github.com/nats-io/nats-server/v2/server"
	"github.com/stretchr/testify/require"

	internalnats "github.com/nnikolov3/multi-agent-mcp/go-mcp/internal/nats"
)

func TestConnectEstablishesConnection(t *testing.T) {
	t.Parallel()

	embeddedServer := startEmbeddedServer(t)
	defer embeddedServer.Shutdown()

	client, connectErr := internalnats.Connect(embeddedServer.ClientURL(), time.Second)
	require.NoError(t, connectErr)

	t.Cleanup(func() {
		closeErr := client.Close()
		require.NoError(t, closeErr)
	})
}

func TestPublishAndSubscribeDeliversMessage(t *testing.T) {
	t.Parallel()

	embeddedServer := startEmbeddedServer(t)
	defer embeddedServer.Shutdown()

	client, connectErr := internalnats.Connect(embeddedServer.ClientURL(), time.Second)
	require.NoError(t, connectErr)

	t.Cleanup(func() {
		closeErr := client.Close()
		require.NoError(t, closeErr)
	})

	messageReceived := make(chan internalnats.Message, 1)

	subscriptionContext, cancelSubscription := context.WithCancel(context.Background())
	defer cancelSubscription()

	subscription, subscribeErr := client.Subscribe(
		subscriptionContext,
		"team.development.work",
		func(_ context.Context, receivedMessage internalnats.Message) error {
			messageReceived <- receivedMessage

			return nil
		},
	)
	require.NoError(t, subscribeErr)

	t.Cleanup(func() {
		closeErr := subscription.Close()
		require.NoError(t, closeErr)
	})

	publishErr := client.Publish(context.Background(), "team.development.work", []byte("implement config loader"))
	require.NoError(t, publishErr)

	select {
	case received := <-messageReceived:
		require.Equal(t, "team.development.work", received.Subject)
		require.Equal(t, []byte("implement config loader"), received.Data)
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for message delivery")
	}
}

func TestSubscribeRejectsNilHandler(t *testing.T) {
	t.Parallel()

	embeddedServer := startEmbeddedServer(t)
	defer embeddedServer.Shutdown()

	client, connectErr := internalnats.Connect(embeddedServer.ClientURL(), time.Second)
	require.NoError(t, connectErr)

	t.Cleanup(func() {
		closeErr := client.Close()
		require.NoError(t, closeErr)
	})

	_, subscribeErr := client.Subscribe(
		context.Background(),
		"team.development.work",
		nil,
	)
	require.ErrorIs(t, subscribeErr, internalnats.ErrClientMissingHandler)
}

func TestSubscribeCancelsOnContextDone(t *testing.T) {
	t.Parallel()

	embeddedServer := startEmbeddedServer(t)
	defer embeddedServer.Shutdown()

	client, connectErr := internalnats.Connect(embeddedServer.ClientURL(), time.Second)
	require.NoError(t, connectErr)

	t.Cleanup(func() {
		closeErr := client.Close()
		require.NoError(t, closeErr)
	})

	subscriptionContext, cancelSubscription := context.WithCancel(context.Background())
	t.Cleanup(cancelSubscription)

	var handlerInvocation sync.WaitGroup
	handlerInvocation.Add(1)

	subscription, subscribeErr := client.Subscribe(
		subscriptionContext,
		"team.development.work",
		func(_ context.Context, _ internalnats.Message) error {
			defer handlerInvocation.Done()

			return nil
		},
	)
	require.NoError(t, subscribeErr)

	t.Cleanup(func() {
		closeErr := subscription.Close()
		require.NoError(t, closeErr)
	})

	publishErr := client.Publish(context.Background(), "team.development.work", []byte("iteration check"))
	require.NoError(t, publishErr)

	handlerInvocation.Wait()
	cancelSubscription()
}

func startEmbeddedServer(t *testing.T) *natsserver.Server {
	t.Helper()

	configuration := new(natsserver.Options)
	configuration.Host = "127.0.0.1"
	configuration.Port = -1
	configuration.NoSystemAccount = true

	embeddedServer, serverErr := natsserver.NewServer(configuration)
	require.NoError(t, serverErr)

	go embeddedServer.Start()

	if !embeddedServer.ReadyForConnections(5 * time.Second) {
		embeddedServer.Shutdown()
		t.Fatalf("embedded NATS server failed to start")
	}

	return embeddedServer
}
