# QDRANT Integration Reference

## Overview

Qdrant is a vector search engine designed to store, index, and search high-dimensional vectors efficiently. It enables semantic search and similarity matching on large datasets, making it ideal for AI applications, recommendation systems, and semantic memory storage.

## Architecture and Components

### Qdrant Server
- **Docker Image**: `qdrant/qdrant`
- **Main Ports**:
  - HTTP API: 6333
  - gRPC API: 6334 (if enabled)
- **Storage**: Persistent or in-memory modes
- **Clustering**: Supports distributed mode for scalability

### Qdrant Python Client
- **Installation**: `pip install qdrant-client`
- **Features**:
  - Type hints for all API methods
  - Local mode without running server
  - REST and gRPC support
  - Minimal dependencies
  - Both sync and async APIs

### Qdrant MCP Server
- **Purpose**: Semantic memory layer for LLM applications via Model Context Protocol
- **Provides**: `qdrant-store` and `qdrant-find` tools
- **Integration**: MCP-compatible clients (Claude, Cursor, VS Code, etc.)

## Client Installation and Connection

### Installation
```bash
# Basic client
pip install qdrant-client

# With local inference (FastEmbed)
pip install qdrant-client[fastembed]

# With GPU support for FastEmbed
pip install 'qdrant-client[fastembed-gpu]'
```

### Connection Methods

#### Local Mode (In-Memory or Persistent)
```python
from qdrant_client import QdrantClient

# In-memory storage
client = QdrantClient(":memory:")

# Persistent storage
client = QdrantClient(path="path/to/db")
```

#### Connect to Qdrant Server
```python
from qdrant_client import QdrantClient

# Using host and port
client = QdrantClient(host="localhost", port=6333)

# Using URL
client = QdrantClient(url="http://localhost:6333")
```

#### Connect to Qdrant Cloud
```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333",
    api_key="<your-api-key>",
)
```

## Core Concepts

### Collections
- Named containers for vector points
- Each collection has specific vector dimensions and distance metrics
- Support for multiple named vector configurations

### Points
- Individual vector entries with ID, vector, and optional payload
- Payloads are JSON-like key-value pairs for metadata
- Vector can be dense, sparse, or multi-vector

### Distance Metrics
- **Cosine**: Cosine similarity
- **Euclid**: Euclidean distance
- **Dot**: Dot product
- **Manhattan**: Manhattan distance

### Payload Indexing
- **Keyword**: Text field indexing
- **Integer**: Numeric field indexing
- **Float**: Float field indexing
- **Geo**: Geospatial indexing
- **Text**: Full-text search indexing
- **Bool**: Boolean field indexing
- **Datetime**: Date/time indexing
- **UUID**: UUID indexing

## Core API Operations

### Collection Management
```python
from qdrant_client.models import Distance, VectorParams

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)
```

### Vector Operations
```python
import numpy as np
from qdrant_client.models import PointStruct

vectors = np.random.rand(100, 100)

# Insert vectors
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]
)

# Upload large datasets (recommended for bulk operations)
client.upload_collection(
    collection_name="my_collection",
    vectors=vectors,
    payload=payload
)
```

### Search Operations
```python
# Basic search
query_vector = np.random.rand(100)
hits = client.query_points(
    collection_name="my_collection",
    query=query_vector,
    limit=5  # Return 5 closest points
)
```

### Filtered Search
```python
from qdrant_client.models import Filter, FieldCondition, Range

hits = client.query_points(
    collection_name="my_collection",
    query=query_vector,
    query_filter=Filter(
        must=[  # These conditions are required for search results
            FieldCondition(
                key='rand_number',  # Condition based on values of `rand_number` field
                range=Range(
                    gte=3  # Select only those results where `rand_number` >= 3
                )
            )
        ]
    ),
    limit=5  # Return 5 closest points
)
```

## Advanced Features

### Multiple Named Vectors
```python
from qdrant_client.models import VectorParams, VectorParamsDiff

client.create_collection(
    collection_name="multi_vectors",
    vectors_config={
        "image": VectorParams(size=512, distance=Distance.COSINE),
        "text": VectorParams(size=128, distance=Distance.COSINE),
    }
)
```

### Sparse Vectors
```python
from qdrant_client.models import SparseVectorParams, SparseVector

client.create_collection(
    collection_name="sparse_collection",
    vectors_config=SparseVectorParams(),
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    }
)
```

### Payload Operations
- **SetPayload**: Add or update specific payload keys
- **OverwritePayload**: Replace entire payload
- **DeletePayload**: Remove specific payload keys
- **ClearPayload**: Remove all payload

### Count Operations
```python
from qdrant_client.models import Filter, FieldCondition

# Count points matching filter
count = client.count(
    collection_name="my_collection",
    count_filter=Filter(
        must=[
            FieldCondition(
                key="color",
                match={"value": "red"}
            )
        ]
    )
)
```

## gRPC vs HTTP

### Enable gRPC for Better Performance
```python
from qdrant_client import QdrantClient

# Enable gRPC for faster operations
client = QdrantClient(
    host="localhost", 
    grpc_port=6334, 
    prefer_grpc=True
)
```

## Async Support

### Async Client Usage
```python
import asyncio
import numpy as np
from qdrant_client import AsyncQdrantClient, models

async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")

    await client.create_collection(
        collection_name="my_collection",
        vectors_config=models.VectorParams(size=10, distance=models.Distance.COSINE),
    )

    await client.upsert(
        collection_name="my_collection",
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(10).tolist(),
            )
            for i in range(100)
        ],
    )

    res = await client.query_points(
        collection_name="my_collection",
        query=np.random.rand(10).tolist(),
        limit=10,
    )

    print(res)

asyncio.run(main())
```

## Local Inference (FastEmbed)

### With Local Embeddings
```python
from qdrant_client import QdrantClient, models

# Local mode with FastEmbed
client = QdrantClient(":memory:")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
payload = [
    {"document": "Qdrant has Langchain integrations", "source": "Langchain-docs", },
    {"document": "Qdrant also has Llama Index integrations", "source": "LlamaIndex-docs"},
]
docs = [models.Document(text=data["document"], model=model_name) for data in payload]
ids = [42, 2]

client.create_collection(
    "demo_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), distance=models.Distance.COSINE)
)

client.upload_collection(
    collection_name="demo_collection",
    vectors=docs,
    ids=ids,
    payload=payload,
)

search_result = client.query_points(
    collection_name="demo_collection",
    query=models.Document(text="This is a query document", model=model_name)
).points
print(search_result)
```

## Model Context Protocol (MCP) Integration

### Qdrant MCP Server Tools

#### qdrant-store
- **Purpose**: Store information in the Qdrant database
- **Input Parameters**:
  - `information` (string): Information to store
  - `metadata` (JSON): Optional metadata to store
  - `collection_name` (string): Name of the collection to store the information in
- **Returns**: Confirmation message

#### qdrant-find
- **Purpose**: Retrieve relevant information from the Qdrant database
- **Input Parameters**:
  - `query` (string): Query to use for searching
  - `collection_name` (string): Name of the collection to search in
- **Returns**: Information stored in the Qdrant database as separate messages

### Configuration Environment Variables
- `QDRANT_URL`: URL of the Qdrant server
- `QDRANT_API_KEY`: API key for the Qdrant server
- `QDRANT_LOCAL_PATH`: Path to the local Qdrant database
- `COLLECTION_NAME`: Name of the default collection to use
- `EMBEDDING_PROVIDER`: Embedding provider (\"fastembed\" supported)
- `EMBEDDING_MODEL`: Name of the embedding model

### Transport Protocols
- **stdio** (default): Standard input/output transport
- **sse**: Server-Sent Events transport
- **streamable-http**: Streamable HTTP transport

## Deployment Options

### Docker
```bash
# Basic run
docker run -p 6333:6333 qdrant/qdrant

# With persistent storage
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Docker Compose
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT_API_KEY=your_api_key

volumes:
  qdrant_storage:
```

### Configuration Options
- Environment variables for authentication, clustering, telemetry
- Custom configuration file mounting
- Health checks available at `/` and `/readyz`

## Use Cases for Multi-Agent MCP Integration

1. **Semantic Memory**: Store and retrieve conversation context across agent sessions
2. **Code Search**: Store code snippets with embeddings for semantic search
3. **Documentation Retrieval**: Semantic search through project documentation
4. **Agent History**: Store and query agent decision history
5. **Knowledge Base**: Maintain searchable knowledge from design documents and principles
6. **Context Assembly**: Enhanced context gathering using vector similarity