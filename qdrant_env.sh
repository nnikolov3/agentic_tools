#!/bin/bash
# qdrant_env.sh: Export Qdrant VM optimizations for server startup.
# Run: source qdrant_env.sh && qdrant (or in Docker entrypoint).
# Tuned for high-throughput, disk-optimized single-node (NODE_TYPE='Normal').

# PERFORMANCE (Auto-threading, no limits)
export QDRANT__SERVICE__MAX_WORKERS=0                      # Use all available CPU cores
export QDRANT__PERFORMANCE__MAX_SEARCH_THREADS=0           # Auto: use all threads
export QDRANT__PERFORMANCE__OPTIMIZER_CPU_BUDGET=0         # Use system default, maximize throughput
export QDRANT__PERFORMANCE__UPDATE_RATE_LIMIT=0            # No artificial limit

# OPTIMIZER-FOCUSED PARAMETERS
export QDRANT__OPTIMIZERS__DELETED_THRESHOLD=0.1           # More frequent segment cleanup
export QDRANT__OPTIMIZERS__VACUUM_MIN_VECTOR_NUMBER=500    # Lower threshold, faster cleanup
export QDRANT__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER=0        # Auto: balance segments per CPU
export QDRANT__OPTIMIZERS__MAX_SEGMENT_SIZE_KB=65536       # Larger segment for better indexing speed
export QDRANT__OPTIMIZERS__INDEXING_THRESHOLD_KB=20000     # Index more data before merge
export QDRANT__OPTIMIZERS__FLUSH_INTERVAL_SEC=2            # Frequent flushing for fast writes
export QDRANT__OPTIMIZERS__MAX_OPTIMIZATION_THREADS=0      # Unlimited optimization threads

# STORAGE AND IO
export QDRANT__STORAGE__STORAGE_PATH='/qdrant/storage'
export QDRANT__STORAGE__SNAPSHOTS_PATH='/qdrant/snapshots'
export QDRANT__STORAGE__ON_DISK_PAYLOAD=1                  # Optimize RAM usage, payloads on disk
export QDRANT__STORAGE__NODE_TYPE='Normal'
export QDRANT__STORAGE__TEMP_PATH='/qdrant/tmp'
export QDRANT__STORAGE__UPDATE_CONCURRENCY=0               # Unlimited concurrent updates
export QDRANT__STORAGE__WAL__WAL_CAPACITY_MB=128           # Larger WAL for batch durability
export QDRANT__STORAGE__WAL__WAL_SEGMENTS_AHEAD=4          # Aggressively preallocate WAL segments

# HNSW INDEXING & SEARCH
export QDRANT__STORAGE__HNSW_INDEX__M=32                   # Dense graph, higher accuracy
export QDRANT__STORAGE__HNSW_INDEX__EF_CONSTRUCT=400       # Broad search, higher accuracy
export QDRANT__STORAGE__HNSW_INDEX__MAX_INDEXING_THREADS=16 # Maximize parallel index construction
export QDRANT__STORAGE__HNSW_INDEX__ON_DISK=1              # Store index on disk to save RAM

# SERVICE & API OPTIMIZATIONS
export QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64             # Allow large POST requests
export QDRANT__SERVICE__ENABLE_CORS=1                      # Enable CORS for multi-origin support
#export QDRANT__SERVICE__API_KEY='your_secret_key'          # Secure API access (uncomment if needed)

# OPTIONAL: SNAPSHOTS AND LOCAL MODE
export QDRANT__STORAGE__SNAPSHOTS_CONFIG__SNAPSHOTS_STORAGE='local'

# SECURITY & TELEMETRY
export QDRANT__TELEMETRY_DISABLED=1                        # Disable telemetry for privacy

# COLLECTION AND QUERY OPTIMIZATION
export QDRANT__COLLECTION__REPLICATION_FACTOR=1
export QDRANT__COLLECTION__WRITE_CONSISTENCY_FACTOR=1
export QDRANT__COLLECTION__VECTORS__ON_DISK=1              # Store vectors on disk
export QDRANT__COLLECTION__MAX_QUERY_LIMIT=10000           # High max query limit
export QDRANT__COLLECTION__MAX_TIMEOUT=60                  # Longer allowable timeouts

# STRICT MODE DISABLED FOR FLEXIBILITY
export QDRANT__STRICT_MODE__ENABLED=0

# SERVICE BINDING (VM-exposed ports)
export QDRANT__SERVICE__HOST='0.0.0.0'      # Listen on all IPs/interfaces
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
export QDRANT__CLUSTER__P2P__PORT=6335      # For potential clustering

echo "Qdrant VM env loaded: HTTP=6333, gRPC=6334, P2P=6335; Storage=/qdrant/storage; On-disk enabled."
# To start: qdrant --config-path /path/to/qdrant_config.yaml (if using YAML overrides)

