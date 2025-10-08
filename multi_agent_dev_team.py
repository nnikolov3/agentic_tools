"""

multi_agent_dev_team.py

Enterprise multi-agent orchestration with automatic project indexing.

Features:

- Scans project directory on startup

- Indexes all source files (.go, .py, .sh, .md) in Qdrant

- Tracks file timestamps for incremental updates

- 24+ specialized agents across 5 API providers

- Role-based access control

- Comprehensive learning from success/failure patterns

- GUARANTEED resource utilization for ALL API providers

Version: 2.1.0

Author: Nikolay Nikolov

Date: October 7, 2025

"""

import fnmatch
import hashlib
import json
import logging
import os
import random
import secrets
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import google.generativeai as genai_stable
import openai
from fastmcp import Context, FastMCP
from google import generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer

# ============================================================================
# Configuration
# ============================================================================

# Project root directory (auto-detect or set manually)
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())

# File patterns to index
INDEXABLE_PATTERNS = [
    "*.go", "*.py", "*.sh", "*.md", "*.yaml", "*.yml",
    "*.json", "*.toml", "*.txt", "*.rs", "*.c", "*.h"
]

# Directories to exclude
EXCLUDE_DIRS = [
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", "vendor", "target", "build", "dist"
]

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_dev_team.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# FastMCP Server
# ============================================================================

mcp = FastMCP(
    name="Multi-Agent Development Team",
    instructions=(
        "Enterprise development team with automatic project indexing, "
        "intelligent routing, and comprehensive memory."
    )
)

# ============================================================================
# Authentication System
# ============================================================================

class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    WRITER = "writer"
    VIEWER = "viewer"

class Permission(Enum):
    """Permissions."""
    CALL_LEADERSHIP = "call_leadership"
    CALL_DEVELOPMENT = "call_development"
    CALL_QA = "call_qa"
    CALL_DOCS = "call_docs"
    CALL_DEVOPS = "call_devops"
    VIEW_TASKS = "view_tasks"
    CREATE_TASKS = "create_tasks"
    DELETE_TASKS = "delete_tasks"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"

ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.CALL_LEADERSHIP, Permission.CALL_DEVELOPMENT,
        Permission.CALL_QA, Permission.CALL_DOCS, Permission.CALL_DEVOPS,
        Permission.VIEW_TASKS, Permission.CREATE_TASKS, Permission.DELETE_TASKS,
        Permission.VIEW_ANALYTICS, Permission.MANAGE_USERS
    },
    Role.DEVELOPER: {
        Permission.CALL_DEVELOPMENT, Permission.CALL_QA, Permission.CALL_DOCS,
        Permission.VIEW_TASKS, Permission.CREATE_TASKS, Permission.VIEW_ANALYTICS
    },
    Role.TESTER: {
        Permission.CALL_QA, Permission.VIEW_TASKS, Permission.VIEW_ANALYTICS
    },
    Role.WRITER: {
        Permission.CALL_DOCS, Permission.VIEW_TASKS
    },
    Role.VIEWER: {
        Permission.VIEW_TASKS, Permission.VIEW_ANALYTICS
    }
}

AGENT_PERMISSIONS: Dict[str, Permission] = {
    "tech_lead": Permission.CALL_LEADERSHIP,
    "architect": Permission.CALL_LEADERSHIP,
    "principal_engineer": Permission.CALL_DEVELOPMENT,
    "senior_code_reviewer": Permission.CALL_DEVELOPMENT,
    "rapid_developer": Permission.CALL_DEVELOPMENT,
    "rapid_prototyper": Permission.CALL_DEVELOPMENT,
    "agentic_developer": Permission.CALL_DEVELOPMENT,
    "senior_developer": Permission.CALL_DEVELOPMENT,
    "parallel_developer": Permission.CALL_DEVELOPMENT,
    "code_specialist": Permission.CALL_DEVELOPMENT,
    "kimi_specialist": Permission.CALL_DEVELOPMENT,
    "llama4_maverick": Permission.CALL_DEVELOPMENT,
    "llama4_scout": Permission.CALL_DEVELOPMENT,
    "gpt_oss_120b": Permission.CALL_DEVELOPMENT,
    "qwen_researcher": Permission.CALL_DEVELOPMENT,
    "fast_qa": Permission.CALL_QA,
    "qa_engineer": Permission.CALL_QA,
    "security_auditor": Permission.CALL_QA,
    "technical_writer": Permission.CALL_DOCS,
    "diagram_specialist": Permission.CALL_DOCS,
    "git_specialist": Permission.CALL_DEVOPS,
    "devops_engineer": Permission.CALL_DEVOPS
}

@dataclass
class User:
    """User with role and API key."""
    user_id: str
    username: str
    role: Role
    api_key: str
    created_at: datetime
    last_access: Optional[datetime] = None
    is_active: bool = True

class AuthManager:
    """Authentication manager."""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}
        self._initialize_default_users()

    def _initialize_default_users(self):
        """Create default admin user."""
        admin_key = os.getenv("ADMIN_API_KEY")
        if not admin_key:
            admin_key = f"mcp_{secrets.token_urlsafe(32)}"

        admin_user = User(
            user_id="admin_1",
            username="admin",
            role=Role.ADMIN,
            api_key=admin_key,
            created_at=datetime.now()
        )

        self.users[admin_user.user_id] = admin_user
        self.api_keys[admin_key] = admin_user.user_id
        logger.info(f"🔑 Admin API Key: {admin_key}")

    def _generate_api_key(self) -> str:
        return f"mcp_{secrets.token_urlsafe(32)}"

    def create_user(self, username: str, role: Role, created_by: str) -> User:
        creator = self.users.get(created_by)
        if not creator or not self.has_permission(created_by, Permission.MANAGE_USERS):
            raise PermissionError("Only admins can create users")

        user_id = f"user_{secrets.token_hex(8)}"
        api_key = self._generate_api_key()
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            api_key=api_key,
            created_at=datetime.now()
        )

        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        logger.info(f"✅ Created user: {username} ({role.value})")
        return user

    def authenticate(self, api_key: str) -> Optional[User]:
        user_id = self.api_keys.get(api_key)
        if not user_id:
            return None
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None
        user.last_access = datetime.now()
        return user

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        return permission in ROLE_PERMISSIONS.get(user.role, set())

    def can_call_agent(self, user_id: str, agent: str) -> bool:
        required_permission = AGENT_PERMISSIONS.get(agent)
        if not required_permission:
            return False
        return self.has_permission(user_id, required_permission)

auth_manager = AuthManager()

# ============================================================================
# API Configuration
# ============================================================================

@dataclass
class APIConfig:
    """API configuration."""
    openai_key: str
    groq_key: str
    google_key: str
    gemini_key: str
    cerebras_personal: str
    cerebras_book_expert: str
    dashscope_key: str
    qdrant_url: str
    qdrant_key: str

    @classmethod
    def from_env(cls) -> 'APIConfig':
        config = {
            'openai_key': os.getenv('OPENAI_API_KEY', ''),
            'groq_key': os.getenv('GROQ_API_KEY', ''),
            'google_key': os.getenv('GOOGLE_API_KEY', ''),
            'gemini_key': os.getenv('GEMINI_API_KEY', ''),
            'cerebras_personal': os.getenv('CEREBRAS_API_KEY_PERSONAL', ''),
            'cerebras_book_expert': os.getenv('CEREBRAS_API_KEY_BOOK_EXPERT', ''),
            'dashscope_key': os.getenv('DASHSCOPE_API_KEY', ''),
            'qdrant_url': os.getenv('QDRANT_URL', ''),
            'qdrant_key': os.getenv('QDRANT_API_KEY', '')
        }

        required = ['openai_key', 'groq_key', 'google_key', 'gemini_key', 'qdrant_url', 'qdrant_key']
        missing = [k for k in required if not config[k]]
        if missing:
            logger.error(f"❌ Missing REQUIRED keys: {', '.join(missing)}")

        optional = ['cerebras_personal', 'cerebras_book_expert', 'dashscope_key']
        missing_optional = [k for k in optional if not config[k]]
        if missing_optional:
            logger.warning(f"⚠️ Missing OPTIONAL keys: {', '.join(missing_optional)}")

        return cls(**config)

config = APIConfig.from_env()

# ============================================================================
# API Clients
# ============================================================================

class APIClients:
    """API client manager with enhanced availability checks."""

    def __init__(self, config: APIConfig):
        self.config = config
        # UPDATED: Allow single-key operation with OR instead of AND
        self.cerebras_available = bool(config.cerebras_personal or config.cerebras_book_expert)
        self.qwen_available = bool(config.dashscope_key)
        self._setup_clients()

    def _setup_clients(self):
        try:
            self.openai = openai.OpenAI(api_key=self.config.openai_key)
            logger.info("✅ OpenAI client initialized")

            self.groq = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.config.groq_key
            )
            logger.info("✅ Groq client initialized")

            genai.configure(api_key=self.config.gemini_key)
            logger.info("✅ Google Gemini preview initialized")

            genai_stable.configure(api_key=self.config.google_key)
            logger.info("✅ Google Gemini stable initialized")

            # UPDATED: Initialize Cerebras clients with single-key support
            if self.cerebras_available:
                self.cerebras_clients = {}

                if self.config.cerebras_personal:
                    self.cerebras_clients["personal"] = {
                        "client": openai.OpenAI(
                            base_url="https://api.cerebras.ai/v1",
                            api_key=self.config.cerebras_personal
                        ),
                        "usage_count": 0
                    }

                if self.config.cerebras_book_expert:
                    self.cerebras_clients["book_expert"] = {
                        "client": openai.OpenAI(
                            base_url="https://api.cerebras.ai/v1",
                            api_key=self.config.cerebras_book_expert
                        ),
                        "usage_count": 0
                    }

                logger.info(f"✅ Cerebras clients initialized ({len(self.cerebras_clients)} key(s))")

            if self.qwen_available:
                self.qwen = openai.OpenAI(
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    api_key=self.config.dashscope_key
                )
                logger.info("✅ Qwen client initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise

    def get_cerebras_client(self) -> Optional[openai.OpenAI]:
        """Get Cerebras client with round-robin load balancing."""
        if not self.cerebras_available or not self.cerebras_clients:
            return None

        # If only one key, use it
        if len(self.cerebras_clients) == 1:
            key_name = list(self.cerebras_clients.keys())[0]
            client_info = self.cerebras_clients[key_name]
            client_info["usage_count"] += 1
            return client_info["client"]

        # Round-robin between multiple keys
        personal = self.cerebras_clients.get("personal")
        book_expert = self.cerebras_clients.get("book_expert")

        if personal and book_expert:
            if personal["usage_count"] <= book_expert["usage_count"]:
                personal["usage_count"] += 1
                return personal["client"]
            else:
                book_expert["usage_count"] += 1
                return book_expert["client"]
        elif personal:
            personal["usage_count"] += 1
            return personal["client"]
        elif book_expert:
            book_expert["usage_count"] += 1
            return book_expert["client"]

        return None

clients = APIClients(config)

# ============================================================================
# Team Configuration
# ============================================================================

@dataclass
class AgentConfig:
    model: str
    provider: str
    role: str
    temperature: float = 0.3
    max_tokens: int = 4000

# UPDATED: Always register ALL agents with intelligent fallbacks
TEAM: Dict[str, AgentConfig] = {
    # OpenAI Agents (Always available)
    "tech_lead": AgentConfig("gpt-4o", "openai", "Technical leadership", 0.1, 3000),
    "architect": AgentConfig("o1", "openai", "Deep reasoning", 0.1, 6000),
    "principal_engineer": AgentConfig("gpt-4o", "openai", "Implementation planning", 0.3, 8000),
    "senior_code_reviewer": AgentConfig("gpt-4o", "openai", "Code review", 0.2, 4000),

    # Groq Agents (Always available)
    "rapid_developer": AgentConfig("groq/compound", "groq", "Ultra-fast development", 0.5, 12000),
    "rapid_prototyper": AgentConfig("groq/compound-mini", "groq", "Quick prototypes", 0.5, 8000),
    "agentic_developer": AgentConfig("llama-3.3-70b-versatile", "groq", "Multi-step reasoning", 0.4, 10000),
    "fast_qa": AgentConfig("llama-3.1-8b-instant", "groq", "High-throughput testing", 0.2, 4000),
    "kimi_specialist": AgentConfig("moonshotai/kimi-k2-instruct-0905", "groq", "256k context", 0.4, 12000),
    "llama4_maverick": AgentConfig("meta-llama/llama-4-maverick-17b-128e-instruct", "groq", "Heavy-duty", 0.4, 10000),
    "llama4_scout": AgentConfig("meta-llama/llama-4-scout-17b-16e-instruct", "groq", "Complex planning", 0.4, 10000),
    "gpt_oss_120b": AgentConfig("openai/gpt-oss-120b", "groq", "Knowledge base", 0.3, 8000),
    "qwen_researcher": AgentConfig("qwen3-32b", "groq", "Research", 0.3, 6000),

    # Google Agents (Always available)
    "qa_engineer": AgentConfig("gemini-2.5-flash-002", "google_stable", "Testing", 0.2, 4000),
    "security_auditor": AgentConfig("gemini-2.5-flash-002", "google_stable", "Security", 0.2, 4000),
    "technical_writer": AgentConfig("gemini-2.5-pro-002", "google_stable", "Documentation", 0.3, 8000),
    "git_specialist": AgentConfig("gemini-2.5-flash-002", "google_stable", "Git workflows", 0.2, 2000),
    "devops_engineer": AgentConfig("gemini-2.5-flash-002", "google_stable", "CI/CD", 0.2, 4000),
    "diagram_specialist": AgentConfig("gemini-2.5-flash-preview-09-2025", "google_preview", "Mermaid diagrams", 0.3, 6000),

    # UPDATED: Cerebras agents with Groq fallback
    "senior_developer": AgentConfig(
        "llama-3.3-70b" if clients.cerebras_available else "llama-3.3-70b-versatile",
        "cerebras" if clients.cerebras_available else "groq",
        "Core development",
        0.5,
        12000
    ),
    "parallel_developer": AgentConfig(
        "llama-3.3-70b" if clients.cerebras_available else "llama-3.3-70b-versatile",
        "cerebras" if clients.cerebras_available else "groq",
        "Parallel development",
        0.5,
        12000
    ),

    # UPDATED: Qwen agent with Groq fallback
    "code_specialist": AgentConfig(
        "qwen-coder-plus" if clients.qwen_available else "qwen3-32b",
        "qwen" if clients.qwen_available else "groq",
        "Algorithm optimization",
        0.4,
        8000
    ),
}

logger.info(f"✅ Team configured with {len(TEAM)} agents")
logger.info(f"   - Cerebras agents: {'ACTIVE' if clients.cerebras_available else 'FALLBACK to Groq'}")
logger.info(f"   - Qwen agents: {'ACTIVE' if clients.qwen_available else 'FALLBACK to Groq'}")

# ============================================================================
# Project Scanner
# ============================================================================

class ProjectScanner:
    """Scans project directory and indexes files in Qdrant."""

    def __init__(self, root_dir: str, qdrant_manager):
        self.root_dir = Path(root_dir)
        self.qdrant = qdrant_manager
        self.indexed_files: Dict[str, float] = {}  # path -> timestamp

    def should_index(self, filepath: Path) -> bool:
        """Check if file should be indexed."""
        # Skip excluded directories
        for part in filepath.parts:
            if part in EXCLUDE_DIRS:
                return False

        # Check if matches indexable patterns
        for pattern in INDEXABLE_PATTERNS:
            if fnmatch.fnmatch(filepath.name, pattern):
                return True

        return False

    def scan_and_index(self):
        """Scan project directory and index new/modified files."""
        logger.info(f"📁 Scanning project: {self.root_dir}")
        files_indexed = 0
        files_skipped = 0
        files_updated = 0

        try:
            for filepath in self.root_dir.rglob('*'):
                if not filepath.is_file():
                    continue

                if not self.should_index(filepath):
                    continue

                try:
                    # Get file modification time
                    mtime = filepath.stat().st_mtime
                    relative_path = str(filepath.relative_to(self.root_dir))

                    # Check if already indexed and up-to-date
                    if relative_path in self.indexed_files:
                        if self.indexed_files[relative_path] >= mtime:
                            files_skipped += 1
                            continue
                        files_updated += 1
                    else:
                        files_indexed += 1

                    # Read file content
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        logger.warning(f"⚠️ Could not read {relative_path}: {e}")
                        continue

                    # Index in Qdrant
                    file_type = filepath.suffix[1:] if filepath.suffix else "unknown"
                    self.qdrant.store("project_files", content, {
                        "file_path": relative_path,
                        "file_type": file_type,
                        "file_name": filepath.name,
                        "size_bytes": len(content),
                        "modified_time": mtime,
                        "indexed_at": datetime.now().isoformat()
                    })

                    self.indexed_files[relative_path] = mtime

                except Exception as e:
                    logger.warning(f"⚠️ Error processing {filepath}: {e}")

        except Exception as e:
            logger.error(f"❌ Scan failed: {e}")

        logger.info(f"✅ Scan complete: {files_indexed} new, {files_updated} updated, {files_skipped} skipped")

        return {
            "files_indexed": files_indexed,
            "files_updated": files_updated,
            "files_skipped": files_skipped,
            "total_tracked": len(self.indexed_files)
        }

# ============================================================================
# Qdrant Manager
# ============================================================================

class QdrantManager:
    """Qdrant operations manager."""

    CACHE_THRESHOLD = 0.95
    CONTEXT_THRESHOLD = 0.85

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self):
        try:
            logger.info("🔗 Connecting to Qdrant Cloud...")
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
            collections = self.client.get_collections()
            logger.info(f"✅ Connected! Collections: {len(collections.collections)}")

            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")

            self._setup_collections()

        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            raise

    def _setup_collections(self):
        collections = [
            "tasks", "architectures", "implementations", "code_reviews",
            "test_strategies", "security_findings", "diagrams", "documentation",
            "git_commits", "ci_cd_configs", "llm_cache", "feedback",
            "context_memory", "optimization_patterns", "groq_workflows",
            "conversations", "success_logs", "failure_logs", "learning_patterns",
            "performance_metrics", "project_files"
        ]

        existing = {c.name for c in self.client.get_collections().collections}

        for collection_name in collections:
            if collection_name in existing:
                continue

            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.debug(f"✅ Created: {collection_name}")
            except Exception as e:
                logger.warning(f"⚠️ {collection_name}: {e}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedding_model.encode(text[:8000]).tolist()
        except:
            return []

    def check_cache(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        try:
            query_embedding = self.get_embedding(prompt)
            if not query_embedding:
                return None

            results = self.client.search(
                collection_name="llm_cache",
                query_vector=query_embedding,
                limit=1,
                score_threshold=self.CACHE_THRESHOLD,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="model", match=MatchValue(value=model)),
                        FieldCondition(key="temperature", match=MatchValue(value=temperature))
                    ]
                )
            )

            if results and len(results) > 0:
                logger.info(f"🎯 Cache HIT! ({results[0].score:.3f})")
                return results[0].payload["response"]

            return None

        except:
            return None

    def store_in_cache(self, prompt: str, response: str, model: str, temperature: float, tokens: int):
        try:
            cache_id = hashlib.md5(f"{prompt}{model}{datetime.now()}".encode()).hexdigest()
            embedding = self.get_embedding(prompt)
            if not embedding:
                return

            self.client.upsert(
                collection_name="llm_cache",
                points=[PointStruct(id=cache_id, vector=embedding, payload={
                    "prompt": prompt[:500],
                    "response": response,
                    "model": model,
                    "temperature": temperature,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })]
            )

        except:
            pass

    def store(self, collection: str, text: str, metadata: dict) -> str:
        try:
            vector_id = hashlib.md5(f"{text}{datetime.now()}".encode()).hexdigest()
            embedding = self.get_embedding(text)
            if not embedding:
                return ""

            metadata["stored_at"] = datetime.now().isoformat()
            metadata["text_preview"] = text[:200]

            self.client.upsert(
                collection_name=collection,
                points=[PointStruct(id=vector_id, vector=embedding, payload=metadata)]
            )

            return vector_id

        except Exception as e:
            logger.warning(f"Store failed: {e}")
            return ""

    def store_conversation(self, user_query: str, agent_response: str, agent: str, user_id: str, satisfaction: str = "pending") -> str:
        conversation_text = f"User: {user_query}\nAgent ({agent}): {agent_response}"
        return self.store("conversations", conversation_text, {
            "user_query": user_query[:500],
            "agent_response": agent_response[:500],
            "agent": agent,
            "user_id": user_id,
            "user_satisfaction": satisfaction
        })

    def log_success(self, task_id: str, success_type: str, description: str,
                    agent: str, outcome: str, metrics: dict, lessons_learned: str):
        success_text = f"{success_type}: {description}. {outcome}. {lessons_learned}"
        self.store("success_logs", success_text, {
            "task_id": task_id,
            "success_type": success_type,
            "description": description,
            "agent": agent,
            "outcome": outcome,
            "metrics": metrics,
            "lessons_learned": lessons_learned
        })
        logger.info(f"✅ Success: {success_type}")

    def log_failure(self, task_id: str, failure_type: str, description: str,
                    agent: str, error_message: str, root_cause: str,
                    resolution: str, lessons_learned: str):
        failure_text = f"{failure_type}: {description}. {error_message}. {root_cause}"
        self.store("failure_logs", failure_text, {
            "task_id": task_id,
            "failure_type": failure_type,
            "description": description,
            "agent": agent,
            "error_message": error_message,
            "root_cause": root_cause,
            "resolution": resolution,
            "lessons_learned": lessons_learned
        })
        logger.warning(f"❌ Failure: {failure_type}")

    def search(self, collection: str, query: str, limit: int = 3, threshold: float = None) -> List[dict]:
        try:
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []

            results = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=threshold or self.CONTEXT_THRESHOLD
            )

            return [{"score": r.score, "payload": r.payload, "id": r.id} for r in results]

        except:
            return []

    def get_relevant_context(self, task_description: str, collections: List[str], limit_per_collection: int = 2) -> Dict[str, List[dict]]:
        context = {}
        for collection in collections:
            results = self.search(collection, task_description, limit=limit_per_collection)
            if results:
                context[collection] = results
        return context

qdrant = QdrantManager(config.qdrant_url, config.qdrant_key)

# ============================================================================
# Initialize Project Scanner
# ============================================================================

project_scanner = ProjectScanner(PROJECT_ROOT, qdrant)
logger.info(f"🚀 Scanning project directory: {PROJECT_ROOT}")
scan_results = project_scanner.scan_and_index()
logger.info(f"📊 Scan results: {json.dumps(scan_results, indent=2)}")

# ============================================================================
# LLM Caller
# ============================================================================

class LLMCaller:
    """LLM caller with auth and context."""

    def __init__(self, clients: APIClients, qdrant: QdrantManager, auth: AuthManager):
        self.clients = clients
        self.qdrant = qdrant
        self.auth = auth

    async def call_agent(self, agent: str, prompt: str, user_id: str, use_context: bool = True, ctx: Optional[Context] = None) -> Tuple[str, int]:
        if not self.auth.can_call_agent(user_id, agent):
            user = self.auth.users.get(user_id)
            role = user.role.value if user else "unknown"
            error_msg = f"Access denied: {role} cannot call {agent}"
            logger.warning(f"🚫 {error_msg}")
            raise PermissionError(error_msg)

        if agent not in TEAM:
            raise ValueError(f"Unknown agent: {agent}")

        agent_config = TEAM[agent]

        # Enhance with project context from indexed files
        if use_context:
            # Get relevant project files
            project_context = self.qdrant.get_relevant_context(
                prompt[:500],
                ["project_files", "implementations", "success_logs", "failure_logs"],
                limit_per_collection=2
            )

            if project_context:
                context_str = "\n\n## Relevant Project Context:\n"
                for coll, items in project_context.items():
                    for item in items:
                        context_str += f"\n[{coll}, similarity: {item['score']:.2f}]\n"
                        if coll == "project_files":
                            context_str += f"File: {item['payload'].get('file_path', 'unknown')}\n"
                        context_str += f"{item['payload'].get('text_preview', '')}...\n"

                prompt = prompt + context_str
                logger.debug(f"🧠 Enhanced with context from {len(project_context)} collections")

        # Check cache
        cached = self.qdrant.check_cache(prompt, agent_config.model, agent_config.temperature)
        if cached:
            return cached, 0

        logger.info(f"🤖 {agent} ({agent_config.model} via {agent_config.provider})")

        try:
            response_text, tokens = await self._call_provider(
                agent_config.provider,
                agent_config.model,
                prompt,
                agent_config.temperature,
                agent_config.max_tokens
            )

            self.qdrant.store_in_cache(prompt, response_text, agent_config.model, agent_config.temperature, tokens)
            self.qdrant.store_conversation(prompt[:500], response_text[:500], agent, user_id)

            return response_text, tokens

        except Exception as e:
            logger.error(f"❌ {agent} failed: {e}")
            raise

    async def _call_provider(self, provider: str, model: str, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int]:
        if provider == "openai":
            response = self.clients.openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens

        elif provider == "groq":
            response = self.clients.groq.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens

        elif provider == "cerebras":
            client = self.clients.get_cerebras_client()
            if not client:
                raise ValueError("Cerebras not available")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens

        elif provider == "qwen":
            if not self.clients.qwen_available:
                raise ValueError("Qwen not available")

            response = self.clients.qwen.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens

        elif provider == "google_stable":
            gemini_model = genai_stable.GenerativeModel(model)
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai_stable.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            tokens = len(prompt.split()) + len(response.text.split())
            return response.text, tokens

        elif provider == "google_preview":
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            tokens = len(prompt.split()) + len(response.text.split())
            return response.text, tokens

        else:
            raise ValueError(f"Unknown provider: {provider}")

llm_caller = LLMCaller(clients, qdrant, auth_manager)

# ============================================================================
# Task Management
# ============================================================================

TASKS: Dict[str, dict] = {}
TOTAL_TOKENS_USED = 0

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool
async def develop_feature(
    description: str,
    requirements: List[str],
    api_key: str,
    use_rapid_development: bool = True,
    ctx: Context = None
) -> Dict:
    """
    🚀 Complete development workflow with automatic project context.

    UPDATED: Now utilizes ALL available API providers with intelligent routing:
    - OpenAI: Leadership and planning (tech_lead, architect)
    - Cerebras: Core implementation (senior_developer, parallel_developer)
    - Qwen: Algorithm optimization (code_specialist)
    - Groq: Rapid prototyping and testing (rapid_developer, fast_qa)
    - Google: QA, security, docs (qa_engineer, security_auditor, technical_writer)
    """
    user = auth_manager.authenticate(api_key)
    if not user:
        if ctx:
            await ctx.error("❌ Invalid API key")
        raise PermissionError("Invalid API key")

    if not auth_manager.has_permission(user.user_id, Permission.CREATE_TASKS):
        if ctx:
            await ctx.error(f"❌ Access denied: {user.role.value}")
        raise PermissionError("Insufficient permissions")

    user_id = user.user_id
    global TOTAL_TOKENS_USED

    task_id = hashlib.md5(f"{description}{datetime.now()}".encode()).hexdigest()[:12]

    if ctx:
        await ctx.info(f"🚀 Starting: {task_id}")

    try:
        similar_tasks = qdrant.search("tasks", description, limit=3)
        if similar_tasks and similar_tasks[0]["score"] > 0.98:
            if ctx:
                await ctx.info(f"♻️ Found identical task!")

            qdrant.log_success(
                task_id=task_id,
                success_type="reused_solution",
                description="Reused from Qdrant",
                agent="system",
                outcome="0 tokens",
                metrics={"tokens_saved": 5000},
                lessons_learned="Semantic caching works"
            )

            return {
                "task_id": task_id,
                "status": "reused",
                "tokens_used": 0
            }

        task = {
            "task_id": task_id,
            "description": description,
            "requirements": requirements,
            "user_id": user_id,
            "stages": {},
            "created_at": datetime.now().isoformat()
        }

        TASKS[task_id] = task
        task_tokens = 0

        qdrant.store("tasks", f"{description} {' '.join(requirements)}", {
            "task_id": task_id,
            "user_id": user_id
        })

        # ===================================================================
        # STAGE 1: Leadership Planning (OpenAI - GPT-4o)
        # ===================================================================
        if ctx:
            await ctx.info("👔 Tech Lead (OpenAI GPT-4o)")

        kickoff_prompt = f"""Tech Lead: {description}

Requirements: {', '.join(requirements)}

Provide strategic analysis as JSON with:
- Project scope and goals
- Key technical decisions
- Resource allocation
- Risk assessment
"""

        kickoff, tokens = await llm_caller.call_agent("tech_lead", kickoff_prompt, user_id, True, ctx)
        task["stages"]["kickoff"] = {"result": kickoff, "tokens": tokens, "agent": "tech_lead", "provider": "openai"}
        task_tokens += tokens

        # ===================================================================
        # STAGE 2: Architecture Design (OpenAI - O1)
        # ===================================================================
        if ctx:
            await ctx.info("🏗️ Architect (OpenAI O1)")

        arch_prompt = f"""Architect: {kickoff[:1000]}

Design comprehensive architecture as JSON with:
- System components and interactions
- Data flow and storage strategy
- API design and interfaces
- Scalability considerations
"""

        architecture, tokens = await llm_caller.call_agent("architect", arch_prompt, user_id, True, ctx)
        task["stages"]["architecture"] = {"result": architecture, "tokens": tokens, "agent": "architect", "provider": "openai"}
        task_tokens += tokens

        qdrant.store("architectures", architecture[:2000], {"task_id": task_id})

        # ===================================================================
        # STAGE 3: Core Implementation (Cerebras - Llama 3.3 70B)
        # ===================================================================
        if ctx:
            await ctx.info(f"💻 Senior Developer ({'Cerebras' if clients.cerebras_available else 'Groq fallback'})")

        impl_prompt = f"""Senior Developer: {architecture[:1500]}

Implement core functionality with:
- Main application logic
- Error handling and validation
- Performance optimization
- Code documentation
"""

        implementation, tokens = await llm_caller.call_agent("senior_developer", impl_prompt, user_id, True, ctx)
        task["stages"]["implementation"] = {
            "result": implementation,
            "tokens": tokens,
            "agent": "senior_developer",
            "provider": TEAM["senior_developer"].provider
        }
        task_tokens += tokens

        qdrant.store("implementations", implementation[:2000], {"task_id": task_id})

        # ===================================================================
        # STAGE 4: Algorithm Optimization (Qwen or Groq)
        # ===================================================================
        if ctx:
            await ctx.info(f"🧮 Code Specialist ({'Qwen' if clients.qwen_available else 'Groq fallback'})")

        opt_prompt = f"""Code Specialist: {implementation[:1500]}

Optimize algorithms and data structures:
- Performance bottleneck analysis
- Algorithm complexity improvements
- Memory optimization
- Best practices application
"""

        optimization, tokens = await llm_caller.call_agent("code_specialist", opt_prompt, user_id, True, ctx)
        task["stages"]["optimization"] = {
            "result": optimization,
            "tokens": tokens,
            "agent": "code_specialist",
            "provider": TEAM["code_specialist"].provider
        }
        task_tokens += tokens

        # ===================================================================
        # STAGE 5: Rapid Prototyping (Groq Compound)
        # ===================================================================
        if ctx:
            await ctx.info("⚡ Rapid Prototyper (Groq Compound)")

        proto_prompt = f"""Rapid Prototyper: {optimization[:1000]}

Create working prototype with:
- Quick proof-of-concept
- Essential features only
- Basic UI/UX
- Integration points
"""

        prototype, tokens = await llm_caller.call_agent("rapid_prototyper", proto_prompt, user_id, True, ctx)
        task["stages"]["prototype"] = {"result": prototype, "tokens": tokens, "agent": "rapid_prototyper", "provider": "groq"}
        task_tokens += tokens

        # ===================================================================
        # STAGE 6: Code Review (OpenAI GPT-4o)
        # ===================================================================
        if ctx:
            await ctx.info("👀 Code Reviewer (OpenAI GPT-4o)")

        review_prompt = f"""Senior Code Reviewer: {implementation[:1000]}

Comprehensive code review covering:
- Code quality and readability
- Security vulnerabilities
- Performance issues
- Best practices adherence
"""

        review, tokens = await llm_caller.call_agent("senior_code_reviewer", review_prompt, user_id, True, ctx)
        task["stages"]["code_review"] = {"result": review, "tokens": tokens, "agent": "senior_code_reviewer", "provider": "openai"}
        task_tokens += tokens

        qdrant.store("code_reviews", review[:2000], {"task_id": task_id})

        # ===================================================================
        # STAGE 7: QA Testing (Google Gemini 2.5 Flash)
        # ===================================================================
        if ctx:
            await ctx.info("🧪 QA Engineer (Google Gemini)")

        qa_prompt = f"""QA Engineer: {prototype[:1000]}

Comprehensive testing strategy:
- Unit test coverage
- Integration test scenarios
- Edge cases and error paths
- Performance testing plan
"""

        testing, tokens = await llm_caller.call_agent("qa_engineer", qa_prompt, user_id, True, ctx)
        task["stages"]["testing"] = {"result": testing, "tokens": tokens, "agent": "qa_engineer", "provider": "google_stable"}
        task_tokens += tokens

        qdrant.store("test_strategies", testing[:2000], {"task_id": task_id})

        # ===================================================================
        # STAGE 8: Security Audit (Google Gemini 2.5 Flash)
        # ===================================================================
        if ctx:
            await ctx.info("🔒 Security Auditor (Google Gemini)")

        security_prompt = f"""Security Auditor: {implementation[:1000]}

Security audit covering:
- Vulnerability assessment
- Input validation
- Authentication/Authorization
- Data protection
"""

        security, tokens = await llm_caller.call_agent("security_auditor", security_prompt, user_id, True, ctx)
        task["stages"]["security"] = {"result": security, "tokens": tokens, "agent": "security_auditor", "provider": "google_stable"}
        task_tokens += tokens

        qdrant.store("security_findings", security[:2000], {"task_id": task_id})

        # ===================================================================
        # STAGE 9: Documentation (Google Gemini 2.5 Pro)
        # ===================================================================
        if ctx:
            await ctx.info("📝 Technical Writer (Google Gemini Pro)")

        docs_prompt = f"""Technical Writer: {implementation[:1000]}

Create comprehensive documentation:
- API documentation
- User guides
- Setup instructions
- Code examples
"""

        documentation, tokens = await llm_caller.call_agent("technical_writer", docs_prompt, user_id, True, ctx)
        task["stages"]["documentation"] = {"result": documentation, "tokens": tokens, "agent": "technical_writer", "provider": "google_stable"}
        task_tokens += tokens

        qdrant.store("documentation", documentation[:2000], {"task_id": task_id})

        # ===================================================================
        # Success Logging
        # ===================================================================
        provider_usage = {}
        for stage_name, stage_data in task["stages"].items():
            provider = stage_data.get("provider", "unknown")
            agent = stage_data.get("agent", "unknown")
            tokens = stage_data.get("tokens", 0)

            if provider not in provider_usage:
                provider_usage[provider] = {"tokens": 0, "agents": []}

            provider_usage[provider]["tokens"] += tokens
            provider_usage[provider]["agents"].append(f"{agent}({tokens})")

        qdrant.log_success(
            task_id=task_id,
            success_type="feature_complete",
            description=description,
            agent="multi_agent_system",
            outcome="Complete workflow across all providers",
            metrics={
                "total_tokens": task_tokens,
                "stages_completed": len(task["stages"]),
                "provider_usage": provider_usage
            },
            lessons_learned="Multi-provider workflow maximizes resource utilization"
        )

        task["tokens_used"] = task_tokens
        task["status"] = "complete"
        task["provider_usage"] = provider_usage
        TOTAL_TOKENS_USED += task_tokens

        if ctx:
            await ctx.info(f"✅ Complete! Total tokens: {task_tokens}")
            await ctx.info(f"📊 Provider usage: {json.dumps(provider_usage, indent=2)}")

        return task

    except Exception as e:
        qdrant.log_failure(
            task_id=task_id,
            failure_type="system_error",
            description=description,
            agent="system",
            error_message=str(e),
            root_cause="Exception during workflow",
            resolution="Manual investigation required",
            lessons_learned="Need better error recovery"
        )

        logger.error(f"Failed: {e}")
        if ctx:
            await ctx.error(f"❌ Failed: {str(e)}")
        raise

@mcp.tool
async def rescan_project(api_key: str, ctx: Context = None) -> Dict:
    """
    🔄 Manually trigger project rescan and reindex.
    Scans project directory for new/modified files and updates Qdrant.
    """
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")

    if ctx:
        await ctx.info(f"🔄 Rescanning project: {PROJECT_ROOT}")

    results = project_scanner.scan_and_index()

    if ctx:
        await ctx.info(f"✅ Scan complete: {results['files_indexed']} new, {results['files_updated']} updated")

    return results

@mcp.tool
async def create_user(
    username: str,
    role: str,
    api_key: str,
    ctx: Context = None
) -> Dict:
    """👤 Create new user (Admin only)."""
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")

    role_enum = Role(role)
    new_user = auth_manager.create_user(username, role_enum, user.user_id)

    if ctx:
        await ctx.info(f"✅ Created: {username}")

    return {
        "user_id": new_user.user_id,
        "username": new_user.username,
        "role": new_user.role.value,
        "api_key": new_user.api_key,
        "created_at": new_user.created_at.isoformat()
    }

@mcp.tool
async def list_team_members(ctx: Context = None) -> Dict:
    """👥 List all team members with their provider info."""
    team_info = {}

    for name, config in TEAM.items():
        team_info[name] = {
            "model": config.model,
            "provider": config.provider,
            "role": config.role,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }

    # Provider summary
    provider_counts = {}
    for config in TEAM.values():
        provider = config.provider
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    return {
        "team": team_info,
        "total_members": len(TEAM),
        "provider_distribution": provider_counts,
        "providers_active": {
            "openai": True,
            "groq": True,
            "google_stable": True,
            "google_preview": True,
            "cerebras": clients.cerebras_available,
            "qwen": clients.qwen_available
        }
    }

@mcp.tool
async def get_api_usage_stats(api_key: str, ctx: Context = None) -> Dict:
    """📊 Get comprehensive API usage stats across all providers."""
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")

    if not auth_manager.has_permission(user.user_id, Permission.VIEW_ANALYTICS):
        raise PermissionError("Insufficient permissions")

    stats = {
        "total_tokens_used": TOTAL_TOKENS_USED,
        "team_size": len(TEAM),
        "active_tasks": len(TASKS),
        "user_id": user.user_id,
        "project_files_indexed": len(project_scanner.indexed_files)
    }

    # Cerebras usage breakdown
    if clients.cerebras_available:
        cerebras_stats = {}
        for key_name, client_info in clients.cerebras_clients.items():
            cerebras_stats[key_name] = client_info["usage_count"]

        stats["cerebras"] = cerebras_stats
        stats["cerebras_total_calls"] = sum(cerebras_stats.values())
    else:
        stats["cerebras"] = "Not configured (using Groq fallback)"

    # Qwen usage
    if clients.qwen_available:
        stats["qwen"] = "Active"
    else:
        stats["qwen"] = "Not configured (using Groq fallback)"

    # Provider agent counts
    provider_agents = {}
    for agent_name, config in TEAM.items():
        provider = config.provider
        if provider not in provider_agents:
            provider_agents[provider] = []
        provider_agents[provider].append(agent_name)

    stats["provider_agent_distribution"] = {
        provider: len(agents) for provider, agents in provider_agents.items()
    }

    return stats

@mcp.tool
async def search_project_files(
    query: str,
    api_key: str,
    limit: int = 5,
    ctx: Context = None
) -> Dict:
    """
    🔍 Search indexed project files.
    Semantic search across all indexed source code.
    """
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")

    results = qdrant.search("project_files", query, limit=limit)

    return {
        "query": query,
        "results": [
            {
                "file_path": r["payload"].get("file_path"),
                "file_type": r["payload"].get("file_type"),
                "similarity": r["score"],
                "preview": r["payload"].get("text_preview")
            }
            for r in results
        ]
    }

@mcp.tool
async def test_all_providers(api_key: str, ctx: Context = None) -> Dict:
    """
    🧪 Test ALL API providers to verify they're working.
    This tool makes a simple call to one agent from each provider.
    """
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")

    user_id = user.user_id
    test_prompt = "Respond with 'OK' if you receive this message."

    results = {}

    # Test OpenAI
    try:
        if ctx:
            await ctx.info("Testing OpenAI...")
        response, tokens = await llm_caller.call_agent("tech_lead", test_prompt, user_id, False, ctx)
        results["openai"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "tech_lead"}
    except Exception as e:
        results["openai"] = {"status": "❌ FAILED", "error": str(e)}

    # Test Groq
    try:
        if ctx:
            await ctx.info("Testing Groq...")
        response, tokens = await llm_caller.call_agent("rapid_developer", test_prompt, user_id, False, ctx)
        results["groq"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "rapid_developer"}
    except Exception as e:
        results["groq"] = {"status": "❌ FAILED", "error": str(e)}

    # Test Google
    try:
        if ctx:
            await ctx.info("Testing Google...")
        response, tokens = await llm_caller.call_agent("qa_engineer", test_prompt, user_id, False, ctx)
        results["google"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "qa_engineer"}
    except Exception as e:
        results["google"] = {"status": "❌ FAILED", "error": str(e)}

    # Test Cerebras
    try:
        if ctx:
            await ctx.info(f"Testing Cerebras ({'configured' if clients.cerebras_available else 'fallback'})...")
        response, tokens = await llm_caller.call_agent("senior_developer", test_prompt, user_id, False, ctx)
        results["cerebras"] = {
            "status": "✅ SUCCESS",
            "tokens": tokens,
            "agent": "senior_developer",
            "using_cerebras": clients.cerebras_available,
            "fallback_to": "groq" if not clients.cerebras_available else None
        }
    except Exception as e:
        results["cerebras"] = {"status": "❌ FAILED", "error": str(e)}

    # Test Qwen
    try:
        if ctx:
            await ctx.info(f"Testing Qwen ({'configured' if clients.qwen_available else 'fallback'})...")
        response, tokens = await llm_caller.call_agent("code_specialist", test_prompt, user_id, False, ctx)
        results["qwen"] = {
            "status": "✅ SUCCESS",
            "tokens": tokens,
            "agent": "code_specialist",
            "using_qwen": clients.qwen_available,
            "fallback_to": "groq" if not clients.qwen_available else None
        }
    except Exception as e:
        results["qwen"] = {"status": "❌ FAILED", "error": str(e)}

    # Summary
    total_tested = len(results)
    successful = sum(1 for r in results.values() if "SUCCESS" in r.get("status", ""))

    return {
        "test_results": results,
        "summary": {
            "total_providers_tested": total_tested,
            "successful": successful,
            "failed": total_tested - successful,
            "success_rate": f"{(successful/total_tested)*100:.1f}%"
        }
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Multi-Agent Development Team MCP Server")
    logger.info(f"👥 Team: {len(TEAM)} agents")
    logger.info(f"🔐 Authentication: Enabled")
    logger.info(f"🗄️ Qdrant: Connected (21 collections)")
    logger.info(f"📁 Project: {PROJECT_ROOT}")
    logger.info(f"📊 Files indexed: {len(project_scanner.indexed_files)}")

    available_providers = ["OpenAI", "Groq", "Google"]

    if clients.cerebras_available:
        available_providers.append("Cerebras")
        logger.info(f"   ✅ Cerebras: {len(clients.cerebras_clients)} key(s) active")
    else:
        logger.info(f"   ⚠️ Cerebras: Not configured (using Groq fallback)")

    if clients.qwen_available:
        available_providers.append("Qwen")
        logger.info(f"   ✅ Qwen: Active")
    else:
        logger.info(f"   ⚠️ Qwen: Not configured (using Groq fallback)")

    logger.info(f"🌐 Providers: {', '.join(available_providers)}")
    logger.info("="*70)
    logger.info("RESOURCE UTILIZATION GUARANTEED:")
    logger.info("  - All agents always registered (with fallbacks)")
    logger.info("  - develop_feature() calls 9 stages across ALL providers")
    logger.info("  - Cerebras/Qwen gracefully fallback to Groq if unconfigured")
    logger.info("  - Use test_all_providers() to verify all APIs")
    logger.info("="*70)

    mcp.run()

