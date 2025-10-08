"""
multi_agent_dev_team.py

Enterprise multi-agent orchestration with automatic project indexing.

Features:
- Scans project directory on startup
- Indexes all source files (.go, .py, .sh, .md) in Qdrant
- Tracks file timestamps for incremental updates
- 30+ specialized agents across 5 API providers
- ALL LATEST MODELS (October 2025)
- Role-based access control
- Comprehensive learning from success/failure patterns
- GUARANTEED resource utilization for ALL API providers

Version: 3.0.0
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

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())

INDEXABLE_PATTERNS = [
    "*.go", "*.py", "*.sh", "*.md", "*.yaml", "*.yml",
    "*.json", "*.toml", "*.txt", "*.rs", "*.c", "*.h"
]

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
        "intelligent routing, comprehensive memory, and ALL latest models."
    )
)

# ============================================================================
# Authentication System
# ============================================================================

class Role(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    WRITER = "writer"
    VIEWER = "viewer"

class Permission(Enum):
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
    "gpt5_flagship": Permission.CALL_DEVELOPMENT,
    "gpt5_mini": Permission.CALL_DEVELOPMENT,
    "gpt5_nano": Permission.CALL_DEVELOPMENT,
    "gpt41_smart": Permission.CALL_DEVELOPMENT,
    "o3_researcher": Permission.CALL_DEVELOPMENT,
    "o4_mini_researcher": Permission.CALL_DEVELOPMENT,
    "gpt_oss_120b": Permission.CALL_DEVELOPMENT,
    "groq_compound": Permission.CALL_DEVELOPMENT,
    "groq_compound_mini": Permission.CALL_DEVELOPMENT,
    "llama4_maverick": Permission.CALL_DEVELOPMENT,
    "llama4_scout": Permission.CALL_DEVELOPMENT,
    "llama33_versatile": Permission.CALL_DEVELOPMENT,
    "kimi_256k": Permission.CALL_DEVELOPMENT,
    "cerebras_llama4_scout": Permission.CALL_DEVELOPMENT,
    "cerebras_llama4_maverick": Permission.CALL_DEVELOPMENT,
    "cerebras_qwen3_235b": Permission.CALL_DEVELOPMENT,
    "cerebras_qwen3_coder": Permission.CALL_DEVELOPMENT,
    "qwen_max": Permission.CALL_DEVELOPMENT,
    "qwen_plus": Permission.CALL_DEVELOPMENT,
    "qwen_coder_480b": Permission.CALL_DEVELOPMENT,
    "qwq_reasoning": Permission.CALL_DEVELOPMENT,
    "gemini_pro": Permission.CALL_DEVELOPMENT,
    "gemini_flash": Permission.CALL_DEVELOPMENT,
    "gemini_flash_lite": Permission.CALL_DEVELOPMENT,
    "gemini_computer_use": Permission.CALL_DEVELOPMENT,
    "qa_engineer": Permission.CALL_QA,
    "security_auditor": Permission.CALL_QA,
    "technical_writer": Permission.CALL_DOCS,
    "diagram_specialist": Permission.CALL_DOCS,
    "git_specialist": Permission.CALL_DEVOPS,
    "devops_engineer": Permission.CALL_DEVOPS
}

@dataclass
class User:
    user_id: str
    username: str
    role: Role
    api_key: str
    created_at: datetime
    last_access: Optional[datetime] = None
    is_active: bool = True

class AuthManager:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}
        self._initialize_default_users()
    
    def _initialize_default_users(self):
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
    def __init__(self, config: APIConfig):
        self.config = config
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
        if not self.cerebras_available or not self.cerebras_clients:
            return None
        
        if len(self.cerebras_clients) == 1:
            key_name = list(self.cerebras_clients.keys())[0]
            client_info = self.cerebras_clients[key_name]
            client_info["usage_count"] += 1
            return client_info["client"]
        
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
# Team Configuration - ALL LATEST MODELS (October 2025)
# ============================================================================

@dataclass
class AgentConfig:
    model: str
    provider: str
    role: str
    temperature: float = 0.3
    max_tokens: int = 4000

# ALL agents always registered with intelligent fallbacks
TEAM: Dict[str, AgentConfig] = {
    # ========================================================================
    # OPENAI AGENTS - Latest GPT-5 Series (August 2025)
    # ========================================================================
    "gpt5_flagship": AgentConfig("gpt-5", "openai", "Flagship coding & agentic tasks", 0.3, 16000),
    "gpt5_mini": AgentConfig("gpt-5-mini", "openai", "Fast cost-efficient GPT-5", 0.3, 12000),
    "gpt5_nano": AgentConfig("gpt-5-nano", "openai", "Fastest GPT-5 variant", 0.4, 8000),
    "gpt41_smart": AgentConfig("gpt-4.1", "openai", "Smartest non-reasoning", 0.2, 12000),
    "o3_researcher": AgentConfig("o3-deep-research", "openai", "Deep research model", 0.1, 16000),
    "o4_mini_researcher": AgentConfig("o4-mini-deep-research", "openai", "Fast research", 0.2, 12000),
    "gpt_oss_120b": AgentConfig("gpt-oss-120b", "openai", "Open-weight 120B", 0.3, 12000),
    
    # ========================================================================
    # GROQ AGENTS - Latest Compound & Llama 4 (2025)
    # ========================================================================
    "groq_compound": AgentConfig("groq/compound", "groq", "Compound system with tools", 0.5, 12000),
    "groq_compound_mini": AgentConfig("groq/compound-mini", "groq", "Fast compound", 0.5, 8000),
    "groq_gpt_oss": AgentConfig("openai/gpt-oss-120b", "groq", "OpenAI open on Groq", 0.3, 12000),
    "llama4_maverick": AgentConfig("meta-llama/llama-4-maverick-17b-128e-instruct", "groq", "Llama 4 heavy-duty", 0.4, 10000),
    "llama4_scout": AgentConfig("meta-llama/llama-4-scout-17b-16e-instruct", "groq", "Llama 4 planning", 0.4, 10000),
    "llama33_versatile": AgentConfig("llama-3.3-70b-versatile", "groq", "Llama 3.3 70B", 0.4, 12000),
    "kimi_256k": AgentConfig("moonshotai/kimi-k2-instruct-0905", "groq", "256k context specialist", 0.4, 16000),
    "groq_qwen": AgentConfig("qwen/qwen3-32b", "groq", "Qwen 3 on Groq", 0.3, 8000),
    
    # ========================================================================
    # GOOGLE GEMINI AGENTS - Latest 2.5 Series (October 2025)
    # ========================================================================
    "gemini_pro": AgentConfig("gemini-2.5-pro", "google_stable", "Flagship reasoning", 0.2, 16000),
    "gemini_flash": AgentConfig("gemini-2.5-flash", "google_stable", "Best price-performance", 0.3, 16000),
    "gemini_flash_lite": AgentConfig("gemini-2.5-flash-lite", "google_stable", "Ultra-fast", 0.3, 12000),
    "gemini_flash_image": AgentConfig("gemini-2.5-flash-image", "google_stable", "Image generation", 0.3, 8000),
    "qa_engineer": AgentConfig("gemini-2.5-flash", "google_stable", "QA Testing", 0.2, 8000),
    "security_auditor": AgentConfig("gemini-2.5-flash", "google_stable", "Security audit", 0.2, 8000),
    "technical_writer": AgentConfig("gemini-2.5-pro", "google_stable", "Documentation", 0.3, 12000),
    "git_specialist": AgentConfig("gemini-2.5-flash-lite", "google_stable", "Git workflows", 0.2, 4000),
    "devops_engineer": AgentConfig("gemini-2.5-flash", "google_stable", "CI/CD", 0.2, 8000),
    
    # ========================================================================
    # CEREBRAS AGENTS - Latest Llama 4 & Qwen 3 (with fallback)
    # ========================================================================
    "cerebras_llama4_scout": AgentConfig(
        "llama-4-scout-17b-16e-instruct" if clients.cerebras_available else "meta-llama/llama-4-scout-17b-16e-instruct",
        "cerebras" if clients.cerebras_available else "groq",
        "Cerebras Llama 4 Scout",
        0.4,
        10000
    ),
    "cerebras_llama4_maverick": AgentConfig(
        "llama-4-maverick-17b-128e-instruct" if clients.cerebras_available else "meta-llama/llama-4-maverick-17b-128e-instruct",
        "cerebras" if clients.cerebras_available else "groq",
        "Cerebras Llama 4 Maverick",
        0.4,
        12000
    ),
    "cerebras_qwen3_235b": AgentConfig(
        "qwen-3-235b-a22b-instruct-2507" if clients.cerebras_available else "qwen/qwen3-32b",
        "cerebras" if clients.cerebras_available else "groq",
        "Qwen 3 235B on Cerebras",
        0.3,
        16000
    ),
    "cerebras_qwen3_coder": AgentConfig(
        "qwen-3-coder-480b" if clients.cerebras_available else "qwen/qwen3-32b",
        "cerebras" if clients.cerebras_available else "groq",
        "Qwen 3 Coder 480B",
        0.3,
        16000
    ),
    "cerebras_gpt_oss": AgentConfig(
        "gpt-oss-120b" if clients.cerebras_available else "openai/gpt-oss-120b",
        "cerebras" if clients.cerebras_available else "groq",
        "GPT-OSS on Cerebras",
        0.3,
        12000
    ),
    
    # ========================================================================
    # QWEN/DASHSCOPE AGENTS - Latest Qwen Max & Coder (with fallback)
    # ========================================================================
    "qwen_max": AgentConfig(
        "qwen-max" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq",
        "Qwen Max flagship (1T+ params)",
        0.3,
        16000
    ),
    "qwen_plus": AgentConfig(
        "qwen-plus" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq",
        "Qwen Plus advanced",
        0.3,
        12000
    ),
    "qwen_coder_480b": AgentConfig(
        "qwen3-coder-480b-a35b-instruct" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq",
        "Qwen Coder 480B specialist",
        0.3,
        16000
    ),
    "qwq_reasoning": AgentConfig(
        "qwq-plus" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq",
        "QwQ reasoning model",
        0.2,
        16000
    ),
}

logger.info(f"✅ Team configured with {len(TEAM)} agents using LATEST models (Oct 2025)")
logger.info(f"   - OpenAI: GPT-5, GPT-5-mini, GPT-5-nano, GPT-4.1, O3/O4-mini, GPT-OSS")
logger.info(f"   - Groq: Compound, Llama 4, Kimi K2 (256k), Qwen 3")
logger.info(f"   - Google: Gemini 2.5 Pro/Flash/Flash-Lite/Flash-Image")
logger.info(f"   - Cerebras: {'ACTIVE' if clients.cerebras_available else 'FALLBACK to Groq'} - Llama 4, Qwen 3 235B/480B")
logger.info(f"   - Qwen: {'ACTIVE' if clients.qwen_available else 'FALLBACK to Groq'} - Max (1T+), Plus, Coder-480B, QwQ")

# ============================================================================
# Project Scanner
# ============================================================================

class ProjectScanner:
    def __init__(self, root_dir: str, qdrant_manager):
        self.root_dir = Path(root_dir)
        self.qdrant = qdrant_manager
        self.indexed_files: Dict[str, float] = {}
    
    def should_index(self, filepath: Path) -> bool:
        for part in filepath.parts:
            if part in EXCLUDE_DIRS:
                return False
        
        for pattern in INDEXABLE_PATTERNS:
            if fnmatch.fnmatch(filepath.name, pattern):
                return True
        
        return False
    
    def scan_and_index(self):
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
                    mtime = filepath.stat().st_mtime
                    relative_path = str(filepath.relative_to(self.root_dir))
                    
                    if relative_path in self.indexed_files:
                        if self.indexed_files[relative_path] >= mtime:
                            files_skipped += 1
                            continue
                        files_updated += 1
                    else:
                        files_indexed += 1
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        logger.warning(f"⚠️ Could not read {relative_path}: {e}")
                        continue
                    
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

project_scanner = ProjectScanner(PROJECT_ROOT, qdrant)
logger.info(f"🚀 Scanning project directory: {PROJECT_ROOT}")
scan_results = project_scanner.scan_and_index()
logger.info(f"📊 Scan results: {json.dumps(scan_results, indent=2)}")

# ============================================================================
# LLM Caller
# ============================================================================

class LLMCaller:
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
        
        if use_context:
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
    ctx: Context = None
) -> Dict:
    """
    🚀 Complete development workflow utilizing ALL LATEST MODELS (October 2025).
    
    Uses 12+ stages across ALL providers:
    - OpenAI: GPT-5, GPT-5-mini, O3-deep-research, GPT-4.1
    - Groq: Compound, Llama 4 Maverick/Scout
    - Cerebras: Llama 4, Qwen 3 235B/480B
    - Qwen: Max (1T+), Coder-480B, QwQ
    - Google: Gemini 2.5 Pro/Flash/Flash-Lite
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
        await ctx.info(f"🚀 Starting task: {task_id}")
        await ctx.info(f"Using LATEST models from ALL providers (Oct 2025)")
    
    try:
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
        
        # STAGE 1: Deep Research (OpenAI O3)
        if ctx:
            await ctx.info("🔬 O3 Deep Research (OpenAI)")
        
        research_prompt = f"""O3 Deep Research: {description}
Requirements: {', '.join(requirements)}

Conduct comprehensive research with:
- State-of-the-art analysis
- Technical feasibility study
- Performance considerations
- Risk assessment
"""
        
        research, tokens = await llm_caller.call_agent("o3_researcher", research_prompt, user_id, True, ctx)
        task["stages"]["research"] = {"result": research, "tokens": tokens, "agent": "o3_researcher", "provider": "openai"}
        task_tokens += tokens
        
        # STAGE 2: Architecture (GPT-5 Flagship)
        if ctx:
            await ctx.info("🏗️ GPT-5 Flagship Architecture (OpenAI)")
        
        arch_prompt = f"""GPT-5 Flagship: {research[:1500]}

Design comprehensive architecture as JSON with:
- System components and interactions
- Data flow and storage strategy
- API design and interfaces
- Scalability considerations
"""
        
        architecture, tokens = await llm_caller.call_agent("gpt5_flagship", arch_prompt, user_id, True, ctx)
        task["stages"]["architecture"] = {"result": architecture, "tokens": tokens, "agent": "gpt5_flagship", "provider": "openai"}
        task_tokens += tokens
        
        qdrant.store("architectures", architecture[:2000], {"task_id": task_id})
        
        # STAGE 3: Core Implementation (Cerebras Qwen 3 Coder 480B)
        if ctx:
            await ctx.info(f"💻 Qwen 3 Coder 480B ({'Cerebras' if clients.cerebras_available else 'Groq fallback'})")
        
        impl_prompt = f"""Qwen 3 Coder 480B: {architecture[:1500]}

Implement core functionality with:
- Main application logic
- Error handling and validation
- Performance optimization
- Code documentation
"""
        
        implementation, tokens = await llm_caller.call_agent("cerebras_qwen3_coder", impl_prompt, user_id, True, ctx)
        task["stages"]["implementation"] = {
            "result": implementation,
            "tokens": tokens,
            "agent": "cerebras_qwen3_coder",
            "provider": TEAM["cerebras_qwen3_coder"].provider
        }
        task_tokens += tokens
        
        qdrant.store("implementations", implementation[:2000], {"task_id": task_id})
        
        # STAGE 4: Algorithm Optimization (Qwen Max)
        if ctx:
            await ctx.info(f"🧮 Qwen Max 1T+ ({'Qwen' if clients.qwen_available else 'Groq fallback'})")
        
        opt_prompt = f"""Qwen Max (1T+ params): {implementation[:1500]}

Optimize algorithms with:
- Performance bottleneck analysis
- Algorithm complexity improvements
- Memory optimization
- Best practices application
"""
        
        optimization, tokens = await llm_caller.call_agent("qwen_max", opt_prompt, user_id, True, ctx)
        task["stages"]["optimization"] = {
            "result": optimization,
            "tokens": tokens,
            "agent": "qwen_max",
            "provider": TEAM["qwen_max"].provider
        }
        task_tokens += tokens
        
        # STAGE 5: Advanced Reasoning (QwQ Reasoning)
        if ctx:
            await ctx.info(f"🧠 QwQ Reasoning Model ({'Qwen' if clients.qwen_available else 'Groq fallback'})")
        
        reasoning_prompt = f"""QwQ Reasoning: {optimization[:1000]}

Apply advanced reasoning to:
- Identify edge cases
- Logical consistency verification
- Complex decision trees
- Error prevention strategies
"""
        
        reasoning, tokens = await llm_caller.call_agent("qwq_reasoning", reasoning_prompt, user_id, True, ctx)
        task["stages"]["reasoning"] = {
            "result": reasoning,
            "tokens": tokens,
            "agent": "qwq_reasoning",
            "provider": TEAM["qwq_reasoning"].provider
        }
        task_tokens += tokens
        
        # STAGE 6: Rapid Prototyping (Groq Compound)
        if ctx:
            await ctx.info("⚡ Groq Compound System (with web search)")
        
        proto_prompt = f"""Groq Compound: {reasoning[:1000]}

Create working prototype with tools:
- Quick proof-of-concept
- Essential features only
- Integration points
- Web search for latest APIs
"""
        
        prototype, tokens = await llm_caller.call_agent("groq_compound", proto_prompt, user_id, True, ctx)
        task["stages"]["prototype"] = {"result": prototype, "tokens": tokens, "agent": "groq_compound", "provider": "groq"}
        task_tokens += tokens
        
        # STAGE 7: Heavy-Duty Development (Llama 4 Maverick)
        if ctx:
            await ctx.info("🦙 Llama 4 Maverick (Groq)")
        
        heavy_prompt = f"""Llama 4 Maverick: {prototype[:1000]}

Heavy-duty development:
- Production-ready code
- Robust error handling
- Performance optimization
- Complete documentation
"""
        
        heavy_dev, tokens = await llm_caller.call_agent("llama4_maverick", heavy_prompt, user_id, True, ctx)
        task["stages"]["heavy_development"] = {"result": heavy_dev, "tokens": tokens, "agent": "llama4_maverick", "provider": "groq"}
        task_tokens += tokens
        
        # STAGE 8: Code Review (GPT-4.1 Smart)
        if ctx:
            await ctx.info("👀 GPT-4.1 Code Review (OpenAI)")
        
        review_prompt = f"""GPT-4.1 Code Reviewer: {heavy_dev[:1000]}

Comprehensive code review covering:
- Code quality and readability
- Security vulnerabilities
- Performance issues
- Best practices adherence
"""
        
        review, tokens = await llm_caller.call_agent("gpt41_smart", review_prompt, user_id, True, ctx)
        task["stages"]["code_review"] = {"result": review, "tokens": tokens, "agent": "gpt41_smart", "provider": "openai"}
        task_tokens += tokens
        
        qdrant.store("code_reviews", review[:2000], {"task_id": task_id})
        
        # STAGE 9: QA Testing (Gemini 2.5 Flash)
        if ctx:
            await ctx.info("🧪 Gemini 2.5 Flash QA (Google)")
        
        qa_prompt = f"""Gemini 2.5 Flash QA: {heavy_dev[:1000]}

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
        
        # STAGE 10: Security Audit (Gemini 2.5 Flash)
        if ctx:
            await ctx.info("🔒 Gemini 2.5 Flash Security (Google)")
        
        security_prompt = f"""Gemini 2.5 Flash Security: {heavy_dev[:1000]}

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
        
        # STAGE 11: Documentation (Gemini 2.5 Pro)
        if ctx:
            await ctx.info("📝 Gemini 2.5 Pro Documentation (Google)")
        
        docs_prompt = f"""Gemini 2.5 Pro: {heavy_dev[:1000]}

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
        
        # STAGE 12: Final Polish (GPT-5 Mini)
        if ctx:
            await ctx.info("✨ GPT-5 Mini Final Polish (OpenAI)")
        
        polish_prompt = f"""GPT-5 Mini: {documentation[:1000]}

Final polish and packaging:
- Code cleanup
- Comments and docstrings
- README preparation
- Deployment checklist
"""
        
        polish, tokens = await llm_caller.call_agent("gpt5_mini", polish_prompt, user_id, True, ctx)
        task["stages"]["final_polish"] = {"result": polish, "tokens": tokens, "agent": "gpt5_mini", "provider": "openai"}
        task_tokens += tokens
        
        # Success Logging
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
            outcome="Complete 12-stage workflow across ALL providers with latest models",
            metrics={
                "total_tokens": task_tokens,
                "stages_completed": len(task["stages"]),
                "provider_usage": provider_usage,
                "models_used": "GPT-5/5-mini/4.1, O3, Gemini 2.5 Pro/Flash, Llama 4, Qwen 3 480B/Max, QwQ, Compound"
            },
            lessons_learned="Multi-provider workflow with latest models maximizes quality and performance"
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
    """🔄 Manually trigger project rescan and reindex."""
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
    """👥 List all 30+ team members with LATEST models (Oct 2025)."""
    team_info = {}
    
    for name, config in TEAM.items():
        team_info[name] = {
            "model": config.model,
            "provider": config.provider,
            "role": config.role,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
    
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
            "cerebras": clients.cerebras_available,
            "qwen": clients.qwen_available
        },
        "latest_models_included": {
            "openai": "GPT-5, GPT-5-mini, GPT-5-nano, GPT-4.1, O3/O4-mini-deep-research, GPT-OSS-120B",
            "groq": "Compound, Llama 4 Maverick/Scout, Kimi K2 (256k), GPT-OSS-120B",
            "google": "Gemini 2.5 Pro/Flash/Flash-Lite/Flash-Image",
            "cerebras": "Llama 4 Scout/Maverick, Qwen 3 235B/480B-Coder, GPT-OSS-120B",
            "qwen": "Qwen Max (1T+), Qwen Plus, Qwen3-Coder-480B, QwQ-Plus"
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
        "project_files_indexed": len(project_scanner.indexed_files),
        "latest_models_version": "October 2025 - All providers updated"
    }
    
    if clients.cerebras_available:
        cerebras_stats = {}
        for key_name, client_info in clients.cerebras_clients.items():
            cerebras_stats[key_name] = client_info["usage_count"]
        
        stats["cerebras"] = cerebras_stats
        stats["cerebras_total_calls"] = sum(cerebras_stats.values())
    else:
        stats["cerebras"] = "Not configured (using Groq fallback)"
    
    if clients.qwen_available:
        stats["qwen"] = "Active - Qwen Max (1T+), Qwen Plus, Coder-480B, QwQ"
    else:
        stats["qwen"] = "Not configured (using Groq fallback)"
    
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
    """🔍 Search indexed project files with semantic search."""
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
    """🧪 Test ALL API providers with LATEST models to verify they're working."""
    user = auth_manager.authenticate(api_key)
    if not user:
        raise PermissionError("Invalid API key")
    
    user_id = user.user_id
    test_prompt = "Respond with 'OK' and your model name if you receive this message."
    
    results = {}
    
    # Test OpenAI GPT-5
    try:
        if ctx:
            await ctx.info("Testing OpenAI GPT-5...")
        response, tokens = await llm_caller.call_agent("gpt5_flagship", test_prompt, user_id, False, ctx)
        results["openai_gpt5"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "gpt5_flagship", "model": "gpt-5"}
    except Exception as e:
        results["openai_gpt5"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test OpenAI O3
    try:
        if ctx:
            await ctx.info("Testing OpenAI O3 Deep Research...")
        response, tokens = await llm_caller.call_agent("o3_researcher", test_prompt, user_id, False, ctx)
        results["openai_o3"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "o3_researcher", "model": "o3-deep-research"}
    except Exception as e:
        results["openai_o3"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Groq Compound
    try:
        if ctx:
            await ctx.info("Testing Groq Compound...")
        response, tokens = await llm_caller.call_agent("groq_compound", test_prompt, user_id, False, ctx)
        results["groq_compound"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "groq_compound", "model": "groq/compound"}
    except Exception as e:
        results["groq_compound"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Groq Llama 4
    try:
        if ctx:
            await ctx.info("Testing Llama 4 Maverick...")
        response, tokens = await llm_caller.call_agent("llama4_maverick", test_prompt, user_id, False, ctx)
        results["llama4_maverick"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "llama4_maverick"}
    except Exception as e:
        results["llama4_maverick"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Google Gemini 2.5 Pro
    try:
        if ctx:
            await ctx.info("Testing Google Gemini 2.5 Pro...")
        response, tokens = await llm_caller.call_agent("gemini_pro", test_prompt, user_id, False, ctx)
        results["gemini_pro"] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": "gemini_pro", "model": "gemini-2.5-pro"}
    except Exception as e:
        results["gemini_pro"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Cerebras Llama 4
    try:
        if ctx:
            await ctx.info(f"Testing Cerebras Llama 4 Scout ({'configured' if clients.cerebras_available else 'fallback'})...")
        response, tokens = await llm_caller.call_agent("cerebras_llama4_scout", test_prompt, user_id, False, ctx)
        results["cerebras_llama4"] = {
            "status": "✅ SUCCESS",
            "tokens": tokens,
            "agent": "cerebras_llama4_scout",
            "using_cerebras": clients.cerebras_available,
            "fallback_to": "groq" if not clients.cerebras_available else None
        }
    except Exception as e:
        results["cerebras_llama4"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Cerebras Qwen 3
    try:
        if ctx:
            await ctx.info(f"Testing Cerebras Qwen 3 Coder 480B ({'configured' if clients.cerebras_available else 'fallback'})...")
        response, tokens = await llm_caller.call_agent("cerebras_qwen3_coder", test_prompt, user_id, False, ctx)
        results["cerebras_qwen3"] = {
            "status": "✅ SUCCESS",
            "tokens": tokens,
            "agent": "cerebras_qwen3_coder",
            "using_cerebras": clients.cerebras_available,
            "model": "qwen-3-coder-480b" if clients.cerebras_available else "qwen3-32b"
        }
    except Exception as e:
        results["cerebras_qwen3"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Test Qwen Max
    try:
        if ctx:
            await ctx.info(f"Testing Qwen Max (1T+) ({'configured' if clients.qwen_available else 'fallback'})...")
        response, tokens = await llm_caller.call_agent("qwen_max", test_prompt, user_id, False, ctx)
        results["qwen_max"] = {
            "status": "✅ SUCCESS",
            "tokens": tokens,
            "agent": "qwen_max",
            "using_qwen": clients.qwen_available,
            "model": "qwen-max (1T+)" if clients.qwen_available else "qwen3-32b",
            "fallback_to": "groq" if not clients.qwen_available else None
        }
    except Exception as e:
        results["qwen_max"] = {"status": "❌ FAILED", "error": str(e)}
    
    # Summary
    total_tested = len(results)
    successful = sum(1 for r in results.values() if "SUCCESS" in r.get("status", ""))
    
    return {
        "test_results": results,
        "summary": {
            "total_providers_tested": total_tested,
            "successful": successful,
            "failed": total_tested - successful,
            "success_rate": f"{(successful/total_tested)*100:.1f}%",
            "models_tested": "GPT-5, O3, Gemini 2.5 Pro, Llama 4, Qwen Max (1T+), Qwen 3 Coder 480B, Groq Compound"
        }
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Multi-Agent Development Team MCP Server v3.0")
    logger.info(f"👥 Team: {len(TEAM)} agents with LATEST models (October 2025)")
    logger.info(f"🔐 Authentication: Enabled")
    logger.info(f"🗄️ Qdrant: Connected (21 collections)")
    logger.info(f"📁 Project: {PROJECT_ROOT}")
    logger.info(f"📊 Files indexed: {len(project_scanner.indexed_files)}")
    
    logger.info("="*70)
    logger.info("LATEST MODELS (OCTOBER 2025):")
    logger.info("  OpenAI: GPT-5, GPT-5-mini/nano, GPT-4.1, O3/O4-mini-deep-research")
    logger.info("  Groq: Compound, Llama 4 Maverick/Scout, Kimi K2 (256k)")
    logger.info("  Google: Gemini 2.5 Pro/Flash/Flash-Lite/Flash-Image")
    logger.info(f"  Cerebras: {'✅ ACTIVE' if clients.cerebras_available else '⚠️ FALLBACK'} - Llama 4, Qwen 3 235B/480B")
    logger.info(f"  Qwen: {'✅ ACTIVE' if clients.qwen_available else '⚠️ FALLBACK'} - Max (1T+), Plus, Coder-480B, QwQ")
    logger.info("="*70)
    logger.info("ALL AGENTS GUARANTEED:")
    logger.info("  ✓ 30+ agents always registered with intelligent fallbacks")
    logger.info("  ✓ develop_feature() uses 12 stages across ALL providers")
    logger.info("  ✓ Use test_all_providers() to verify all APIs")
    logger.info("="*70)
    
    mcp.run()

