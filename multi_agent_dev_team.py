"""
multi_agent_dev_team.py

Enterprise multi-agent orchestration with comprehensive chat storage.

Version: 3.1.0
Date: October 7, 2025
Author: Nikolay Nikolov

Features:
- 30+ agents with latest models (GPT-5, Gemini 2.5, Llama 4, Qwen 3, etc.)
- Comprehensive chat storage - EVERY interaction saved with full metadata
- Chat search, analytics, ratings, and export
- Project file indexing in Qdrant
- Role-based access control
- Intelligent fallbacks for all providers

Tools (12 total):
1. develop_feature - 12-stage development workflow
2. create_user - User management
3. list_team_members - View all 30+ agents
4. get_api_usage_stats - Provider usage analytics
5. rescan_project - Reindex project files
6. search_project_files - Semantic file search
7. test_all_providers - Verify all APIs
8. search_chats - Semantic chat search
9. get_chat_history - User conversation history
10. rate_chat - Rate conversations
11. analyze_chat_patterns - Chat analytics
12. export_chats - Export conversations to JSON
"""

import fnmatch
import hashlib
import json
import logging
import os
import secrets
import subprocess
from dataclasses import dataclass
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

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
INDEXABLE_PATTERNS = ["*.go", "*.py", "*.sh", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.txt", "*.rs", "*.c", "*.h"]
EXCLUDE_DIRS = [".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", "vendor", "target", "build", "dist"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('multi_agent_dev_team.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

mcp = FastMCP(name="Multi-Agent Development Team", 
             instructions="Enterprise team with 30+ agents, latest models, and comprehensive chat storage")

class Role(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    WRITER = "writer"
    VIEWER = "viewer"

class Permission(Enum):
    CALL_LEADERSHIP, CALL_DEVELOPMENT, CALL_QA, CALL_DOCS, CALL_DEVOPS = "call_leadership", "call_development", "call_qa", "call_docs", "call_devops"
    VIEW_TASKS, CREATE_TASKS, DELETE_TASKS, VIEW_ANALYTICS, MANAGE_USERS = "view_tasks", "create_tasks", "delete_tasks", "view_analytics", "manage_users"

ROLE_PERMISSIONS = {
    Role.ADMIN: {Permission.CALL_LEADERSHIP, Permission.CALL_DEVELOPMENT, Permission.CALL_QA, Permission.CALL_DOCS, 
                 Permission.CALL_DEVOPS, Permission.VIEW_TASKS, Permission.CREATE_TASKS, Permission.DELETE_TASKS, 
                 Permission.VIEW_ANALYTICS, Permission.MANAGE_USERS},
    Role.DEVELOPER: {Permission.CALL_DEVELOPMENT, Permission.CALL_QA, Permission.CALL_DOCS, Permission.VIEW_TASKS, 
                     Permission.CREATE_TASKS, Permission.VIEW_ANALYTICS},
    Role.TESTER: {Permission.CALL_QA, Permission.VIEW_TASKS, Permission.VIEW_ANALYTICS},
    Role.WRITER: {Permission.CALL_DOCS, Permission.VIEW_TASKS},
    Role.VIEWER: {Permission.VIEW_TASKS, Permission.VIEW_ANALYTICS}
}

AGENT_PERMISSIONS = {a: Permission.CALL_DEVELOPMENT for a in ["gpt5_flagship", "gpt5_mini", "gpt5_nano", "gpt41_smart", "o3_researcher", 
    "o4_mini_researcher", "gpt_oss_120b", "groq_compound", "groq_compound_mini", "llama4_maverick", "llama4_scout", "llama33_versatile",
    "kimi_256k", "cerebras_llama4_scout", "cerebras_llama4_maverick", "cerebras_qwen3_235b", "cerebras_qwen3_coder", "qwen_max", 
    "qwen_plus", "qwen_coder_480b", "qwq_reasoning", "gemini_pro", "gemini_flash", "gemini_flash_lite"]}
AGENT_PERMISSIONS.update({"tech_lead": Permission.CALL_LEADERSHIP, "architect": Permission.CALL_LEADERSHIP,
    "qa_engineer": Permission.CALL_QA, "security_auditor": Permission.CALL_QA, "technical_writer": Permission.CALL_DOCS,
    "diagram_specialist": Permission.CALL_DOCS, "git_specialist": Permission.CALL_DEVOPS, "devops_engineer": Permission.CALL_DEVOPS})

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
        self.users, self.api_keys = {}, {}
        self._initialize_default_users()

    def _initialize_default_users(self):
        admin_key = os.getenv("ADMIN_API_KEY") or f"mcp_{secrets.token_urlsafe(32)}"
        admin_user = User("admin_1", "admin", Role.ADMIN, admin_key, datetime.now())
        self.users[admin_user.user_id] = admin_user
        self.api_keys[admin_key] = admin_user.user_id
        logger.info(f"🔑 Admin API Key: {admin_key}")

    def create_user(self, username: str, role: Role, created_by: str) -> User:
        if not self.has_permission(created_by, Permission.MANAGE_USERS):
            raise PermissionError("Only admins can create users")
        user_id, api_key = f"user_{secrets.token_hex(8)}", f"mcp_{secrets.token_urlsafe(32)}"
        user = User(user_id, username, role, api_key, datetime.now())
        self.users[user_id], self.api_keys[api_key] = user, user_id
        logger.info(f"✅ Created user: {username} ({role.value})")
        return user

    def authenticate(self, api_key: str) -> Optional[User]:
        user_id = self.api_keys.get(api_key)
        if not user_id: return None
        user = self.users.get(user_id)
        if not user or not user.is_active: return None
        user.last_access = datetime.now()
        return user

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        user = self.users.get(user_id)
        return user and user.is_active and permission in ROLE_PERMISSIONS.get(user.role, set())

    def can_call_agent(self, user_id: str, agent: str) -> bool:
        return self.has_permission(user_id, AGENT_PERMISSIONS.get(agent))

auth_manager = AuthManager()

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
    def from_env(cls):
        config = {k: os.getenv(v, '') for k, v in [('openai_key', 'OPENAI_API_KEY'), ('groq_key', 'GROQ_API_KEY'),
            ('google_key', 'GOOGLE_API_KEY'), ('gemini_key', 'GEMINI_API_KEY'), ('cerebras_personal', 'CEREBRAS_API_KEY_PERSONAL'),
            ('cerebras_book_expert', 'CEREBRAS_API_KEY_BOOK_EXPERT'), ('dashscope_key', 'DASHSCOPE_API_KEY'),
            ('qdrant_url', 'QDRANT_URL'), ('qdrant_key', 'QDRANT_API_KEY')]}
        required = [k for k in ['openai_key', 'groq_key', 'google_key', 'gemini_key', 'qdrant_url', 'qdrant_key'] if not config[k]]
        if required: logger.error(f"❌ Missing REQUIRED: {', '.join(required)}")
        return cls(**config)

config = APIConfig.from_env()

class APIClients:
    def __init__(self, config: APIConfig):
        self.config = config
        self.cerebras_available = bool(config.cerebras_personal or config.cerebras_book_expert)
        self.qwen_available = bool(config.dashscope_key)
        self._setup_clients()

    def _setup_clients(self):
        self.openai = openai.OpenAI(api_key=self.config.openai_key)
        self.groq = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.config.groq_key)
        genai.configure(api_key=self.config.gemini_key)
        genai_stable.configure(api_key=self.config.google_key)
        logger.info("✅ OpenAI, Groq, Google initialized")

        if self.cerebras_available:
            self.cerebras_clients = {}
            if self.config.cerebras_personal:
                self.cerebras_clients["personal"] = {"client": openai.OpenAI(base_url="https://api.cerebras.ai/v1", 
                    api_key=self.config.cerebras_personal), "usage_count": 0}
            if self.config.cerebras_book_expert:
                self.cerebras_clients["book_expert"] = {"client": openai.OpenAI(base_url="https://api.cerebras.ai/v1",
                    api_key=self.config.cerebras_book_expert), "usage_count": 0}
            logger.info(f"✅ Cerebras: {len(self.cerebras_clients)} key(s)")

        if self.qwen_available:
            self.qwen = openai.OpenAI(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", api_key=self.config.dashscope_key)
            logger.info("✅ Qwen initialized")

    def get_cerebras_client(self):
        if not self.cerebras_available or not self.cerebras_clients: return None
        if len(self.cerebras_clients) == 1:
            key = list(self.cerebras_clients.keys())[0]
            self.cerebras_clients[key]["usage_count"] += 1
            return self.cerebras_clients[key]["client"]
        p, b = self.cerebras_clients.get("personal"), self.cerebras_clients.get("book_expert")
        if p and b:
            selected = p if p["usage_count"] <= b["usage_count"] else b
            selected["usage_count"] += 1
            return selected["client"]
        selected = p or b
        selected["usage_count"] += 1
        return selected["client"]

clients = APIClients(config)

@dataclass
class AgentConfig:
    model: str
    provider: str
    role: str
    temperature: float = 0.3
    max_tokens: int = 4000

TEAM = {
    "gpt5_flagship": AgentConfig("gpt-5", "openai", "Flagship GPT-5", 0.3, 16000),
    "gpt5_mini": AgentConfig("gpt-5-mini", "openai", "Fast GPT-5", 0.3, 12000),
    "gpt5_nano": AgentConfig("gpt-5-nano", "openai", "Fastest GPT-5", 0.4, 8000),
    "gpt41_smart": AgentConfig("gpt-4.1", "openai", "Smartest non-reasoning", 0.2, 12000),
    "o3_researcher": AgentConfig("o3-deep-research", "openai", "Deep research", 0.1, 16000),
    "o4_mini_researcher": AgentConfig("o4-mini-deep-research", "openai", "Fast research", 0.2, 12000),
    "gpt_oss_120b": AgentConfig("gpt-oss-120b", "openai", "Open-weight 120B", 0.3, 12000),
    "groq_compound": AgentConfig("groq/compound", "groq", "Compound system", 0.5, 12000),
    "groq_compound_mini": AgentConfig("groq/compound-mini", "groq", "Fast compound", 0.5, 8000),
    "groq_gpt_oss": AgentConfig("openai/gpt-oss-120b", "groq", "GPT-OSS on Groq", 0.3, 12000),
    "llama4_maverick": AgentConfig("meta-llama/llama-4-maverick-17b-128e-instruct", "groq", "Llama 4 heavy", 0.4, 10000),
    "llama4_scout": AgentConfig("meta-llama/llama-4-scout-17b-16e-instruct", "groq", "Llama 4 planning", 0.4, 10000),
    "llama33_versatile": AgentConfig("llama-3.3-70b-versatile", "groq", "Llama 3.3 70B", 0.4, 12000),
    "kimi_256k": AgentConfig("moonshotai/kimi-k2-instruct-0905", "groq", "256k context", 0.4, 16000),
    "groq_qwen": AgentConfig("qwen/qwen3-32b", "groq", "Qwen 3 on Groq", 0.3, 8000),
    "gemini_pro": AgentConfig("gemini-2.5-pro", "google_stable", "Gemini Pro", 0.2, 16000),
    "gemini_flash": AgentConfig("gemini-2.5-flash", "google_stable", "Gemini Flash", 0.3, 16000),
    "gemini_flash_lite": AgentConfig("gemini-2.5-flash-lite", "google_stable", "Ultra-fast", 0.3, 12000),
    "qa_engineer": AgentConfig("gemini-2.5-flash", "google_stable", "QA Testing", 0.2, 8000),
    "security_auditor": AgentConfig("gemini-2.5-flash", "google_stable", "Security", 0.2, 8000),
    "technical_writer": AgentConfig("gemini-2.5-pro", "google_stable", "Documentation", 0.3, 12000),
    "git_specialist": AgentConfig("gemini-2.5-flash-lite", "google_stable", "Git", 0.2, 4000),
    "devops_engineer": AgentConfig("gemini-2.5-flash", "google_stable", "CI/CD", 0.2, 8000),
    "cerebras_llama4_scout": AgentConfig("llama-4-scout-17b-16e-instruct" if clients.cerebras_available else "meta-llama/llama-4-scout-17b-16e-instruct",
        "cerebras" if clients.cerebras_available else "groq", "Cerebras Llama 4 Scout", 0.4, 10000),
    "cerebras_llama4_maverick": AgentConfig("llama-4-maverick-17b-128e-instruct" if clients.cerebras_available else "meta-llama/llama-4-maverick-17b-128e-instruct",
        "cerebras" if clients.cerebras_available else "groq", "Cerebras Llama 4 Maverick", 0.4, 12000),
    "cerebras_qwen3_235b": AgentConfig("qwen-3-235b-a22b-instruct-2507" if clients.cerebras_available else "qwen/qwen3-32b",
        "cerebras" if clients.cerebras_available else "groq", "Qwen 3 235B", 0.3, 16000),
    "cerebras_qwen3_coder": AgentConfig("qwen-3-coder-480b" if clients.cerebras_available else "qwen/qwen3-32b",
        "cerebras" if clients.cerebras_available else "groq", "Qwen 3 Coder 480B", 0.3, 16000),
    "qwen_max": AgentConfig("qwen-max" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq", "Qwen Max 1T+", 0.3, 16000),
    "qwen_plus": AgentConfig("qwen-plus" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq", "Qwen Plus", 0.3, 12000),
    "qwen_coder_480b": AgentConfig("qwen3-coder-480b-a35b-instruct" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq", "Qwen Coder 480B", 0.3, 16000),
    "qwq_reasoning": AgentConfig("qwq-plus" if clients.qwen_available else "qwen/qwen3-32b",
        "qwen" if clients.qwen_available else "groq", "QwQ reasoning", 0.2, 16000),
}

logger.info(f"✅ Team: {len(TEAM)} agents - Cerebras {'ACTIVE' if clients.cerebras_available else 'FALLBACK'}, Qwen {'ACTIVE' if clients.qwen_available else 'FALLBACK'}")

class ProjectScanner:
    def __init__(self, root_dir: str, qdrant_manager):
        self.root_dir, self.qdrant, self.indexed_files = Path(root_dir), qdrant_manager, {}

    def should_index(self, filepath: Path):
        return not any(part in EXCLUDE_DIRS for part in filepath.parts) and any(fnmatch.fnmatch(filepath.name, p) for p in INDEXABLE_PATTERNS)

    def scan_and_index(self):
        logger.info(f"📁 Scanning: {self.root_dir}")
        indexed, skipped, updated = 0, 0, 0
        try:
            for filepath in self.root_dir.rglob('*'):
                if not filepath.is_file() or not self.should_index(filepath): continue
                try:
                    mtime, rpath = filepath.stat().st_mtime, str(filepath.relative_to(self.root_dir))
                    if rpath in self.indexed_files:
                        if self.indexed_files[rpath] >= mtime: 
                            skipped += 1
                            continue
                        updated += 1
                    else: indexed += 1

                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    self.qdrant.store("project_files", content, {"file_path": rpath, "file_type": filepath.suffix[1:] or "unknown",
                        "file_name": filepath.name, "size_bytes": len(content), "modified_time": mtime, "indexed_at": datetime.now().isoformat()})
                    self.indexed_files[rpath] = mtime
                except Exception as e: logger.warning(f"⚠️ Error: {e}")
        except Exception as e: logger.error(f"❌ Scan failed: {e}")
        logger.info(f"✅ Scan: {indexed} new, {updated} updated, {skipped} skipped")
        return {"files_indexed": indexed, "files_updated": updated, "files_skipped": skipped, "total_tracked": len(self.indexed_files)}

class QdrantManager:
    CACHE_THRESHOLD, CONTEXT_THRESHOLD = 0.95, 0.85

    def __init__(self, url: str, api_key: str):
        self.url, self.api_key, self.client, self.embedding_model = url, api_key, None, None
        self._initialize()

    def _initialize(self):
        logger.info("🔗 Connecting to Qdrant...")
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
        logger.info(f"✅ Connected! Collections: {len(self.client.get_collections().collections)}")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
        self._setup_collections()

    def _setup_collections(self):
        collections = ["tasks", "architectures", "implementations", "code_reviews", "test_strategies", "security_findings", "diagrams", 
            "documentation", "git_commits", "ci_cd_configs", "llm_cache", "feedback", "context_memory", "optimization_patterns", 
            "groq_workflows", "conversations", "success_logs", "failure_logs", "learning_patterns", "performance_metrics", 
            "project_files", "chats", "chat_threads", "chat_feedback", "chat_analytics"]
        existing = {c.name for c in self.client.get_collections().collections}
        for name in collections:
            if name not in existing:
                try:
                    self.client.create_collection(name, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
                except Exception as e: logger.warning(f"⚠️ {name}: {e}")

    def get_embedding(self, text: str):
        try: return self.embedding_model.encode(text[:8000]).tolist()
        except: return []

    def check_cache(self, prompt: str, model: str, temperature: float):
        try:
            emb = self.get_embedding(prompt)
            if not emb: return None
            results = self.client.search("llm_cache", query_vector=emb, limit=1, score_threshold=self.CACHE_THRESHOLD,
                query_filter=Filter(must=[FieldCondition(key="model", match=MatchValue(value=model)),
                                          FieldCondition(key="temperature", match=MatchValue(value=temperature))]))
            if results:
                logger.info(f"🎯 Cache HIT! ({results[0].score:.3f})")
                return results[0].payload["response"]
        except: pass
        return None

    def store_in_cache(self, prompt: str, response: str, model: str, temp: float, tokens: int):
        try:
            cid, emb = hashlib.md5(f"{prompt}{model}{datetime.now()}".encode()).hexdigest(), self.get_embedding(prompt)
            if emb:
                self.client.upsert("llm_cache", points=[PointStruct(id=cid, vector=emb, payload={"prompt": prompt[:500], "response": response,
                    "model": model, "temperature": temp, "tokens_used": tokens, "timestamp": datetime.now().isoformat()})])
        except: pass

    def store(self, collection: str, text: str, metadata: dict):
        try:
            vid, emb = hashlib.md5(f"{text}{datetime.now()}".encode()).hexdigest(), self.get_embedding(text)
            if not emb: return ""
            metadata["stored_at"], metadata["text_preview"] = datetime.now().isoformat(), text[:200]
            self.client.upsert(collection, points=[PointStruct(id=vid, vector=emb, payload=metadata)])
            return vid
        except Exception as e: 
            logger.warning(f"Store failed: {e}")
            return ""

    def store_chat(self, prompt: str, response: str, agent_name: str, model: str, provider: str, user_id: str, tokens_used: int,
                   temperature: float, duration_ms: int, task_id=None, stage_name=None, context_used=False, cache_hit=False, parent_chat_id=None):
        try:
            chat_id = hashlib.md5(f"{user_id}{agent_name}{datetime.now()}".encode()).hexdigest()
            combined, emb = f"User: {prompt}\nAgent ({agent_name}): {response}", self.get_embedding(f"{prompt} {response}")
            if not emb: return ""
            metadata = {"chat_id": chat_id, "user_id": user_id, "agent_name": agent_name, "model": model, "provider": provider,
                "prompt": prompt[:2000], "response": response[:2000], "prompt_length": len(prompt), "response_length": len(response),
                "tokens_used": tokens_used, "temperature": temperature, "duration_ms": duration_ms, "timestamp": datetime.now().isoformat(),
                "context_used": context_used, "cache_hit": cache_hit, "task_id": task_id or "", "stage_name": stage_name or "",
                "parent_chat_id": parent_chat_id or "", "user_rating": 0, "user_feedback": ""}
            self.client.upsert("chats", points=[PointStruct(id=chat_id, vector=emb, payload=metadata)])
            logger.debug(f"💬 Chat stored: {chat_id}")
            return chat_id
        except Exception as e:
            logger.warning(f"Chat store failed: {e}")
            return ""

    def search_chats(self, query: str, user_id=None, limit=5):
        try:
            emb = self.get_embedding(query)
            if not emb: return []
            qf = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]) if user_id else None
            results = self.client.search("chats", query_vector=emb, limit=limit, score_threshold=self.CONTEXT_THRESHOLD, query_filter=qf)
            return [{"score": r.score, "payload": r.payload, "id": r.id} for r in results]
        except: return []

    def get_user_chats(self, user_id: str, limit=20):
        try:
            results = self.client.scroll("chats", scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
                limit=limit, with_vectors=False)
            chats = [{"chat_id": p.id, "payload": p.payload} for p in results[0]]
            chats.sort(key=lambda x: x["payload"].get("timestamp", ""), reverse=True)
            return chats
        except: return []

    def get_task_chats(self, task_id: str):
        try:
            results = self.client.scroll("chats", scroll_filter=Filter(must=[FieldCondition(key="task_id", match=MatchValue(value=task_id))]),
                limit=100, with_vectors=False)
            chats = [{"chat_id": p.id, "payload": p.payload} for p in results[0]]
            chats.sort(key=lambda x: x["payload"].get("timestamp", ""))
            return chats
        except: return []

    def store_chat_feedback(self, chat_id: str, rating: int, feedback: str, user_id: str):
        try:
            self.client.set_payload("chats", payload={"user_rating": rating, "user_feedback": feedback, 
                "feedback_timestamp": datetime.now().isoformat()}, points=[chat_id])
            fid, ft, emb = hashlib.md5(f"{chat_id}{datetime.now()}".encode()).hexdigest(), f"Rating: {rating}/5. {feedback}", self.get_embedding(ft)
            if emb:
                self.client.upsert("chat_feedback", points=[PointStruct(id=fid, vector=emb, payload={"feedback_id": fid, "chat_id": chat_id,
                    "user_id": user_id, "rating": rating, "feedback": feedback, "timestamp": datetime.now().isoformat()})])
            logger.info(f"⭐ Feedback: {rating}/5")
            return True
        except: return False

    def get_chat_analytics(self, user_id=None):
        try:
            chats = self.get_user_chats(user_id, 1000) if user_id else [{"chat_id": p.id, "payload": p.payload} 
                for p in self.client.scroll("chats", limit=1000, with_vectors=False)[0]]
            if not chats: return {"error": "No chats"}
            total, tokens = len(chats), sum(c["payload"].get("tokens_used", 0) for c in chats)
            agent_usage = {}
            for c in chats:
                a = c["payload"].get("agent_name", "unknown")
                agent_usage[a] = agent_usage.get(a, 0) + 1
            provider_usage = {}
            for c in chats:
                p = c["payload"].get("provider", "unknown")
                provider_usage[p] = provider_usage.get(p, 0) + 1
            rated = [c for c in chats if c["payload"].get("user_rating", 0) > 0]
            avg_rating = sum(c["payload"].get("user_rating", 0) for c in rated) / len(rated) if rated else 0
            cache_hits = sum(1 for c in chats if c["payload"].get("cache_hit"))
            return {"total_chats": total, "total_tokens_used": tokens, "average_tokens_per_chat": round(tokens/total, 2),
                "agent_usage": agent_usage, "provider_usage": provider_usage, "average_rating": round(avg_rating, 2),
                "rated_chats": len(rated), "cache_hit_rate": round((cache_hits/total*100), 2), "cache_hits": cache_hits}
        except Exception as e: return {"error": str(e)}

    def log_success(self, task_id: str, success_type: str, description: str, agent: str, outcome: str, metrics: dict, lessons: str):
        self.store("success_logs", f"{success_type}: {description}. {outcome}. {lessons}", {"task_id": task_id, "success_type": success_type,
            "description": description, "agent": agent, "outcome": outcome, "metrics": metrics, "lessons_learned": lessons})
        logger.info(f"✅ Success: {success_type}")

    def log_failure(self, task_id: str, failure_type: str, description: str, agent: str, error: str, root_cause: str, resolution: str, lessons: str):
        self.store("failure_logs", f"{failure_type}: {description}. {error}. {root_cause}", {"task_id": task_id, "failure_type": failure_type,
            "description": description, "agent": agent, "error_message": error, "root_cause": root_cause, "resolution": resolution, "lessons_learned": lessons})
        logger.warning(f"❌ Failure: {failure_type}")

    def search(self, collection: str, query: str, limit=3, threshold=None):
        try:
            emb = self.get_embedding(query)
            if not emb: return []
            results = self.client.search(collection, query_vector=emb, limit=limit, score_threshold=threshold or self.CONTEXT_THRESHOLD)
            return [{"score": r.score, "payload": r.payload, "id": r.id} for r in results]
        except: return []

    def get_relevant_context(self, task_desc: str, collections: List[str], limit=2):
        context = {}
        for c in collections:
            results = self.search(c, task_desc, limit=limit)
            if results: context[c] = results
        return context

qdrant = QdrantManager(config.qdrant_url, config.qdrant_key)
project_scanner = ProjectScanner(PROJECT_ROOT, qdrant)
logger.info(f"🚀 Scanning: {PROJECT_ROOT}")
scan_results = project_scanner.scan_and_index()
logger.info(f"📊 Scan: {json.dumps(scan_results)}")

class LLMCaller:
    def __init__(self, clients: APIClients, qdrant: QdrantManager, auth: AuthManager):
        self.clients, self.qdrant, self.auth = clients, qdrant, auth

    async def call_agent(self, agent: str, prompt: str, user_id: str, use_context=True, ctx=None, task_id=None, stage_name=None, parent_chat_id=None):
        start_time = datetime.now()
        if not self.auth.can_call_agent(user_id, agent):
            raise PermissionError(f"Access denied: cannot call {agent}")
        if agent not in TEAM: raise ValueError(f"Unknown agent: {agent}")

        agent_config, cache_hit, context_used, original_prompt = TEAM[agent], False, False, prompt

        if use_context:
            pc = self.qdrant.get_relevant_context(prompt[:500], ["project_files", "implementations", "success_logs", "failure_logs"], 2)
            if pc:
                cs = "\n\n## Context:\n"
                for coll, items in pc.items():
                    for item in items:
                        cs += f"\n[{coll}, {item['score']:.2f}]\n"
                        if coll == "project_files": cs += f"File: {item['payload'].get('file_path')}\n"
                        cs += f"{item['payload'].get('text_preview', '')}...\n"
                prompt, context_used = prompt + cs, True

        cached = self.qdrant.check_cache(prompt, agent_config.model, agent_config.temperature)
        if cached:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            chat_id = self.qdrant.store_chat(original_prompt, cached, agent, agent_config.model, agent_config.provider, user_id, 0,
                agent_config.temperature, duration_ms, task_id, stage_name, context_used, True, parent_chat_id)
            return cached, 0, chat_id

        logger.info(f"🤖 {agent} ({agent_config.model} via {agent_config.provider})")

        response_text, tokens = await self._call_provider(agent_config.provider, agent_config.model, prompt, 
            agent_config.temperature, agent_config.max_tokens)
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        self.qdrant.store_in_cache(prompt, response_text, agent_config.model, agent_config.temperature, tokens)
        chat_id = self.qdrant.store_chat(original_prompt, response_text, agent, agent_config.model, agent_config.provider, user_id, tokens,
            agent_config.temperature, duration_ms, task_id, stage_name, context_used, False, parent_chat_id)

        return response_text, tokens, chat_id

    async def _call_provider(self, provider: str, model: str, prompt: str, temp: float, max_tokens: int):
        if provider == "openai":
            r = self.clients.openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], 
                temperature=temp, max_tokens=max_tokens)
            return r.choices[0].message.content, r.usage.total_tokens
        elif provider == "groq":
            r = self.clients.groq.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temp, max_tokens=max_tokens)
            return r.choices[0].message.content, r.usage.total_tokens
        elif provider == "cerebras":
            c = self.clients.get_cerebras_client()
            if not c: raise ValueError("Cerebras unavailable")
            r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temp, max_tokens=max_tokens)
            return r.choices[0].message.content, r.usage.total_tokens
        elif provider == "qwen":
            if not self.clients.qwen_available: raise ValueError("Qwen unavailable")
            r = self.clients.qwen.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temp, max_tokens=max_tokens)
            return r.choices[0].message.content, r.usage.total_tokens
        elif provider == "google_stable":
            gm = genai_stable.GenerativeModel(model)
            r = gm.generate_content(prompt, generation_config=genai_stable.GenerationConfig(temperature=temp, max_output_tokens=max_tokens))
            return r.text, len(prompt.split()) + len(r.text.split())
        elif provider == "google_preview":
            gm = genai.GenerativeModel(model)
            r = gm.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=temp, max_output_tokens=max_tokens))
            return r.text, len(prompt.split()) + len(r.text.split())
        raise ValueError(f"Unknown provider: {provider}")

llm_caller = LLMCaller(clients, qdrant, auth_manager)

TASKS, TOTAL_TOKENS_USED = {}, 0

@mcp.tool
async def develop_feature(description: str, requirements: List[str], api_key: str, ctx: Context = None):
    """🚀 Complete 12-stage development workflow using ALL latest models across providers."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if not auth_manager.has_permission(user.user_id, Permission.CREATE_TASKS): raise PermissionError("Insufficient permissions")

    user_id, task_id = user.user_id, hashlib.md5(f"{description}{datetime.now()}".encode()).hexdigest()[:12]
    global TOTAL_TOKENS_USED

    if ctx: await ctx.info(f"🚀 Task: {task_id}")

    task = {"task_id": task_id, "description": description, "requirements": requirements, "user_id": user_id, 
        "stages": {}, "created_at": datetime.now().isoformat()}
    TASKS[task_id], task_tokens = task, 0
    qdrant.store("tasks", f"{description} {' '.join(requirements)}", {"task_id": task_id, "user_id": user_id})

    # Stage 1: O3 Research
    if ctx: await ctx.info("🔬 O3 Deep Research")
    research, tokens, _ = await llm_caller.call_agent("o3_researcher", f"Research: {description}\nReqs: {', '.join(requirements)}", 
        user_id, True, ctx, task_id, "research")
    task["stages"]["research"] = {"result": research, "tokens": tokens, "agent": "o3_researcher", "provider": "openai"}
    task_tokens += tokens

    # Stage 2: GPT-5 Architecture
    if ctx: await ctx.info("🏗️ GPT-5 Architecture")
    arch, tokens, _ = await llm_caller.call_agent("gpt5_flagship", f"Architecture: {research[:1500]}", user_id, True, ctx, task_id, "architecture")
    task["stages"]["architecture"] = {"result": arch, "tokens": tokens, "agent": "gpt5_flagship", "provider": "openai"}
    task_tokens += tokens
    qdrant.store("architectures", arch[:2000], {"task_id": task_id})

    # Stage 3: Cerebras Qwen Coder
    if ctx: await ctx.info(f"💻 Qwen Coder 480B ({'Cerebras' if clients.cerebras_available else 'Groq'})")
    impl, tokens, _ = await llm_caller.call_agent("cerebras_qwen3_coder", f"Implement: {arch[:1500]}", user_id, True, ctx, task_id, "implementation")
    task["stages"]["implementation"] = {"result": impl, "tokens": tokens, "agent": "cerebras_qwen3_coder", "provider": TEAM["cerebras_qwen3_coder"].provider}
    task_tokens += tokens
    qdrant.store("implementations", impl[:2000], {"task_id": task_id})

    # Stage 4: Qwen Max Optimization
    if ctx: await ctx.info(f"🧮 Qwen Max ({'Qwen' if clients.qwen_available else 'Groq'})")
    opt, tokens, _ = await llm_caller.call_agent("qwen_max", f"Optimize: {impl[:1500]}", user_id, True, ctx, task_id, "optimization")
    task["stages"]["optimization"] = {"result": opt, "tokens": tokens, "agent": "qwen_max", "provider": TEAM["qwen_max"].provider}
    task_tokens += tokens

    # Stage 5: QwQ Reasoning
    if ctx: await ctx.info("🧠 QwQ Reasoning")
    reasoning, tokens, _ = await llm_caller.call_agent("qwq_reasoning", f"Reason: {opt[:1000]}", user_id, True, ctx, task_id, "reasoning")
    task["stages"]["reasoning"] = {"result": reasoning, "tokens": tokens, "agent": "qwq_reasoning", "provider": TEAM["qwq_reasoning"].provider}
    task_tokens += tokens

    # Stage 6: Groq Compound
    if ctx: await ctx.info("⚡ Groq Compound")
    proto, tokens, _ = await llm_caller.call_agent("groq_compound", f"Prototype: {reasoning[:1000]}", user_id, True, ctx, task_id, "prototype")
    task["stages"]["prototype"] = {"result": proto, "tokens": tokens, "agent": "groq_compound", "provider": "groq"}
    task_tokens += tokens

    # Stage 7: Llama 4
    if ctx: await ctx.info("🦙 Llama 4 Maverick")
    heavy, tokens, _ = await llm_caller.call_agent("llama4_maverick", f"Production: {proto[:1000]}", user_id, True, ctx, task_id, "production")
    task["stages"]["production"] = {"result": heavy, "tokens": tokens, "agent": "llama4_maverick", "provider": "groq"}
    task_tokens += tokens

    # Stage 8: GPT-4.1 Review
    if ctx: await ctx.info("👀 GPT-4.1 Review")
    review, tokens, _ = await llm_caller.call_agent("gpt41_smart", f"Review: {heavy[:1000]}", user_id, True, ctx, task_id, "review")
    task["stages"]["review"] = {"result": review, "tokens": tokens, "agent": "gpt41_smart", "provider": "openai"}
    task_tokens += tokens
    qdrant.store("code_reviews", review[:2000], {"task_id": task_id})

    # Stage 9-12: QA, Security, Docs, Polish
    for stage, agent, prompt in [("testing", "qa_engineer", "Test"), ("security", "security_auditor", "Audit"),
        ("documentation", "technical_writer", "Document"), ("polish", "gpt5_mini", "Polish")]:
        if ctx: await ctx.info(f"Processing {stage}...")
        result, tokens, _ = await llm_caller.call_agent(agent, f"{prompt}: {heavy[:1000]}", user_id, True, ctx, task_id, stage)
        task["stages"][stage] = {"result": result, "tokens": tokens, "agent": agent, "provider": TEAM[agent].provider}
        task_tokens += tokens

    provider_usage = {}
    for _, sd in task["stages"].items():
        p = sd.get("provider", "unknown")
        if p not in provider_usage: provider_usage[p] = {"tokens": 0, "agents": []}
        provider_usage[p]["tokens"] += sd.get("tokens", 0)
        provider_usage[p]["agents"].append(f"{sd.get('agent')}({sd.get('tokens')})")

    qdrant.log_success(task_id, "feature_complete", description, "multi_agent_system", "12-stage workflow complete",
        {"total_tokens": task_tokens, "stages": len(task["stages"]), "provider_usage": provider_usage}, "Multi-provider workflow successful")

    task["tokens_used"], task["status"], task["provider_usage"] = task_tokens, "complete", provider_usage
    TOTAL_TOKENS_USED += task_tokens

    if ctx: 
        await ctx.info(f"✅ Complete! {task_tokens} tokens")
        await ctx.info(f"📊 Providers: {json.dumps(provider_usage)}")

    return task

@mcp.tool
async def search_chats(query: str, api_key: str, user_id_filter: Optional[str] = None, limit: int = 5, ctx: Context = None):
    """🔍 Search all chats semantically."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if user.role != Role.ADMIN and user_id_filter != user.user_id: user_id_filter = user.user_id

    results = qdrant.search_chats(query, user_id_filter, limit)
    return {"query": query, "user_filter": user_id_filter, "results_count": len(results), "results": [
        {"chat_id": r["id"], "similarity": r["score"], "agent": r["payload"].get("agent_name"), "model": r["payload"].get("model"),
         "timestamp": r["payload"].get("timestamp"), "prompt_preview": r["payload"].get("prompt", "")[:200],
         "response_preview": r["payload"].get("response", "")[:200], "tokens": r["payload"].get("tokens_used"),
         "rating": r["payload"].get("user_rating", 0)} for r in results]}

@mcp.tool
async def get_chat_history(api_key: str, user_id_filter: Optional[str] = None, limit: int = 20, ctx: Context = None):
    """📜 Get chat history for a user."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if user.role != Role.ADMIN: user_id_filter = user.user_id
    target = user_id_filter or user.user_id

    chats = qdrant.get_user_chats(target, limit)
    return {"user_id": target, "total_chats": len(chats), "chats": [
        {"chat_id": c["chat_id"], "agent": c["payload"].get("agent_name"), "model": c["payload"].get("model"),
         "provider": c["payload"].get("provider"), "timestamp": c["payload"].get("timestamp"),
         "prompt": c["payload"].get("prompt", "")[:500], "response": c["payload"].get("response", "")[:500],
         "tokens": c["payload"].get("tokens_used"), "duration_ms": c["payload"].get("duration_ms"),
         "task_id": c["payload"].get("task_id"), "stage_name": c["payload"].get("stage_name"),
         "cache_hit": c["payload"].get("cache_hit"), "rating": c["payload"].get("user_rating", 0)} for c in chats]}

@mcp.tool
async def rate_chat(chat_id: str, rating: int, feedback: str, api_key: str, ctx: Context = None):
    """⭐ Rate a chat (1-5 stars) with feedback."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if rating < 1 or rating > 5: raise ValueError("Rating must be 1-5")

    success = qdrant.store_chat_feedback(chat_id, rating, feedback, user.user_id)
    return {"status": "success" if success else "failed", "chat_id": chat_id, "rating": rating, "feedback": feedback,
        "message": "Thank you!" if success else "Failed"}

@mcp.tool
async def analyze_chat_patterns(api_key: str, user_id_filter: Optional[str] = None, ctx: Context = None):
    """📊 Analyze chat patterns and get insights."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if not auth_manager.has_permission(user.user_id, Permission.VIEW_ANALYTICS): raise PermissionError("Insufficient permissions")
    if user.role != Role.ADMIN: user_id_filter = user.user_id

    analytics = qdrant.get_chat_analytics(user_id_filter)
    return {"user_filter": user_id_filter or "all_users", "analytics": analytics}

@mcp.tool
async def export_chats(api_key: str, user_id_filter: Optional[str] = None, task_id: Optional[str] = None, limit: int = 100, ctx: Context = None):
    """📦 Export chats to JSON."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if user.role != Role.ADMIN: user_id_filter = user.user_id

    chats = qdrant.get_task_chats(task_id) if task_id else qdrant.get_user_chats(user_id_filter, limit) if user_id_filter else []
    if not chats: raise ValueError("No chats found")

    return {"export_timestamp": datetime.now().isoformat(), "export_by": user.user_id, "filter": {"user_id": user_id_filter, "task_id": task_id},
        "total_chats": len(chats), "chats": [{"chat_id": c["chat_id"], "timestamp": c["payload"].get("timestamp"),
        "agent": c["payload"].get("agent_name"), "model": c["payload"].get("model"), "provider": c["payload"].get("provider"),
        "user_id": c["payload"].get("user_id"), "task_id": c["payload"].get("task_id"), "stage_name": c["payload"].get("stage_name"),
        "prompt": c["payload"].get("prompt"), "response": c["payload"].get("response"), "tokens_used": c["payload"].get("tokens_used"),
        "duration_ms": c["payload"].get("duration_ms"), "temperature": c["payload"].get("temperature"),
        "context_used": c["payload"].get("context_used"), "cache_hit": c["payload"].get("cache_hit"),
        "user_rating": c["payload"].get("user_rating"), "user_feedback": c["payload"].get("user_feedback")} for c in chats]}

@mcp.tool
async def rescan_project(api_key: str, ctx: Context = None):
    """🔄 Rescan project and reindex files."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    return project_scanner.scan_and_index()

@mcp.tool
async def create_user(username: str, role: str, api_key: str, ctx: Context = None):
    """👤 Create new user (Admin only)."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    new_user = auth_manager.create_user(username, Role(role), user.user_id)
    return {"user_id": new_user.user_id, "username": new_user.username, "role": new_user.role.value,
        "api_key": new_user.api_key, "created_at": new_user.created_at.isoformat()}

@mcp.tool
async def list_team_members(ctx: Context = None):
    """👥 List all team members with latest models."""
    provider_counts = {}
    for cfg in TEAM.values():
        provider_counts[cfg.provider] = provider_counts.get(cfg.provider, 0) + 1

    return {"team": {n: {"model": c.model, "provider": c.provider, "role": c.role, "temperature": c.temperature, 
        "max_tokens": c.max_tokens} for n, c in TEAM.items()}, "total_members": len(TEAM), "provider_distribution": provider_counts,
        "providers_active": {"openai": True, "groq": True, "google_stable": True, "cerebras": clients.cerebras_available, 
        "qwen": clients.qwen_available}, "latest_models": {"openai": "GPT-5, GPT-5-mini/nano, GPT-4.1, O3/O4, GPT-OSS-120B",
        "groq": "Compound, Llama 4, Kimi K2 (256k)", "google": "Gemini 2.5 Pro/Flash/Lite",
        "cerebras": "Llama 4, Qwen 3 235B/480B", "qwen": "Max (1T+), Plus, Coder-480B, QwQ"}}

@mcp.tool
async def get_api_usage_stats(api_key: str, ctx: Context = None):
    """📊 Get comprehensive API usage stats."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if not auth_manager.has_permission(user.user_id, Permission.VIEW_ANALYTICS): raise PermissionError("Insufficient permissions")

    stats = {"total_tokens_used": TOTAL_TOKENS_USED, "team_size": len(TEAM), "active_tasks": len(TASKS), "user_id": user.user_id,
        "project_files_indexed": len(project_scanner.indexed_files), "models_version": "October 2025"}

    if clients.cerebras_available:
        stats["cerebras"] = {k: v["usage_count"] for k, v in clients.cerebras_clients.items()}
        stats["cerebras_total"] = sum(stats["cerebras"].values())
    else: stats["cerebras"] = "Not configured (Groq fallback)"

    stats["qwen"] = "Active" if clients.qwen_available else "Not configured (Groq fallback)"

    provider_agents = {}
    for a, c in TEAM.items():
        if c.provider not in provider_agents: provider_agents[c.provider] = []
        provider_agents[c.provider].append(a)
    stats["provider_agent_distribution"] = {p: len(a) for p, a in provider_agents.items()}
    return stats

@mcp.tool
async def search_project_files(query: str, api_key: str, limit: int = 5, ctx: Context = None):
    """🔍 Search indexed project files."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    results = qdrant.search("project_files", query, limit)
    return {"query": query, "results": [{"file_path": r["payload"].get("file_path"), "file_type": r["payload"].get("file_type"),
        "similarity": r["score"], "preview": r["payload"].get("text_preview")} for r in results]}

@mcp.tool
async def test_all_providers(api_key: str, ctx: Context = None):
    """🧪 Test ALL providers with latest models."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")

    user_id, test_prompt, results = user.user_id, "Respond with 'OK' and your model name.", {}

    for provider, agent, model in [("openai_gpt5", "gpt5_flagship", "gpt-5"), ("openai_o3", "o3_researcher", "o3-deep-research"),
        ("groq_compound", "groq_compound", "groq/compound"), ("llama4", "llama4_maverick", "llama-4"),
        ("gemini_pro", "gemini_pro", "gemini-2.5-pro")]:
        try:
            if ctx: await ctx.info(f"Testing {provider}...")
            response, tokens, _ = await llm_caller.call_agent(agent, test_prompt, user_id, False, ctx)
            results[provider] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": agent, "model": model}
        except Exception as e: results[provider] = {"status": "❌ FAILED", "error": str(e)}

    for provider, agent in [("cerebras", "cerebras_llama4_scout"), ("qwen", "qwen_max")]:
        try:
            if ctx: await ctx.info(f"Testing {provider}...")
            response, tokens, _ = await llm_caller.call_agent(agent, test_prompt, user_id, False, ctx)
            using = clients.cerebras_available if provider == "cerebras" else clients.qwen_available
            results[provider] = {"status": "✅ SUCCESS", "tokens": tokens, "agent": agent, "using": using, 
                "fallback": None if using else "groq"}
        except Exception as e: results[provider] = {"status": "❌ FAILED", "error": str(e)}

    total, success = len(results), sum(1 for r in results.values() if "SUCCESS" in r.get("status", ""))
    return {"test_results": results, "summary": {"total_tested": total, "successful": success, "failed": total - success,
        "success_rate": f"{(success/total*100):.1f}%", "models_tested": "GPT-5, O3, Gemini 2.5 Pro, Llama 4, Qwen Max, Compound"}}

if __name__ == "__main__":
    logger.info("🚀 Multi-Agent Dev Team v3.1 - Comprehensive Chat Storage")
    logger.info(f"👥 {len(TEAM)} agents | 📁 {len(project_scanner.indexed_files)} files indexed")
    logger.info(f"🔐 Auth: Enabled | 🗄️ Qdrant: Connected")
    logger.info(f"📊 Providers: OpenAI, Groq, Google, Cerebras {'✅' if clients.cerebras_available else '⚠️'}, Qwen {'✅' if clients.qwen_available else '⚠️'}")
    logger.info("💬 Chat Storage: ACTIVE - All interactions saved with full metadata")
    logger.info("🛠️ Tools: 12 total (develop_feature, chat search/history/rating/analytics/export, + 6 others)")
    mcp.run()

