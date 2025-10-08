
"""
multi_agent_dev_team.py

Enterprise multi-agent orchestration with comprehensive chat storage and agent memory.

Version: 4.1.0
Date: October 7, 2025
Author: Nikolay Nikolov
"""

import asyncio, fnmatch, hashlib, json, logging, os, secrets, subprocess, yaml
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import google.generativeai as genai_stable
import openai
from fastmcp import Context, FastMCP
from google import generativeai as genai
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration Loading ---

def load_config():
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    
    def replace_env_vars(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = replace_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = replace_env_vars(item)
        elif isinstance(data, str) and data.startswith("${basedir}") and data.endswith("}"):
            return os.getenv(data[2:-1])
        return data

    return replace_env_vars(config_data)

CONFIG = load_config()

# --- Logging Setup ---

logging.basicConfig(
    level=CONFIG["logging"]["level"],
    format=CONFIG["logging"]["format"],
    handlers=[logging.FileHandler(CONFIG["logging"]["file"]), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Constants ---

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
INDEXABLE_PATTERNS = ["*.go", "*.py", "*.sh", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.txt", "*.rs", "*.c", "*.h"]
EXCLUDE_DIRS = [".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", "vendor", "target", "build", "dist"]

mcp = FastMCP(name=CONFIG["project"]["name"], instructions="Enterprise team with 30+ agents, latest models, and comprehensive chat storage")

# --- Authorization ---

class Role(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    WRITER = "writer"
    VIEWER = "viewer"

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
    def __init__(self, roles_config, agent_perms_config):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}
        self.roles_config = roles_config
        self.agent_perms_config = agent_perms_config
        self._initialize_default_users()

    def _initialize_default_users(self):
        admin_key = os.getenv("ADMIN_API_KEY") or f"mcp_{secrets.token_urlsafe(32)}"
        admin_user = User("admin_1", "admin", Role.ADMIN, admin_key, datetime.now())
        self.users[admin_user.user_id] = admin_user
        self.api_keys[admin_key] = admin_user.user_id
        logger.info(f"🔑 Admin API Key: {admin_key}")

    def create_user(self, username: str, role: Role, created_by: str) -> User:
        if "manage_users" not in self.roles_config.get(self.users[created_by].role.value, []):
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

    def has_permission(self, user_id: str, permission: str) -> bool:
        user = self.users.get(user_id)
        return user and user.is_active and permission in self.roles_config.get(user.role.value, [])

    def can_call_agent(self, user_id: str, agent: str) -> bool:
        user = self.users.get(user_id)
        if not user: return False
        for perm, agents in self.agent_perms_config.items():
            if agent in agents and self.has_permission(user_id, perm):
                return True
        return False

auth_manager = AuthManager(CONFIG["auth"]["roles"], CONFIG["auth"]["agent_permissions"])

# --- API Clients ---

class APIClients:
    def __init__(self, providers_config):
        self.providers_config = providers_config
        self.openai = openai.AsyncOpenAI(api_key=providers_config["openai"]["api_key"])
        self.groq = openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=providers_config["groq"]["api_key"])
        genai.configure(api_key=providers_config["gemini"]["api_key"])
        genai_stable.configure(api_key=providers_config["google"]["api_key"])
        logger.info("✅ OpenAI, Groq, Google initialized")

        self.cerebras_clients = {}
        if providers_config["cerebras"]["personal_key"]:
            self.cerebras_clients["personal"] = {"client": openai.AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=providers_config["cerebras"]["personal_key"]), "usage_count": 0}
        if providers_config["cerebras"]["book_expert_key"]:
            self.cerebras_clients["book_expert"] = {"client": openai.AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=providers_config["cerebras"]["book_expert_key"]), "usage_count": 0}
        if self.cerebras_clients:
            logger.info(f"✅ Cerebras: {len(self.cerebras_clients)} key(s)")

        self.qwen = None
        if providers_config["dashscope"]["api_key"]:
            self.qwen = openai.AsyncOpenAI(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", api_key=providers_config["dashscope"]["api_key"])
            logger.info("✅ Qwen initialized")

    def get_cerebras_client(self):
        if not self.cerebras_clients: return None
        # ... (rest of the logic is the same)

clients = APIClients(CONFIG["providers"])

# --- Qdrant & Project Scanning ---

class QdrantManager:
    CACHE_THRESHOLD, CONTEXT_THRESHOLD = 0.95, 0.85

    def __init__(self, url: str, api_key: str, collections: List[str], context_retrieval_config: dict):
        self.url, self.api_key, self.collections = url, api_key, collections
        self.context_retrieval_config = context_retrieval_config
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._setup_collections()

    def _setup_collections(self):
        # ... (same as before)

    async def get_embedding(self, text: str):
        return await asyncio.to_thread(self.embedding_model.encode, text[:8000].tolist())

    async def store(self, collection: str, text: str, metadata: dict):
        # ... (same as before)
    
    async def search(self, collection: str, query: str, limit=3, threshold=None, query_filter=None):
        try:
            emb = await self.get_embedding(query)
            if not emb: return []
            results = await asyncio.to_thread(self.client.search, collection, query_vector=emb, limit=limit, 
                                              score_threshold=threshold or self.CONTEXT_THRESHOLD, query_filter=query_filter)
            return [{"score": r.score, "payload": r.payload, "id": r.id} for r in results]
        except: return []

    async def get_relevant_context(self, task_desc: str, agent_id: str):
        context_str = ""
        # 1. Get agent-specific memory
        agent_memories = await self.search("agent_memory", task_desc, limit=3, query_filter=models.Filter(must=[models.FieldCondition(key="agent_id", match=models.MatchValue(value=agent_id))]))
        if agent_memories:
            context_str += "\n--- Agent Memory ---\n"
            for mem in agent_memories:
                context_str += f"- {mem['payload']['text_preview']}\n"

        # 2. Get general context from configured collections
        collections_to_search = self.context_retrieval_config["collections"]
        limit = self.context_retrieval_config["limit_per_collection"]
        
        for collection in collections_to_search:
            results = await self.search(collection, task_desc, limit=limit)
            if results:
                context_str += f"\n--- {collection.replace('_', ' ').title()} ---\n"
                for res in results:
                    context_str += f"- {res['payload']['text_preview']}\n"
        
        return context_str

class ProjectScanner:
    # ... (same as before)

qdrant = QdrantManager(CONFIG["qdrant"]["url"], CONFIG["qdrant"]["api_key"], CONFIG["qdrant"]["collections"], CONFIG["context_retrieval"])
project_scanner = ProjectScanner(PROJECT_ROOT, qdrant)

# --- LLM Caller ---

class LLMCaller:
    def __init__(self, clients: APIClients, qdrant: QdrantManager, auth: AuthManager, agents_config: dict):
        self.clients, self.qdrant, self.auth, self.agents_config = clients, qdrant, auth, agents_config

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def call_agent(self, agent: str, prompt: str, user_id: str, use_context=True, ctx=None, task_id=None, stage_name=None, parent_chat_id=None):
        if not self.auth.can_call_agent(user_id, agent):
            raise PermissionError(f"Access denied: cannot call {agent}")
        if agent not in self.agents_config:
            raise ValueError(f"Unknown agent: {agent}")

        agent_config = self.agents_config[agent]
        original_prompt = prompt

        if use_context:
            context = await self.qdrant.get_relevant_context(prompt[:500], agent)
            if context:
                prompt += "\n\n--- Relevant Context ---" + context

        # ... (caching logic can be enhanced here)

        response_text, tokens = await self._call_provider(agent_config["provider"], agent_config["model"], prompt, 
            agent_config["temp"], agent_config["tokens"])
        
        # Store agent memory
        await self.qdrant.store("agent_memory", f"Prompt: {original_prompt}\nResponse: {response_text}", {"agent_id": agent})

        return response_text, tokens, "chat_id_placeholder"

    async def _call_provider(self, provider: str, model: str, prompt: str, temp: float, max_tokens: int):
        # ... (same as before)

llm_caller = LLMCaller(clients, qdrant, auth_manager, CONFIG["agents"])

# --- Tools ---

@mcp.tool
async def develop_feature(description: str, requirements: List[str], api_key: str, ctx: Context = None):
    """🚀 Complete a dynamic development workflow using the best agents for the job."""
    user = auth_manager.authenticate(api_key)
    if not user: raise PermissionError("Invalid API key")
    if not auth_manager.has_permission(user.user_id, "create_tasks"): raise PermissionError("Insufficient permissions")

    user_id, task_id = user.user_id, hashlib.md5(f"{description}{datetime.now()}".encode()).hexdigest()[:12]
    if ctx: await ctx.info(f"🚀 Task: {task_id}")

    task = {"task_id": task_id, "description": description, "requirements": requirements, "user_id": user_id, 
        "stages": {}, "created_at": datetime.now().isoformat()}
    
    await qdrant.store("tasks", f"{description} {' '.join(requirements)}", {"task_id": task_id, "user_id": user_id})

    stage_results = {"description": description, "requirements": ', '.join(requirements)}
    total_tokens = 0

    for stage_config in CONFIG["workflows"]["develop_feature"]:
        stage, agent, prompt_template = stage_config["stage"], stage_config["agent"], stage_config["prompt"]
        
        # Check for cached result for this stage
        cache_key = f"{task_id}_{stage}"
        cached_result = await qdrant.search("llm_cache", cache_key, limit=1)
        if cached_result:
            if ctx: await ctx.info(f"🔄 Using cached result for {stage}...")
            result = cached_result[0]['payload']['response']
            tokens = 0
        else:
            if ctx: await ctx.info(f"🔄 Processing {stage}...")
            prompt = prompt_template.format(**stage_results)
            result, tokens, _ = await llm_caller.call_agent(agent, prompt, user_id, True, ctx, task_id, stage)
            # Cache the result
            await qdrant.store("llm_cache", result, {"key": cache_key, "response": result})

        task["stages"][stage] = {"result": result, "tokens": tokens, "agent": agent}
        stage_results[stage] = result
        total_tokens += tokens

        if stage_config["store"]:
            await qdrant.store(stage_config["store"], result[:2000], {"task_id": task_id})

    task["tokens_used"] = total_tokens
    task["status"] = "complete"
    if ctx: await ctx.info(f"✅ Complete! {total_tokens} tokens")

    return task

# ... (other tools would be refactored similarly) ...

async def main():
    logger.info(f"🚀 {CONFIG['project']['name']} v{CONFIG['project']['version']} - Enhanced Qdrant Integration")
    await project_scanner.scan_and_index()
    logger.info(f"📁 {len(project_scanner.indexed_files)} files indexed")
    logger.info(f"🔐 Auth: Enabled | 🗄️ Qdrant: Connected")
    mcp.run()

if __name__ == "__main__":
    asyncio.run(main())
