# src/tools/api_tools.py

"""
Async Mistral API tools for chat, embed, OCR.
Uses mistralai client; retries with tenacity; base64 for images.
Config from [mistral] section.
"""

import asyncio
import base64
import logging
from typing import Any

from mistralai import AsyncMistral  # pip install mistralai
from mistralai.models.chat_completion import ChatMessage
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MistralTools:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        mistral_cfg = config.get("mistral", {})
        api_key = os.getenv(mistral_cfg.get("api_key_name", "MISTRAL_API_KEY"))
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        self.client = AsyncMistral(api_key=api_key)
        self.embedding_model = mistral_cfg.get("embedding_model", "mistral-embed")
        self.ocr_model = mistral_cfg.get("ocr_model", "mistral-ocr-latest")
        self.summary_model = mistral_cfg.get("summary_model", "mistral-large-latest")
        self.max_tokens = mistral_cfg.get("max_tokens_summary", 2000)
        self.batch_size = mistral_cfg.get("batch_size_embed", 50)
        self.retries = mistral_cfg.get("retries", 3)
        self.timeout = mistral_cfg.get("timeout_api", 120)
        self.include_base64 = mistral_cfg.get("include_image_base64", True)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat(
        self, prompt: str, model: str = None, messages: list[ChatMessage] = None
    ) -> str:
        model = model or self.summary_model
        try:
            if messages:
                response = await asyncio.wait_for(
                    self.client.chat_completion(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout,
                )
            else:
                messages = [ChatMessage(role="user", content=prompt)]
                response = await asyncio.wait_for(
                    self.client.chat_completion(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout,
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Mistral chat failed: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed(self, texts: list[str], model: str = None) -> list[list[float]]:
        model = model or self.embedding_model
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = await asyncio.wait_for(
                    self.client.embeddings(
                        model=model,
                        inputs=batch,
                    ),
                    timeout=self.timeout,
                )
                batch_emb = [emb.embedding for emb in response.data]
                embeddings.extend(batch_emb)
            except Exception as e:
                logger.error("Mistral embed failed for batch %d: %s", i, e)
                embeddings.extend([[0.0] * 1024 for _ in batch])  # Fallback zero vec
        return embeddings

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def ocr(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        prompt = "Extract text from this image/PDF page, including structure."
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            )
        ]
        try:
            response = await asyncio.wait_for(
                self.client.chat_completion(
                    model=self.ocr_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Mistral OCR failed: %s", e)
            return ""

    # Execute for MCP payload
    async def execute(self, payload: dict[str, Any]) -> Any:
        tool_type = payload.get("type", "chat")
        if tool_type == "chat":
            return await self.chat(payload.get("prompt", ""), payload.get("model"))
        elif tool_type == "embed":
            texts = payload.get("texts", [])
            return await self.embed(texts, payload.get("model"))
        elif tool_type == "ocr":
            image_path = payload.get("image_path", "")
            return await self.ocr(image_path)
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")


# Fallback if no config
def get_mistral_tool(config: dict[str, Any]) -> MistralTools:
    return MistralTools(config)
