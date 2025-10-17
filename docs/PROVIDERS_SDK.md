
# Complete Provider SDK Documentation with Response Schemas

## 1. Cerebras Cloud SDK

### Installation \& Setup

```
pip install cerebras_cloud_sdk
```


### Client Initialization

```python
from cerebras.cloud.sdk import Cerebras
client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
```


### Response Schema

#### Non-Streaming Response

OpenAI-compatible Chat Completion format:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "llama-3.3-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "system_fingerprint": "fp_abcdef"
}
```


#### Streaming Response

Streaming chunks with `stream=True`:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1712345679,
  "model": "llama-3.3-70b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```


#### Tool Calling Response

Function/tool calls in `choices[].message.tool_calls[]`:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Boston, MA\"}"
            }
          }
        ]
      }
    }
  ]
}
```


### Python SDK Response Type

```python
from cerebras.cloud.sdk import Cerebras

response: Cerebras.Chat.ChatCompletion = client.chat.completions.create(...)
# Access: response.choices[^0].message.content
# Usage: response.usage.total_tokens
```


### Error Handling

Exception classes:

- `cerebras.cloud.sdk.APIError` (base)
- `cerebras.cloud.sdk.RateLimitError` (HTTP 429)
- `cerebras.cloud.sdk.APIConnectionError`
- `cerebras.cloud.sdk.AuthenticationError` (HTTP 401)
- `cerebras.cloud.sdk.BadRequestError` (HTTP 400)
- `cerebras.cloud.sdk.PermissionDeniedError` (HTTP 403)
- `cerebras.cloud.sdk.NotFoundError` (HTTP 404)
- `cerebras.cloud.sdk.UnprocessableEntityError` (HTTP 422)
- `cerebras.cloud.sdk.InternalServerError` (HTTP >=500)


### Rate Limit Headers

Accessible via `.with_raw_response.create()`:

```python
response = client.chat.completions.with_raw_response.create(...)
headers = response.headers
completion = response.parse()
```

Headers available:

- `x-ratelimit-limit-requests-day`: Maximum requests per day
- `x-ratelimit-remaining-requests-day`: Requests remaining today
- `x-ratelimit-reset-requests-day`: Seconds until daily reset
- `x-ratelimit-limit-tokens-minute`: Maximum tokens per minute
- `x-ratelimit-remaining-tokens-minute`: Tokens remaining this minute
- `x-ratelimit-reset-tokens-minute`: Seconds until minute reset


### Retry Behavior

Automatically retries 429, 408, 409, and >=500 errors twice by default with exponential backoff. Configurable via `max_retries` parameter.

***

## 2. Groq SDK

### Installation \& Setup

```
pip install groq
```


### Client Initialization

```python
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])
```


### Response Schema

#### Non-Streaming Response

OpenAI-compatible format:

```json
{
  "id": "chatcmpl-xyz789",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "llama-3.3-70b-versatile",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm happy to help!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 88,
    "completion_tokens": 212,
    "total_tokens": 300
  },
  "system_fingerprint": "fp_qwerty"
}
```


#### Streaming Response

Streaming chunks with `stream=True`:

```json
{
  "id": "chatcmpl-xyz789",
  "object": "chat.completion.chunk",
  "created": 1712345680,
  "model": "llama-3.3-70b-versatile",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "I'm"
      },
      "finish_reason": null
    }
  ]
}
```


#### Tool Calling Response

Same OpenAI format via `choices[].message.tool_calls[]`.

### Python SDK Response Type

```python
from groq import Groq

response: Groq.Chat.ChatCompletion = client.chat.completions.create(...)
# Access: response.choices[^0].message.content
# Usage: response.usage.total_tokens
```


### Error Handling

Exception classes:

- `groq.APIError` (base)
- `groq.RateLimitError` (HTTP 429)
- `groq.APIConnectionError`
- `groq.AuthenticationError` (HTTP 401)
- `groq.BadRequestError` (HTTP 400)
- `groq.PermissionDeniedError` (HTTP 403)
- `groq.NotFoundError` (HTTP 404)
- `groq.UnprocessableEntityError` (HTTP 422)
- `groq.InternalServerError` (HTTP >=500)


### Rate Limit Headers

Accessible via `.with_raw_response.create()`:

```python
response = client.chat.completions.with_raw_response.create(...)
headers = response.headers
completion = response.parse()
```

Headers available:

- `x-ratelimit-limit-requests`: Total requests per minute
- `x-ratelimit-remaining-requests`: Requests remaining this minute
- `x-ratelimit-reset-requests`: Epoch time when RPM resets
- `x-ratelimit-limit-tokens`: Total tokens per minute
- `x-ratelimit-remaining-tokens`: Tokens remaining this minute
- `x-ratelimit-reset-tokens`: Epoch time when TPM resets
- `retry-after`: Seconds to wait on 429 error


### Retry Behavior

Automatically retries 429, 408, 409, and >=500 errors twice by default. Configurable via `max_retries` parameter.

***

## 3. SambaNova SDK

### Installation \& Setup

```
pip install sambanova
```


### Client Initialization

```python
from sambanova import SambaNova
client = SambaNova(
    api_key=os.environ.get("SAMBANOVA_API_KEY")
    # base_url defaults to https://api.sambanova.ai/v1
)
```

**Note**: The SambaNova SDK does NOT require explicit `base_url` parameter - it defaults to the correct endpoint.

### Response Schema

#### Non-Streaming Response

OpenAI-compatible format:

```json
{
  "id": "chatcmpl-snv001",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "Meta-Llama-3.3-70B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I can help with that!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 64,
    "completion_tokens": 128,
    "total_tokens": 192
  }
}
```


#### Streaming Response

Streaming chunks with `stream=True`:

```json
{
  "id": "chatcmpl-snv001",
  "object": "chat.completion.chunk",
  "created": 1712345680,
  "model": "Meta-Llama-3.3-70B-Instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "I can"
      },
      "finish_reason": null
    }
  ]
}
```


#### Tool Calling Response

Same OpenAI format via `choices[].message.tool_calls[]`.

### Python SDK Response Type

```python
from sambanova import SambaNova

response: SambaNova.Chat.ChatCompletion = client.chat.completions.create(...)
# Access: response.choices[^0].message.content
# Usage: response.usage.total_tokens
```


### Error Handling

Exception classes:

- `sambanova.APIError` (base)
- `sambanova.RateLimitError` (HTTP 429)
- `sambanova.APIConnectionError`
- `sambanova.APITimeoutError`
- `sambanova.AuthenticationError` (HTTP 401)
- `sambanova.BadRequestError` (HTTP 400)
- `sambanova.PermissionDeniedError` (HTTP 403)
- `sambanova.NotFoundError` (HTTP 404)
- `sambanova.UnprocessableEntityError` (HTTP 422)
- `sambanova.InternalServerError` (HTTP >=500)


### Rate Limit Headers

Accessible via `.with_raw_response.create()`:

```python
response = client.chat.completions.with_raw_response.create(...)
headers = response.headers
completion = response.parse()
```

Headers available:

- `x-ratelimit-limit-requests`: Maximum requests per minute
- `x-ratelimit-remaining-requests`: Requests remaining this minute
- `x-ratelimit-reset-requests`: Epoch time when RPM resets
- `x-ratelimit-limit-requests-day`: Maximum requests per day
- `x-ratelimit-remaining-requests-day`: Requests remaining today
- `x-ratelimit-reset-requests-day`: Epoch time when RPD resets


### Additional Features

- Async client: `AsyncSambaNova`
- Streaming: `stream=True` parameter
- HTTP backend: httpx by default, aiohttp via `sambanova[aiohttp]`
- Audio transcription: `client.audio.transcriptions.create()`
- Embeddings: `client.embeddings.create()`
- Model listing: `client.models.list()`


### Retry Behavior

Automatically retries 429, 408, 409, and >=500 errors twice by default with exponential backoff. Configurable via `max_retries` parameter.

***

## 4. Google Gen AI SDK

### Installation \& Setup

```
pip install google-genai
```


### Client Initialization

```python
from google import genai

# Option 1: Gemini Developer API
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Option 2: Vertex AI (Enterprise)
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1'
)
```


### Response Schema

#### Non-Streaming Response

**Different from OpenAI format**:

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {
            "text": "Hello! How can I assist you today?"
          }
        ]
      },
      "finishReason": "STOP",
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "probability": "NEGLIGIBLE"
        }
      ]
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 12,
    "candidatesTokenCount": 8,
    "totalTokenCount": 20
  },
  "modelVersion": "gemini-2.5-flash",
  "responseId": "resp-abc123"
}
```


#### Streaming Response

Incremental chunks with partial text updates:

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {
            "text": "Hello!"
          }
        ]
      }
    }
  ]
}
```

Final chunk includes `finishReason`.

#### Function Calling Response

Function call in `content.parts[]` with `functionCall` object:

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {
            "functionCall": {
              "name": "get_weather",
              "args": {
                "location": "Boston, MA"
              }
            }
          }
        ]
      }
    }
  ]
}
```

Function response sent back with `functionResponse`:

```json
{
  "role": "function",
  "parts": [
    {
      "functionResponse": {
        "name": "get_weather",
        "response": {
          "temperature": 72,
          "conditions": "sunny"
        }
      }
    }
  ]
}
```


### Python SDK Response Type

```python
from google import genai

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Hello'
)

# Access text: response.text
# Access parsed data: response.parsed (for structured outputs)
# Access candidates: response.candidates[^0].content.parts[^0].text
# Usage: response.usage_metadata.total_token_count
```


### Unique Features

- **Automatic function calling**: Python functions passed directly are called automatically
- **Structured outputs**: Native Pydantic model support via `response_schema`
- **Safety ratings**: Each response includes safety category ratings
- **No raw headers access**: Unlike other SDKs, no `.with_raw_response` pattern
- **Dual-mode operation**: Single SDK for both Developer API and Vertex AI


### Error Handling

Google SDK uses different exception patterns than OpenAI-compatible providers. Follows Google Cloud error standards.

### Rate Limit Information

Rate limits **not exposed via response headers** like other providers. Managed through:

- Google Cloud Console (Vertex AI)
- API key tier settings (Developer API)
- Error responses when limits are exceeded

Free tier limits:

- Gemini 2.0 Flash: 15 RPM, 1M TPM, 200 RPD
- Gemini 2.5 Flash: 10 RPM, 250K TPM, 250 RPD
- Gemini 2.5 Pro: 5 RPM, 125K TPM, 100 RPD

***

## Summary: Response Schema Comparison

| Feature              | Cerebras/Groq/SambaNova                                      | Google Gemini                                                          |
|:---------------------|:-------------------------------------------------------------|:-----------------------------------------------------------------------|
| **Schema format**    | OpenAI-compatible                                            | Google native format                                                   |
| **Message access**   | `choices[^0].message.content`                                | `candidates[^0].content.parts[^0].text` or `.text`                     |
| **Role names**       | `assistant`                                                  | `model`                                                                |
| **Usage tokens**     | `usage.prompt_tokens`, `usage.completion_tokens`             | `usageMetadata.promptTokenCount`, `usageMetadata.candidatesTokenCount` |
| **Tool calling**     | `tool_calls[]` with `function.name` and `function.arguments` | `functionCall` in `parts[]` with `name` and `args`                     |
| **Streaming chunks** | `delta.content`                                              | `parts[].text`                                                         |
| **Response type**    | `ChatCompletion` (Pydantic model)                            | `GenerateContentResponse`                                              |
| **Headers access**   | `.with_raw_response.create()`                                | Not available                                                          |

## SDK Commonalities

All four SDKs share these patterns:

1. **OpenAI-compatible interface**: `client.chat.completions.create()` (except Google uses `client.models.generate_content()`)
2. **Sync/async clients**: All provide both synchronous and asynchronous clients
3. **Automatic retries**: Built-in retry logic for 429 and 5xx errors (2 retries by default)
4. **Type safety**: TypedDict parameters and Pydantic response models
5. **Streaming support**: Native SSE streaming with `stream=True`
6. **HTTP backend**: httpx by default, aiohttp available
7. **Context managers**: Support `with` statement for resource management

## Key Normalization Requirements

When building a unified API wrapper, normalize:

1. **Message extraction**: Extract `choices[^0].message.content` (OpenAI) vs `candidates[^0].content.parts[^0].text` (Google)
2. **Token counting**: Map `usage.*` (OpenAI) to `usageMetadata.*` (Google)
3. **Role mapping**: Convert `model` → `assistant` for consistency
4. **Tool call format**: Transform Google's `functionCall` to OpenAI's `tool_calls` structure
5. **Error handling**: Catch provider-specific exceptions and normalize to common error types

All three OpenAI-compatible SDKs (Cerebras, Groq, SambaNova) use identical response schemas, simplifying normalization.

***

## References

### Provider Documentation

1. Cerebras Cloud SDK Python: <https://github.com/Cerebras/cerebras-cloud-sdk-python>
2. Cerebras Inference Docs: <https://inference-docs.cerebras.ai>
3. Cerebras Rate Limits: <https://inference-docs.cerebras.ai/support/rate-limits>
4. Groq Python SDK: <https://github.com/groq/groq-python>
5. Groq Developer Documentation: <https://console.groq.com/docs>
6. Groq Rate Limits: <https://console.groq.com/docs/rate-limits>
7. SambaNova SDK Announcement: <https://sambanova.ai/blog/introducing-the-sambanova-sdk>
8. SambaNova PyPI: <https://pypi.org/project/sambanova/>
9. SambaNova Documentation: <https://docs.sambanova.ai>
10. SambaNova Rate Limits: <https://docs.sambanova.ai/docs/en/models/rate-limits>
11. Google Gen AI SDK: <https://github.com/googleapis/python-genai>
12. Google Gen AI Documentation: <https://googleapis.github.io/python-genai/>
13. Google Gemini API Quickstart: <https://ai.google.dev/gemini-api/docs/quickstart>
14. Google Gemini Rate Limits: <https://ai.google.dev/gemini-api/docs/rate-limits>

### Attached Files

- api_providers_python.md: Canonical Python examples from provider documentation
- provider_schemas.md: Detailed response schemas and JSON schema definitions
- llm_limits.md: Provider rate limits and model offerings table
- api.py: Current implementation requiring fixes
<span style="display:none">[^1][^10][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: <https://blog.dreamfactory.com/8-api-documentation-examples>

[^2]: <https://nordicapis.com/5-examples-of-excellent-api-documentation/>

[^3]: <https://www.reddit.com/r/technicalwriting/comments/1e61mkd/best_api_docs_youve_seen/>

[^4]: <https://treblle.com/blog/best-api-documentation-examples>

[^5]: <https://stoplight.io/api-documentation-guide>

[^6]: <https://www.postman.com/api-platform/api-documentation/>

[^7]: <https://www.archbee.com/blog/api-documentation-examples>

[^8]: <https://stackoverflow.com/questions/1966243/restful-api-documentation>

[^9]: <https://www.mintlify.com/blog/our-recommendations-for-creating-api-documentation-with-examples>

[^10]: <https://idratherbewriting.com/learnapidoc/docendpoints.html>


