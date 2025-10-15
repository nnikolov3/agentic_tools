# Provider Schemas
### Cerebras

- Format: OpenAI‑compatible Chat Completions.

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "llama-3.3-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
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

- Streaming chunk:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1712345679,
  "model": "llama-3.3-70b-instruct",
  "choices": [
    { "index": 0, "delta": { "role": "assistant", "content": "par" }, "finish_reason": null }
  ]
}
```

- Tool calling fields:
    - `choices[].message.tool_calls[]` with items `{ "id": "...","type": "function","function": { "name": "fn", "arguments": "{...json...}" } }`.
- Common headers:
    - `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests`.
    - `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens`.


### Groq

- Format: OpenAI‑compatible Chat Completions.

```json
{
  "id": "chatcmpl-xyz789",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "llama-3.3-70b-versatile",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
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

- Streaming chunk:

```json
{
  "id": "chatcmpl-xyz789",
  "object": "chat.completion.chunk",
  "created": 1712345680,
  "model": "llama-3.3-70b-versatile",
  "choices": [
    { "index": 0, "delta": { "content": "..." }, "finish_reason": null }
  ]
}
```

- Tool calling fields:
    - Same OpenAI format via `choices[].message.tool_calls[]`.
- Common headers:
    - `x-ratelimit-limit-requests` (per day), `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests`.
    - `x-ratelimit-limit-tokens` (TPM), `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens`.
    - `retry-after` on 429.


### SambaNova

- Format: OpenAI‑compatible Chat Completions (OpenAI SDK works by setting `base_url` and API key).

```json
{
  "id": "chatcmpl-snv001",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "Meta-Llama-3.3-70B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
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

- Streaming chunk:

```json
{
  "id": "chatcmpl-snv001",
  "object": "chat.completion.chunk",
  "created": 1712345680,
  "model": "Meta-Llama-3.3-70B-Instruct",
  "choices": [
    { "index": 0, "delta": { "content": "..." }, "finish_reason": null }
  ]
}
```

- Tool calling fields:
    - Same OpenAI format via `choices[].message.tool_calls[]`.
- Common headers:
    - `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-requests`.
    - `x-ratelimit-limit-requests-day`, `x-ratelimit-remaining-requests-day`, `x-ratelimit-reset-requests-day`.


### Google Gemini

- Format: Generate Content (non‑OpenAI schema).

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [ { "text": "..." } ]
      },
      "finishReason": "STOP",
      "safetyRatings": [ { "category": "HARM_CATEGORY_X", "probability": "LOW" } ]
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 123,
    "candidatesTokenCount": 456,
    "totalTokenCount": 579
  },
  "modelVersion": "gemini-2.5-flash",
  "responseId": "resp-abc123"
}
```

- Streaming delta (high level):
    - Multiple incremental messages containing partial `candidates[0].content.parts[].text` updates and a final message with `finishReason`.
- Tool/function calling fields:
    - A tool request appears as a `content.parts[]` item with a `functionCall`:

```json
{ "functionCall": { "name": "fn", "args": { "param": "value" } } }
```

    - Tool result is returned by sending a new `content` from role `tool` with a `functionResponse`, and the model’s next response incorporates those results.
- Common metadata:
    - Token counts are in `usageMetadata`.
    - No OpenAI‑style `usage.prompt_tokens`; names follow camelCase.

```
{
  "providers": {
    "cerebras": {
      "schemas": {
        "chat_completion": {
          "type": "object",
          "required": ["id", "object", "created", "model", "choices"],
          "properties": {
            "id": { "type": "string" },
            "object": { "const": "chat.completion" },
            "created": { "type": "integer" },
            "model": { "type": "string" },
            "choices": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["index", "message"],
                "properties": {
                  "index": { "type": "integer" },
                  "message": {
                    "type": "object",
                    "required": ["role"],
                    "properties": {
                      "role": { "enum": ["system", "user", "assistant", "tool"] },
                      "content": { "type": ["string", "null"] },
                      "tool_calls": {
                        "type": "array",
                        "items": {
                          "type": "object",
                          "required": ["id", "type", "function"],
                          "properties": {
                            "id": { "type": "string" },
                            "type": { "const": "function" },
                            "function": {
                              "type": "object",
                              "required": ["name", "arguments"],
                              "properties": {
                                "name": { "type": "string" },
                                "arguments": { "type": "string" }
                              },
                              "additionalProperties": true
                            }
                          },
                          "additionalProperties": true
                        }
                      }
                    },
                    "additionalProperties": true
                  },
                  "finish_reason": { "type": ["string", "null"] },
                  "logprobs": { "type": ["object", "null"] }
                },
                "additionalProperties": true
              }
            },
            "usage": {
              "type": "object",
              "properties": {
                "prompt_tokens": { "type": "integer" },
                "completion_tokens": { "type": "integer" },
                "total_tokens": { "type": "integer" }
              },
              "additionalProperties": true
            },
            "system_fingerprint": { "type": ["string", "null"] }
          },
          "additionalProperties": true
        },
        "chat_chunk": {
          "type": "object",
          "required": ["object", "choices"],
          "properties": {
            "id": { "type": "string" },
            "object": { "const": "chat.completion.chunk" },
            "created": { "type": "integer" },
            "model": { "type": "string" },
            "choices": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "index": { "type": "integer" },
                  "delta": {
                    "type": "object",
                    "properties": {
                      "role": { "enum": ["assistant"] },
                      "content": { "type": ["string", "null"] },
                      "tool_calls": { "type": "array" }
                    },
                    "additionalProperties": true
                  },
                  "finish_reason": { "type": ["string", "null"] }
                },
                "additionalProperties": true
              }
            }
          },
          "additionalProperties": true
        },
        "rate_limit_headers": {
          "type": "object",
          "properties": {
            "x-ratelimit-limit-requests": { "type": "string" },
            "x-ratelimit-remaining-requests": { "type": "string" },
            "x-ratelimit-reset-requests": { "type": "string" },
            "x-ratelimit-limit-tokens": { "type": "string" },
            "x-ratelimit-remaining-tokens": { "type": "string" },
            "x-ratelimit-reset-tokens": { "type": "string" }
          },
          "additionalProperties": true
        }
      },
      "examples": {
        "non_stream": {
          "id": "chatcmpl-abc123",
          "object": "chat.completion",
          "created": 1712345678,
          "model": "llama-3.3-70b-instruct",
          "choices": [
            {
              "index": 0,
              "message": { "role": "assistant", "content": "Hello!" },
              "finish_reason": "stop"
            }
          ],
          "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        }
      }
    },
    "groq": {
      "schemas": {
        "chat_completion": { "$ref": "#/providers/cerebras/schemas/chat_completion" },
        "chat_chunk": { "$ref": "#/providers/cerebras/schemas/chat_chunk" },
        "rate_limit_headers": {
          "type": "object",
          "properties": {
            "retry-after": { "type": "string" },
            "x-ratelimit-limit-requests": { "type": "string" },
            "x-ratelimit-remaining-requests": { "type": "string" },
            "x-ratelimit-reset-requests": { "type": "string" },
            "x-ratelimit-limit-tokens": { "type": "string" },
            "x-ratelimit-remaining-tokens": { "type": "string" },
            "x-ratelimit-reset-tokens": { "type": "string" }
          },
          "additionalProperties": true }
      },
      "examples": {
        "non_stream": {
          "id": "chatcmpl-xyz789",
          "object": "chat.completion",
          "created": 1712345678,
          "model": "llama-3.3-70b-versatile",
          "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hi!" }, "finish_reason": "stop" }],
          "usage": { "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19 }
        }
      }
    },
    "sambanova": {
      "schemas": {
        "chat_completion": { "$ref": "#/providers/cerebras/schemas/chat_completion" },
        "chat_chunk": { "$ref": "#/providers/cerebras/schemas/chat_chunk" },
        "rate_limit_headers": {
          "type": "object",
          "properties": {
            "x-ratelimit-limit-requests": { "type": "string" },
            "x-ratelimit-remaining-requests": { "type": "string" },
            "x-ratelimit-reset-requests": { "type": "string" },
            "x-ratelimit-limit-requests-day": { "type": "string" },
            "x-ratelimit-remaining-requests-day": { "type": "string" },
            "x-ratelimit-reset-requests-day": { "type": "string" }
          },
          "additionalProperties": true
        }
      },
      "examples": {
        "non_stream": {
          "id": "chatcmpl-snv001",
          "object": "chat.completion",
          "created": 1712345678,
          "model": "Meta-Llama-3.3-70B-Instruct",
          "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Howdy!" }, "finish_reason": "stop" }],
          "usage": { "prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15 }
        }
      }
    },
    "gemini": {
      "schemas": {
        "generate_content": {
          "type": "object",
          "required": ["candidates"],
          "properties": {
            "candidates": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "content": {
                    "type": "object",
                    "properties": {
                      "role": { "type": "string" },
                      "parts": {
                        "type": "array",
                        "items": {
                          "oneOf": [
                            { "type": "object", "properties": { "text": { "type": "string" } }, "additionalProperties": true },
                            {
                              "type": "object",
                              "properties": {
                                "functionCall": {
                                  "type": "object",
                                  "required": ["name", "args"],
                                  "properties": {
                                    "name": { "type": "string" },
                                    "args": { "type": "object", "additionalProperties": true }
                                  },
                                  "additionalProperties": true
                                }
                              },
                              "additionalProperties": true
                            }
                          ]
                        }
                      }
                    },
                    "additionalProperties": true
                  },
                  "finishReason": { "type": ["string", "null"] },
                  "safetyRatings": { "type": "array", "items": { "type": "object" } }
                },
                "additionalProperties": true
              }
            },
            "usageMetadata": {
              "type": "object",
              "properties": {
                "promptTokenCount": { "type": "integer" },
                "candidatesTokenCount": { "type": "integer" },
                "totalTokenCount": { "type": "integer" }
              },
              "additionalProperties": true
            },
            "modelVersion": { "type": ["string", "null"] },
            "responseId": { "type": ["string", "null"] }
          },
          "additionalProperties": true
        },
        "stream_chunk": {
          "type": "object",
          "properties": {
            "candidates": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "content": {
                    "type": "object",
                    "properties": {
                      "role": { "type": "string" },
                      "parts": { "type": "array", "items": { "type": "object" } }
                    },
                    "additionalProperties": true
                  }
                },
                "additionalProperties": true
              }
            },
            "object": { "type": ["string", "null"] }
          },
          "additionalProperties": true
        },
        "tool_use": {
          "type": "object",
          "properties": {
            "functionCall": {
              "type": "object",
              "required": ["name", "args"],
              "properties": {
                "name": { "type": "string" },
                "args": { "type": "object", "additionalProperties": true }
              },
              "additionalProperties": true
            },
            "functionResponse": {
              "type": "object",
              "required": ["name", "response"],
              "properties": {
                "name": { "type": "string" },
                "response": { "type": "object", "additionalProperties": true }
              },
              "additionalProperties": true
            }
          },
          "additionalProperties": true
        }
      },
      "examples": {
        "non_stream": {
          "candidates": [
            {
              "content": { "role": "model", "parts": [ { "text": "Hello from Gemini." } ] },
              "finishReason": "STOP"
            }
          ],
          "usageMetadata": { "promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 17 },
          "modelVersion": "gemini-2.5-flash",
          "responseId": "resp-abc123"
        }
      }
    }
  }
}
```
