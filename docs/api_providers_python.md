
Here are canonical Python snippets from official docs for Cerebras, Groq, SambaNova, OpenRouter, and Google Gemini suitable for quick integration into coding/math agents.[^1][^2][^3][^4][^5]

### Cerebras

Canonical Python example.[^2]

```python
# pip install --upgrade cerebras_cloud_sdk
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
chat_completion = client.chat.completions.create(
    model="llama-4-scout-17b-16e-instruct",
    messages=[{"role": "user", "content": "Why is fast inference important?"}],
)
print(chat_completion)
```


### Groq

Canonical Python example.[^3]

```python
# pip install groq
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Explain the importance of fast language models"}],
)
print(chat_completion.choices[^0].message.content)
```


### SambaNova

Canonical OpenAI‑compatible Python example.[^1]

```python
# pip install openai
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["SAMBANOVA_BASE_URL"],  # e.g., https://api.sambanova.ai/v1
    api_key=os.environ["SAMBANOVA_API_KEY"],
)
completion = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "Answer in a couple sentences."},
        {"role": "user", "content": "Share a happy story with me"},
    ],
)
print(completion.choices[^0].message)
```


### OpenRouter

Canonical OpenAI‑compatible Python example.[^4]

```python
# pip install openai
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  # your site
        "X-Title": "My App",                      # your app name
    },
)
completion = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a Python function to test primality."}],
)
print(completion.choices[^0].message.content)
```


### Google (Gemini API)

Canonical Python example using the Google GenAI SDK.[^5]

```python
# pip install -U google-genai
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Compute 123 * 456 and show the steps.",
)
print(resp.text)
```

<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://docs.sambanova.ai/docs/en/features/openai-compatibility

[^2]: https://inference-docs.cerebras.ai/quickstart

[^3]: https://console.groq.com/docs/quickstart

[^4]: https://openrouter.ai/docs/quickstart

[^5]: https://ai.google.dev/gemini-api/docs/quickstart

[^6]: https://docs-legacy.sambanova.ai/sambastudio/latest/open-ai-api.html

[^7]: https://sambanova.ai/blog/introducing-the-sambanova-sdk

[^8]: https://docs.sambanova.ai/cloud/docs/get-started/quickstart

[^9]: https://haystack.deepset.ai/integrations/sambanova

[^10]: https://github.com/Cerebras/cerebras-cloud-sdk-python

[^11]: https://github.com/OE-LUCIFER/Cerebra.aiAPI

[^12]: https://cloud.sambanova.ai

[^13]: https://haystack.deepset.ai/integrations/cerebras

[^14]: https://github.com/sambanova/ai-starter-kit

[^15]: https://inference-docs.cerebras.ai

[^16]: https://www.promptfoo.dev/docs/providers/cerebras/

[^17]: https://www.f22labs.com/blogs/how-to-use-hugging-face-with-openai-compatible-apis/

[^18]: https://inference-docs.cerebras.ai/resources/integrations

[^19]: https://microsoft.github.io/autogen/0.2/docs/reference/oai/cerebras/

[^20]: https://developers.llamaindex.ai/python/framework-api-reference/llms/sambanovasystems/

[^21]: https://github.com/Cerebras/inference-examples

[^22]: https://inference-docs.cerebras.ai/resources/examples

[^23]: https://inspect.aisi.org.uk/providers.html

[^24]: https://www.youtube.com/watch?v=jScpBCBoGdU

[^25]: https://github.com/groq/groq-python

[^26]: https://console.groq.com/docs/examples

[^27]: https://replit.com/guides/groq-chatbot-quickstart

[^28]: https://www.geeksforgeeks.org/nlp/groq-api-with-llama-3/

[^29]: https://github.com/OpenRouterTeam/openrouter-examples-python

[^30]: https://github.com/google-gemini/gemini-api-quickstart

[^31]: https://icrisstudio1.pythonanywhere.com/blog/how-to-create-a-simple-ai-function-agent-using-groq-api

[^32]: https://snyk.io/articles/openrouter-in-python-use-any-llm-with-one-api-key/

[^33]: https://www.youtube.com/watch?v=qfWpPEgea2A

[^34]: https://github.com/OpenRouterTeam/openrouter-examples

[^35]: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart

[^36]: https://python.useinstructor.com/integrations/openrouter/

[^37]: https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/quickstart_colab.ipynb

[^38]: https://openrouter.ai/docs/features/tool-calling

[^39]: https://www.geeksforgeeks.org/artificial-intelligence/getting-started-with-google-gemini-with-python-api-integration-and-model-capabilities/

[^40]: https://dev.to/allanninal/building-your-first-agentic-ai-workflow-with-openrouter-api-1fo6

