# LLM Configuration

The MVP now supports optional real LLM calls for:

- `DisclosureParserAgent`
- `KeywordExpansionAgent`

If no API variables are set, the project falls back to local rule-based logic.

## Where To Add The API Key

Set environment variables before running Streamlit.

PowerShell example:

```powershell
cd C:\Users\maoye\Documents\RAG\ip-agent-lab
conda activate ip-agent-lab

$env:IP_AGENT_LLM_PROVIDER="openai_compatible"
$env:IP_AGENT_LLM_BASE_URL="https://api.openai.com/v1"
$env:IP_AGENT_LLM_MODEL="gpt-4o-mini"
$env:IP_AGENT_LLM_API_KEY="your_api_key_here"

streamlit run frontend/app.py
```

Do not put a real API key into Git.

## OpenAI-Compatible Providers

The current client calls:

```text
{IP_AGENT_LLM_BASE_URL}/chat/completions
```

with a bearer token:

```text
Authorization: Bearer {IP_AGENT_LLM_API_KEY}
```

So it works with providers that expose an OpenAI-compatible chat completions API.

Examples:

```powershell
# OpenAI
$env:IP_AGENT_LLM_BASE_URL="https://api.openai.com/v1"
$env:IP_AGENT_LLM_MODEL="gpt-4o-mini"

# DeepSeek
$env:IP_AGENT_LLM_BASE_URL="https://api.deepseek.com/v1"
$env:IP_AGENT_LLM_MODEL="deepseek-chat"

# Local OpenAI-compatible server
$env:IP_AGENT_LLM_BASE_URL="http://localhost:8000/v1"
$env:IP_AGENT_LLM_MODEL="your-local-model"
```

## What The LLM Does Now

`DisclosureParserAgent` asks the model to output strict JSON:

- title
- technical field
- problem
- solution
- innovation points
- effects
- applications
- key terms

`KeywordExpansionAgent` asks the model to output strict JSON:

- core terms
- synonyms
- English terms
- query groups
- classification hints

## What Still Uses Fallback Logic

These parts are not LLM-backed yet:

- patent retrieval
- prior-art comparison
- risk scoring
- report generation

They still use local deterministic logic so the demo remains runnable without external services.
