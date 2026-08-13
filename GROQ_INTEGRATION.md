# Multi-Provider Support - Groq Integration

## Overview

ABZ Agent SDK now supports **multiple LLM providers**:
- ✅ **Google Gemini** (existing)
- ✅ **Groq** (new!)

The SDK automatically detects which provider to use based on the model name you specify.

## Quick Start

### Using Groq Models

```python
from abzagent import Agent
import os

# Create an agent with a Groq model
agent = Agent(
    name="MyGroqAgent",
    instructions="You are a helpful assistant.",
    model="qwen/qwen3.6-27b",  # Groq model - auto-detected
    api_key=os.getenv("GROQ_API_KEY")
)

response = agent.run("Hello!")
print(response.content)
```

### Using Gemini Models (unchanged)

```python
from abzagent import Agent
import os

# Create an agent with a Gemini model
agent = Agent(
    name="MyGeminiAgent",
    instructions="You are a helpful assistant.",
    model="gemini-2.5-flash",  # Gemini model - auto-detected
    api_key=os.getenv("GEMINI_API_KEY")
)

response = agent.run("Hello!")
print(response.content)
```

## Environment Variables

Add both API keys to your `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

The SDK will automatically use the appropriate key based on the model you select.

## Supported Models

> This list is a point-in-time snapshot. The source of truth is
> `abzagent/providers/model_types.py` (`GeminiModel`/`GroqModel`), where every entry is
> verified against the live provider API — see that file's docstring. Run
> `python scripts/update_model_catalog.py` to check for drift.

### Groq Models

Current Groq models (auto-detected):
- `llama-3.1-8b-instant` - Llama 3.1 8B (fastest)
- `llama-3.3-70b-versatile` - Llama 3.3 70B (highest quality)
- `openai/gpt-oss-120b` - GPT-OSS 120B
- `openai/gpt-oss-20b` - GPT-OSS 20B
- `qwen/qwen3.6-27b` - Qwen 3.6 27B
- `groq/compound` / `groq/compound-mini` - Agentic system models
- `allam-2-7b` - Compact, Arabic-specialized

### Gemini Models

Current Gemini models (auto-detected):
- `gemini-2.5-flash` - Default. Fast and capable.
- `gemini-flash-latest` / `gemini-flash-lite-latest` / `gemini-pro-latest` - Always-current aliases
- `gemini-3.5-flash` / `gemini-3.5-flash-lite` / `gemini-3.6-flash` / `gemini-3.1-flash-lite` - Newer generations

## How Provider Detection Works

The SDK checks for an exact match against the known-current Groq model list first, then
falls back to model name patterns for anything else:

**Groq Provider** is used for an exact match against the known Groq models above, or when
the model name contains:
- `qwen/`
- `llama`
- `mixtral`
- `deepseek`
- `gemma2` or `gemma-`

**Gemini Provider** is used for:
- Models starting with `gemini`
- Models starting with `models/gemini`
- Any other model (default fallback)

## Switching Between Providers

You can easily switch between providers by just changing the model name:

```python
from abzagent import Agent
import os

# Use Groq
groq_agent = Agent(
    name="GroqAgent",
    instructions="You are helpful.",
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Use Gemini
gemini_agent = Agent(
    name="GeminiAgent",
    instructions="You are helpful.",
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
```

## Implementation Details

### New Files Added

1. **`abzagent/providers/groq.py`** - Groq provider implementation
2. **`abzagent/providers/groq_catalog.py`** - Groq model catalog and utilities

### Modified Files

1. **`abzagent/config.py`** - Added multi-provider support
2. **`abzagent/core/agent.py`** - Dynamic provider selection
3. **`abzagent/providers/__init__.py`** - Export Groq provider
4. **`abzagent/__init__.py`** - Public API updates

### Backward Compatibility

✅ **All existing Gemini code continues to work without changes!**

Your existing code like this still works:
```python
agent = Agent(
    name="MyAgent",
    instructions="...",
    model="gemini-2.5-flash"
)
```

## Testing

Run the test examples:

```bash
# Test Groq provider
python test_groq_sdk.py

# Test Gemini provider (verify backward compatibility)
python examples/test_gemini.py

# Test switching between providers
python examples/test_switching.py
```

## Version

- **SDK Version**: 0.9.0
- **New Feature**: Multi-provider support (Gemini + Groq)

## Next Steps

To use Groq models:
1. Get a Groq API key from [console.groq.com](https://console.groq.com)
2. Add `GROQ_API_KEY` to your `.env` file
3. Use any Groq model name when creating an Agent
4. The SDK handles the rest automatically!

---

**Questions or Issues?** The SDK automatically handles provider selection, API key management, and response formatting. Just specify your model and go!
