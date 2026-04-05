# RAG Chatbot for Obsidian

Vault is a local, privacy-first AI chatbot that reads your Obsidian vault, indexes it into a searchable knowledge base, and lets you have multi-turn conversations with your own notes - all running on your machine with no data leaving it.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
   - [1. Clone and create environment](#1-clone-and-create-environment)
   - [2. Install Python dependencies](#2-install-python-dependencies)
   - [3. Install and start Ollama](#3-install-and-start-ollama)
   - [4. Configure your vault](#4-configure-your-vault)
   - [5. Build the knowledge base](#5-build-the-knowledge-base)
5. [Running the app](#running-the-app)
6. [Using the chat interface](#using-the-chat-interface)
7. [Configuration reference](#configuration-reference)
8. [Using OpenAI instead of Ollama](#using-openai-instead-of-ollama)
9. [Rebuilding the index](#rebuilding-the-index)
10. [API reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [Privacy](#privacy)
13. [Demo](#demo)

---

## How it works

https://github.com/user-attachments/assets/030de6a5-3c6b-4ce7-b748-e818aa9ab00d


```
Your Obsidian notes (.md files)
        │
        ▼
[ obsidian_connector.py ]
  • Reads all markdown files recursively
  • Strips wikilinks, tags, frontmatter, code blocks
  • Splits each note by heading into chunks
  • Embeds every chunk with sentence-transformers
  • Saves to knowledge_base.pkl
        │
        ▼
[ RAG (rag.py) ]
  • Embeds the user's query
  • Finds the most similar chunks via cosine similarity
  • If nothing found → rephrases query and retries
  • Builds a context string within token limits
        │
        ▼
[ Chatbot (chatbot.py) ]
  • Maintains full conversation history
  • Injects fresh retrieved context into every turn
  • Sends system prompt + history + context to the LLM
        │
        ▼
[ LLMClient (llm_client.py) ]
  • Talks to Ollama (local) or OpenAI (cloud)
  • Normalises both into a single response interface
        │
        ▼
[ FastAPI (api.py) ]
  • Exposes /chat, /reset, /settings, /settings/rebuild
  • Serves the web UI at /
        │
        ▼
[ Browser UI (index.html) ]
  • Dark, startup-style chat interface
  • Thinking animation while model responds
  • Settings panel to change vault path at runtime
```

---

## Project structure

```
rag-chat/
├── .env                       # Configuration (vault path, etc.)
├── chatbot/
│   └── chatbot.py             # Conversation history manager
├── connectors/
│   └── obsidian_connector.py  # Vault reader + embedder
├── rags/
│   ├── rag.py                 # Core retrieval + generation logic
│   ├── llm_client.py          # Ollama / OpenAI abstraction
│   └── rag_config.py          # Prompts, thresholds, token limits
├── templates/
│   └── index.html             # Web UI
├── app.py                     # FastAPI server
├── requirements.txt
└── generated_sources/
    └── knowledge_base.pkl     # Auto-generated, do not edit
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- An Obsidian vault with markdown notes (or any folder of `.md` files)
- ~2 GB disk space for the default model

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/podgorskip/rag-chatbot-obsidian-vault
cd rag-chat

conda create -n rag-chat python=3.12
conda activate rag-chat
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
sentence-transformers==3.0.1
scikit-learn==1.5.1
numpy==2.0.1
pandas==2.2.2
torch==2.4.0
requests==2.32.3
openai==1.40.6
fastapi==0.115.6
uvicorn==0.30.6
pydantic==2.8.2
```

### 3. Install and start Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Pull a model (choose one)
ollama pull llama3.2        # recommended — ~2 GB, good quality
ollama pull llama3.2:1b     # lighter — ~1 GB, faster

# Start the Ollama server (runs in background)
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434
# → Ollama is running
```

### 4. Configure your vault

Create a `.env` file in the project root:

```bash
VAULT_PATH=/Users/yourname/Documents/MyVault
EXCLUDE_FOLDERS=templates,archive,.trash
KNOWLEDGE_BASE=generated_sources/knowledge_base.pkl
```

| Variable | Description |
|---|---|
| `VAULT_PATH` | Absolute path to your Obsidian vault root |
| `EXCLUDE_FOLDERS` | Comma-separated folder names to skip |
| `KNOWLEDGE_BASE` | Where to save the embedded index |

### 5. Build the knowledge base

The server builds it automatically on first start, but you can also run it manually:

```bash
python obsidian_connector.py /Users/yourname/Documents/MyVault
```

With options:
```bash
python obsidian_connector.py /path/to/vault \
    --output generated_sources/knowledge_base.pkl \
    --model all-MiniLM-L6-v2 \
    --exclude-dirs templates archive .trash \
    --exclude-files README.md
```

This step can take 30 seconds to a few minutes depending on vault size. It only needs to run again when you add new notes.

---

## Running the app

```bash
uvicorn app:app --reload --port 8000
```

Then open your browser at:
```
http://localhost:8000
```

---

## Using the chat interface

**Asking questions** - type your question and press `Enter`. The model retrieves the most relevant passages from your notes and uses them to answer. It remembers the conversation so follow-up questions like *"tell me more"* or *"what did you mean by that?"* work correctly.

**Token counter** - if you're using OpenAI, each response shows the cumulative token count. Ollama does not report real token counts so this is hidden.

**Reset** - clears the conversation history on both the frontend and the server. The knowledge base is not affected. Use this to start a fresh topic.

**Settings panel** - click ⚙ Settings in the header to:
- Change the vault path
- Update excluded folders
- Point to a different knowledge base file
- Trigger a full re-index of the vault

Changes take effect immediately without restarting the server.

<img width="1116" height="620" alt="vault-chatbot-settings" src="https://github.com/user-attachments/assets/0673219a-d7f2-4680-8d00-8d1bf5a074b6" />

---

## Configuration reference

All model behaviour is controlled in `rags/rag_config.py`:

```python
class Config:
    TOP_K          = 0.05      # top 5% of chunks by similarity score
    MIN_SIMILARITY = 0.3       # minimum cosine similarity to include a chunk
    DELTA_CUTOFF   = 0.15      # drop chunks if similarity drops sharply
    MAX_CONTEXT_TOKENS = 2000  # max tokens to send as context

    ANSWER_PROMPT = """You are a helpful assistant. Answer the user's question
    using ONLY the provided context passages. If the answer is not in the
    context, say so clearly."""

    REPHRASE_PROMPT = """Rephrase the following question to be more precise
    and retrieval-friendly. Return only the rephrased question."""
```

| Parameter | Effect |
|---|---|
| `TOP_K` | Higher = more chunks retrieved, slower but more context |
| `MIN_SIMILARITY` | Lower = looser matching, may retrieve less relevant chunks |
| `DELTA_CUTOFF` | Controls how aggressively low-relevance chunks are cut off |
| `MAX_CONTEXT_TOKENS` | Hard cap on how much context is sent to the LLM |

---

## Using OpenAI instead of Ollama

Update `api.py`:

```python
client = LLMClient(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini"
)
rag = RAG(client=client, embedding_model=embed_model, df=df, llm_model="gpt-4o-mini")
```

Or store the key in `.env`:

```bash
OPENAI_API_KEY=sk-...
```

And load it in `api.py`:

```python
client = LLMClient(
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

When using OpenAI, cumulative token counts are displayed in the UI after each response.

---

## Rebuilding the index

You need to rebuild when you add, edit, or delete notes in your vault:

**Via the UI** — open Settings → click **Rebuild Index**

**Via the API:**
```bash
curl -X POST http://localhost:8000/settings/rebuild
```

**Via CLI:**
```bash
python obsidian_connector.py /path/to/vault --output generated_sources/knowledge_base.pkl
```

The server reloads the new index automatically without a restart.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/chat` | Send a message, get an answer |
| `POST` | `/reset` | Clear conversation history |
| `GET` | `/settings` | Read current `.env` config |
| `POST` | `/settings` | Save config and rebuild chatbot |
| `POST` | `/settings/rebuild` | Force re-index the vault |
| `GET` | `/health` | Check if server is ready |

### POST /chat

```json
// request
{ "message": "What is RAG?" }

// response
{ "answer": "RAG stands for...", "total_tokens": 0 }
```

### POST /settings

```json
{
  "vault_path": "/Users/you/vault",
  "exclude_folders": "templates,archive",
  "knowledge_base": "generated_sources/knowledge_base.pkl"
}
```

---

## Troubleshooting

**`404 Not Found` from Ollama**
The model name doesn't match what's installed. Run `ollama list` and use the exact name shown.

**Empty Ollama list / no models**
```bash
ollama pull llama3.2
```

**Answers are irrelevant or hallucinated**

Lower `MIN_SIMILARITY` in `rag_config.py` to retrieve more chunks, or rebuild the index after updating your notes.

**Slow responses**

Switch to a smaller model (`llama3.2:1b`). First response after starting Ollama is always slower — subsequent ones are faster as the model stays loaded in memory.

---

## Privacy

Everything runs locally. Your notes, queries, and conversation history never leave your machine when using Ollama. The only network requests are to Google Fonts for the UI typography and to `huggingface.co` once to download the embedding model on first run.

---

## Demo

▶ [Watch the full demo - querying Vault in action](https://drive.google.com/file/d/1ZTRFv3HtKwf3BFj4jp_u6shBhU8wo-zv/view?usp=share_link)
