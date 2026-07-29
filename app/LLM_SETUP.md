# OpenAI and Ollama setup

The application uses one common LLM interface and selects the provider at startup through `LLM_PROVIDER`.

## 1. Create the environment file

Copy `.env.example` to `.env`.

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.5
```

### Ollama

Install Ollama, then run:

```bash
ollama pull llama3
ollama serve
```

Use:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run

From the folder that contains the `app` directory:

```bash
python -m app.main
```

The active provider and model appear in the application header.

## Deployment note

`localhost` means the same machine or container running the application. A cloud deployment cannot reach Ollama running on your personal computer through `localhost`; Ollama must be deployed in a reachable environment and `OLLAMA_BASE_URL` must point to it.
