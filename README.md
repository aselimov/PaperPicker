# Paper Picker

Paper Picker is a CLI-first Python tool that:

1. Pulls recent papers from arXiv categories in a config file.
2. Filters papers published in the last 7 days.
3. Uses a local LLM (Ollama or llama.cpp server) to rank papers.
4. Lets you select one or more papers to download to `~/Downloads`.
5. Prints APA-style citations for each downloaded paper.

## Quick start (uv)

```bash
uv sync --dev
uv run paper-picker -n 5
```

## Run tests (pytest via uv)

```bash
uv run pytest
```

## Configuration

Edit `config.toml`:

- `arxiv.categories`: list of arXiv categories (for example, `cs.LG`, `cs.CL`).
- `arxiv.max_results`: API query size before filtering to last week.
- `llm.provider`: `ollama` or `llama_cpp`.
- `llm.model`: model name.
- `llm.base_url`: endpoint base URL:
  - Ollama example: `http://localhost:11434`
  - llama.cpp server (OpenAI-compatible) example: `http://localhost:8080`
- `llm.prompt`: ranking instructions given to the model.

## Notes

- `-n` / `--num` controls how many top papers the model should return.
- The model is instructed to return strict JSON and scores in `[0, 100]`.
- APA citations are printed to stdout after download.
