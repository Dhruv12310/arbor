# Changelog

All notable changes to Arbor will be documented here.

## v0.1.0 (2026-03-11)

Initial release.

- Tree generation with 3-mode fallback (TOC-with-pages → TOC-no-pages → no-TOC)
- Tree search with LLM reasoning (navigates summaries, not full text)
- Full RAG pipeline (generate → search → answer) via `arbor.query()`
- 5 LLM providers: Groq (free), Ollama (local), OpenAI, Claude, any OpenAI-compatible
- Recursive node subdivision for large sections
- Markdown document support (regex-based, no LLM needed)
- Accuracy-based TOC verification with automatic fixing
- Truncation continuation: handles LLM responses cut off mid-JSON
- Async-first design with configurable concurrency (`max_concurrent_llm_calls`)
- 185 tests (181 unit + 4 E2E on "Attention Is All You Need")
