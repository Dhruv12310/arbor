"""
Arbor Quickstart — vectorless RAG in 5 lines.

Requires: GROQ_API_KEY environment variable (free tier at console.groq.com)

    pip install arbor-rag
    export GROQ_API_KEY=your_key_here
    python examples/quickstart.py
"""

import asyncio
import arbor

# ── 1. Choose a provider ──────────────────────────────────────────────────────

# Option A: Groq (free tier — fastest to get started)
provider = arbor.GroqProvider()                        # uses GROQ_API_KEY env var

# Option B: Ollama (100% local, zero cost)
# provider = arbor.OllamaProvider(model="qwen2.5:7b")  # requires: ollama pull qwen2.5:7b

# Option C: OpenAI
# provider = arbor.OpenAIProvider(model="gpt-4o-mini")  # uses OPENAI_API_KEY env var

# Option D: Claude
# provider = arbor.AnthropicProvider(model="haiku")     # uses ANTHROPIC_API_KEY env var


# ── 2. Full RAG pipeline ──────────────────────────────────────────────────────

async def main():
    document = "your_document.pdf"   # <-- replace with your PDF path

    # One-shot RAG: generate tree + search + answer
    response = await arbor.query(
        document=document,
        question="What are the main conclusions?",
        provider=provider,
    )

    print(f"Answer: {response.answer}")
    print(f"\nCitations:")
    for c in response.citations:
        print(f"  p.{c['start_page']}-{c['end_page']}: {c['title']}")


# ── 3. Advanced: reuse a pre-built tree ───────────────────────────────────────

async def advanced():
    document = "your_document.pdf"   # <-- replace with your PDF path

    # Build the tree once (expensive — involves many LLM calls)
    tree = await arbor.generate_tree(document, provider=provider)

    # Ask multiple questions without rebuilding
    questions = [
        "What is the methodology?",
        "What are the key results?",
        "What limitations are discussed?",
    ]

    for question in questions:
        result = await arbor.search_tree(tree, question, provider)
        print(f"\nQ: {question}")
        print(f"Top nodes: {[n.title for n in result.nodes[:3]]}")

    # Full answer for the last question
    response = await arbor.query(
        document=document,
        question=questions[-1],
        provider=provider,
        tree=tree,   # reuse tree — skips generation
    )
    print(f"\nAnswer: {response.answer}")


if __name__ == "__main__":
    asyncio.run(main())
