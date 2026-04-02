"""
Arbor MCP Server — exposes Arbor pipeline as MCP tools.

Usage (STDIO — for Claude Code, Cursor, Claude Desktop):
    python -m arbor.mcp_server

Usage (HTTP/SSE — for web clients):
    python -m arbor.mcp_server --http --port 8080

Add to Claude Code (.mcp.json in project root):
    {
      "mcpServers": {
        "arbor": {
          "command": "python",
          "args": ["-m", "arbor.mcp_server"],
          "env": {"GEMINI_API_KEY": "your_key"}
        }
      }
    }
"""

import argparse
import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

import arbor
from arbor.providers.gemini_provider import GeminiProvider


def _get_default_provider() -> arbor.LLMProvider:
    """Build provider from environment variables. Tries GEMINI first, then OPENAI."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        return arbor.OpenAIProvider()
    if os.environ.get("GROQ_API_KEY"):
        return arbor.GroqProvider()
    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY."
    )


def create_server() -> Server:
    server = Server("arbor")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="query_document",
                description=(
                    "Query a PDF document using Arbor's vectorless RAG pipeline. "
                    "Parses the PDF into a hierarchical tree, navigates the tree to find "
                    "relevant sections, and generates a grounded answer with citations. "
                    "No vector database required."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["pdf_path", "question"],
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Absolute path to the PDF file to query",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer from the document",
                        },
                        "max_hops": {
                            "type": "integer",
                            "description": "Max tree levels to navigate (default: 5)",
                            "default": 5,
                        },
                        "max_cost_usd": {
                            "type": "number",
                            "description": "Max API spend in USD (default: 0.50, 0 = unlimited)",
                            "default": 0.50,
                        },
                    },
                },
            ),
            Tool(
                name="generate_tree",
                description=(
                    "Parse a PDF into a hierarchical JSON tree structure. "
                    "Returns the tree with section titles, page ranges, node IDs, and summaries. "
                    "Use this to inspect document structure before querying."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["pdf_path"],
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Absolute path to the PDF file",
                        },
                        "add_summaries": {
                            "type": "boolean",
                            "description": "Generate LLM summaries for each node (default: true)",
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="search_tree",
                description=(
                    "Search a pre-generated Arbor document tree for nodes relevant to a question. "
                    "Uses multi-hop navigation: navigates level-by-level, 10-20 sections at a time. "
                    "Returns relevant node IDs, titles, page ranges, and navigation reasoning."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["tree_json", "question"],
                    "properties": {
                        "tree_json": {
                            "type": "string",
                            "description": "JSON string of a DocumentTree (from generate_tree output)",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to find relevant sections for",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        provider = _get_default_provider()

        if name == "query_document":
            pdf_path = arguments["pdf_path"]
            question = arguments["question"]
            config = arbor.ArborConfig(
                max_hops=arguments.get("max_hops", 5),
                max_cost_usd=arguments.get("max_cost_usd", 0.50),
                add_summaries=True,
            )

            nav_events: list[str] = []
            answer_text = ""
            citations: list = []

            async for event in arbor.query_stream(pdf_path, question, provider, config):
                if isinstance(event, arbor.TreeLoadedEvent):
                    nav_events.append(
                        f"Loaded tree: {event.node_count} nodes, {event.page_count} pages"
                    )
                elif isinstance(event, arbor.NavigatingEvent):
                    nav_events.append(
                        f"→ Level {event.level}: exploring {event.section_titles}"
                    )
                elif isinstance(event, arbor.NodeFoundEvent):
                    nav_events.append(
                        f"✓ Found: {event.title} (pages {event.page_range})"
                    )
                elif isinstance(event, arbor.AnswerEvent):
                    answer_text = event.text
                    citations = event.citations or []
                elif isinstance(event, arbor.ErrorEvent):
                    return [TextContent(type="text", text=f"Error: {event.message}")]

            lines: list[str] = [f"**Answer:** {answer_text}"]
            if citations:
                lines.append("\n**Sources:**")
                for c in citations:
                    lines.append(f"  - [{c.get('title')}] pages {c.get('start_page')}-{c.get('end_page')}")
            if nav_events:
                lines.append("\n**Navigation path:**")
                for e in nav_events:
                    lines.append(f"  {e}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "generate_tree":
            pdf_path = arguments["pdf_path"]
            config = arbor.ArborConfig(
                add_summaries=arguments.get("add_summaries", True),
                add_node_ids=True,
            )
            tree = await arbor.generate_tree(pdf_path, provider, config)
            return [TextContent(type="text", text=json.dumps(tree.to_dict(), indent=2))]

        elif name == "search_tree":
            tree_data = json.loads(arguments["tree_json"])
            question = arguments["question"]
            tree = arbor.DocumentTree.from_dict(tree_data)
            result = await arbor.search_tree(tree, question, provider, multihop=True)
            output = {
                "node_ids": result.node_ids,
                "thinking": result.thinking,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "title": n.title,
                        "pages": f"{n.start_index}-{n.end_index}",
                    }
                    for n in (result.nodes or [])
                ],
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _run_stdio() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Arbor MCP Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP/SSE server instead of STDIO")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args()

    if args.http:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        import uvicorn

        server = create_server()
        sse = SseServerTransport("/messages")

        async def handle_sse(request):  # type: ignore[no-untyped-def]
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse.handle_post_message),
        ])
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
