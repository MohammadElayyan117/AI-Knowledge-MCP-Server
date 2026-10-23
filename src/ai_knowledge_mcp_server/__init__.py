
def main() -> None:
    """Start the AI Knowledge MCP Server using stdio transport."""
    from ai_knowledge_mcp_server.server import mcp

    mcp.run()
