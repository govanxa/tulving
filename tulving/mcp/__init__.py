"""Tulving MCP server package.

Empty marker by design: importing ``tulving.mcp`` (and ``tulving`` itself)
must never pull in the ``mcp``/``anyio`` packages. Those live behind the
``[mcp]`` extra and are imported lazily inside ``tulving.mcp.server`` (gated
by ``_require_mcp``), so a core-only install can still import this package.
"""
