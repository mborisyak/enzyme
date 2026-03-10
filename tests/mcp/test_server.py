from __future__ import annotations

import runpy


def test_only_simulate_tool_is_exposed() -> None:
    namespace = runpy.run_path("mcp/server.py", run_name="not_main")
    server = namespace["server"]
    assert list(server._tool_manager._tools.keys()) == ["simulate_enzyme_dynamics"]

