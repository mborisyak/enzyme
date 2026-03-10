from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: install the `mcp` package to run this server."
    ) from exc

try:  # pragma: no cover - import style depends on invocation mode
    from .mcp_contracts import SimulateEnzymeDynamicsRequestPayload
    from .mcp_tools import EnzymeMcpService
except ImportError:  # pragma: no cover
    MODULE_DIR = Path(__file__).resolve().parent
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    from mcp_contracts import SimulateEnzymeDynamicsRequestPayload  # type: ignore
    from mcp_tools import EnzymeMcpService  # type: ignore

service = EnzymeMcpService()

server = FastMCP(
    name="enzyme-cli-mcp",
    instructions=(
        "MCP server for enzyme-reaction experiments."
        "Exposed tool: simulate_enzyme_dynamics. "
        "Call simulate_enzyme_dynamics with conditions(A,B,E, temperature) and time settings(t_start, t_end and measurements) to "
        "compute trajectories per experiment. "
        # "Kinetic parameters are loaded from config/parameters-123456789.yaml. "
        "Requests should be strict JSON objects with finite numeric values; responses "
        "use a structured envelope: {ok: true, data: ...} on success or "
        "{ok: false, error: {code, message, details}} on failure."
    ),
)


@server.tool()
def simulate_enzyme_dynamics(
    request: SimulateEnzymeDynamicsRequestPayload,
) -> Dict[str, Any]:
    """Run enzyme dynamics simulation by invoking scripts/experiment.py."""
    return service.simulate_enzyme_dynamics(request)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
