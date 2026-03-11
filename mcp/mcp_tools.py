from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Type

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, ValidationError

try:  # pragma: no cover - import style depends on invocation mode
    from .mcp_contracts import SimulateEnzymeDynamicsRequest
    from .mcp_errors import ToolExecutionError, error_response, success_response
    from .mcp_simulation import EnzymeCliRunner
except ImportError:  # pragma: no cover
    MODULE_DIR = Path(__file__).resolve().parent
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    from mcp_contracts import SimulateEnzymeDynamicsRequest  # type: ignore
    from mcp_errors import ToolExecutionError, error_response, success_response  # type: ignore
    from mcp_simulation import EnzymeCliRunner  # type: ignore


class EnzymeMcpService:
    def __init__(self, runner: EnzymeCliRunner | None = None) -> None:
        self.runner = runner or EnzymeCliRunner()

    def _validate_request(
        self,
        payload: Dict[str, Any],
        model: Type[BaseModel],
    ) -> BaseModel:
        return model.parse_obj(payload)

    def simulate_enzyme_dynamics(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed = self._validate_request(request, SimulateEnzymeDynamicsRequest)
            response = self.runner.simulate(parsed)
        except ValidationError as exc:
            return error_response(
                code="validation_error",
                message="Request payload failed schema validation.",
                details={"errors": exc.errors()},
            )
        except ToolExecutionError as exc:
            return error_response(
                code=exc.code, message=exc.message, details=exc.details
            )
        except Exception as exc:  # pragma: no cover
            return error_response(
                code="internal_error",
                message="Internal error while running simulation.",
                details={"type": type(exc).__name__},
            )

        return success_response(response.dict())
