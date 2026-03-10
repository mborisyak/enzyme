from __future__ import annotations

import math
from typing import Any, Dict, List, NotRequired, Optional, Required, TypedDict

try:
    from pydantic.v1 import BaseModel, Field, root_validator, validator
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, root_validator, validator

TOOL_CONTRACT_VERSION = "1.0"


def _ensure_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _ensure_json_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number.")
    return float(value)


def _ensure_json_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


class EnzymeConditionPayload(TypedDict):
    A: float
    B: float
    E: float
    temperature: float


class TimeConfigPayload(TypedDict, total=False):
    t_start: float
    t_end: float
    measurements: int


class SolutionConcentrationsPayload(TypedDict, total=False):
    A: float
    B: float
    E: float


class UnitsMapPayload(TypedDict, total=False):
    time: str
    temperature: str
    solution_volume: str
    concentration: str


class SimulateEnzymeDynamicsRequestPayload(TypedDict):
    conditions: Required[Dict[str, EnzymeConditionPayload]]
    contract_version: NotRequired[str]
    time: NotRequired[TimeConfigPayload]
    solutions: NotRequired[SolutionConcentrationsPayload]
    device: NotRequired[str]
    units: NotRequired[UnitsMapPayload]


class EnzymeCondition(BaseModel):
    A: float = Field(..., description="Volume of substrate A solution.")
    B: float = Field(..., description="Volume of substrate B solution.")
    E: float = Field(..., description="Volume of enzyme solution.")
    temperature: float = Field(..., description="Temperature in Celsius.")

    class Config:
        extra = "forbid"

    @validator("A", "B", "E", pre=True)
    def validate_non_negative_volume(cls, value: float, field: Any) -> float:
        value = _ensure_json_number(field.name, value)
        value = _ensure_finite(field.name, value)
        if value < 0.0:
            raise ValueError(f"{field.name} must be non-negative.")
        return value

    @validator("temperature", pre=True)
    def validate_temperature(cls, value: float) -> float:
        value = _ensure_json_number("temperature", value)
        return _ensure_finite("temperature", value)


class TimeConfig(BaseModel):
    t_start: float = Field(0.0, description="Start time in seconds.")
    t_end: float = Field(30.0, description="End time in seconds.")
    measurements: int = Field(10, description="Number of measurement points.")

    # nested class for pydantic
    class Config:
        extra = "forbid"

    @validator("t_start", "t_end", pre=True)
    def validate_finite_times(cls, value: float, field: Any) -> float:
        value = _ensure_json_number(field.name, value)
        return _ensure_finite(field.name, value)

    @validator("measurements", pre=True)
    def validate_measurements(cls, value: int) -> int:
        value = _ensure_json_int("measurements", value)
        if value <= 1:
            raise ValueError("measurements must be greater than 1.")
        return value

    @root_validator
    def validate_time_window(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        t_start = values.get("t_start")
        t_end = values.get("t_end")
        if t_start is not None and t_end is not None and t_end <= t_start:
            raise ValueError("t_end must be greater than t_start.")
        return values


class SolutionConcentrations(BaseModel):
    A: float = Field(3.0, description="Concentration of A solution in mM.")
    B: float = Field(3.0, description="Concentration of B solution in mM.")
    E: float = Field(3.0e-3, description="Concentration of E solution in mM.")

    # nested class for pydantic
    class Config:
        extra = "forbid"

    @validator("A", "B", "E", pre=True)
    def validate_solution_concentration(cls, value: float, field: Any) -> float:
        value = _ensure_json_number(field.name, value)
        value = _ensure_finite(field.name, value)
        if value < 0.0:
            raise ValueError(f"{field.name} concentration must be non-negative.")
        return value


class UnitsMap(BaseModel):
    time: str = "s"
    temperature: str = "C"
    solution_volume: str = "mL"
    concentration: str = "mM"

    # nested class for pydantic
    class Config:
        extra = "forbid"


class SimulateEnzymeDynamicsRequest(BaseModel):
    contract_version: str = Field(TOOL_CONTRACT_VERSION)
    conditions: Dict[str, EnzymeCondition]
    time: TimeConfig = Field(default_factory=TimeConfig)
    solutions: SolutionConcentrations = Field(default_factory=SolutionConcentrations)
    device: str = Field("cpu", description="JAX device name, e.g. cpu or gpu:0.")
    units: UnitsMap = Field(default_factory=UnitsMap)

    # nested class for pydantic
    class Config:
        extra = "forbid"

    @validator("contract_version")
    def validate_contract_version(cls, value: str) -> str:
        if value != TOOL_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract_version '{value}'. Expected '{TOOL_CONTRACT_VERSION}'."
            )
        return value

    @validator("conditions")
    def validate_conditions(
        cls, value: Dict[str, EnzymeCondition]
    ) -> Dict[str, EnzymeCondition]:
        if not value:
            raise ValueError("conditions must contain at least one experiment.")
        return value

    @validator("device", pre=True)
    def validate_device(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("device must be a string.")
        if not value.strip():
            raise ValueError("device must be a non-empty string.")
        return value

class ExperimentTrajectory(BaseModel):
    time_points: List[float]
    state_trajectories: Dict[str, List[float]]

    @validator("time_points")
    def validate_time_points(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("time_points cannot be empty.")
        return [_ensure_finite("time_points", float(v)) for v in value]

    @validator("state_trajectories")
    def validate_state_trajectories(
        cls,
        value: Dict[str, List[float]],
        values: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        if not value:
            raise ValueError("state_trajectories cannot be empty.")

        expected = len(values.get("time_points", []))
        cleaned: Dict[str, List[float]] = {}
        for name, series in value.items():
            if len(series) != expected:
                raise ValueError(
                    f"Length mismatch for '{name}': expected {expected}, got {len(series)}."
                )
            cleaned[name] = [_ensure_finite(name, float(v)) for v in series]
        return cleaned


class MetadataofRun(BaseModel):
    model_identifier: str
    model_version: str
    solver: Dict[str, Any]
    units_map: Dict[str, str]
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    deterministic: bool
    seed: Optional[int]
    tool_contract_version: str = TOOL_CONTRACT_VERSION


class SimulateEnzymeDynamicsResponse(BaseModel):
    contract_version: str = TOOL_CONTRACT_VERSION
    experiments: Dict[str, ExperimentTrajectory]
    metadata: MetadataofRun
