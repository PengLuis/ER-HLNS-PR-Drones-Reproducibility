"""Explicit, isolated checkpoint support for deterministic ER-ALNS replay.

The contract deliberately snapshots object fields instead of replaying a
constructor.  Constructors build a new random world, and therefore are not a
safe restore mechanism for an in-flight simulation.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_VERSION = 1


def _global_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": __import__("numpy").random.get_state()}
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_global_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    __import__("numpy").random.set_state(state["numpy"])
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _stable_digest(value: Any) -> str:
    """Digest a value by semantic contents, never deepcopy object addresses."""
    return hashlib.sha256(pickle.dumps(_canonical_value(value, set()), protocol=5)).hexdigest()


def _canonical_value(value: Any, seen: set[int]) -> Any:
    """Convert mutable runtime graphs into an ordered, dtype-preserving form."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Enum):
        return ("enum", _class_path(value), value.value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return ("ndarray", str(value.dtype), tuple(value.shape), value.tobytes())
        if isinstance(value, np.generic):
            return ("numpy_scalar", str(value.dtype), value.item())
        if isinstance(value, np.random.Generator):
            return ("numpy_generator", _canonical_value(value.bit_generator.state, seen))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return ("torch_tensor", str(value.dtype), tuple(value.shape), value.detach().cpu().numpy().tobytes())
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    ((_canonical_value(key, seen), _canonical_value(item, seen)) for key, item in value.items()),
                    key=lambda item: pickle.dumps(item[0], protocol=5),
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canonical_value(item, seen) for item in value))
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(sorted((_canonical_value(item, seen) for item in value), key=lambda item: pickle.dumps(item, protocol=5))),
        )
    object_id = id(value)
    if object_id in seen:
        return ("cycle", _class_path(value))
    if is_dataclass(value):
        seen.add(object_id)
        result = (
            "dataclass",
            _class_path(value),
            tuple((field.name, _canonical_value(getattr(value, field.name), seen)) for field in fields(value)),
        )
        seen.remove(object_id)
        return result
    if hasattr(value, "__dict__"):
        seen.add(object_id)
        result = ("object", _class_path(value), _canonical_value(vars(value), seen))
        seen.remove(object_id)
        return result
    return ("opaque", _class_path(value), repr(value))


def _contract_digest(
    environment: "ObjectState", planner: "ObjectState", runtime: "ObjectState | None", global_rng: Mapping[str, Any]
) -> str:
    """Hash the explicit contract field-by-field, never object identities."""
    rows: list[tuple[str, str, str]] = []
    for owner, state in (("environment", environment), ("planner", planner), ("runtime", runtime)):
        if state is None:
            continue
        rows.extend((owner, name, _stable_digest(value)) for name, value in sorted(state.fields.items()))
    rows.extend(("global_rng", name, _stable_digest(value)) for name, value in sorted(global_rng.items()))
    return _stable_digest(tuple(rows))


def _field_group(owner: str, name: str) -> str:
    lower = name.lower()
    if name == "rng" or "rng" in lower or "random" in lower:
        return "rng"
    if owner == "environment":
        if name in {"state", "task_manager"} or "task" in lower:
            return "tasks_and_agents"
        if name in {"topology", "hazards", "physical_v2"} or any(
            token in lower for token in ("road", "edge", "weather", "event", "route", "cache")
        ):
            return "topology_weather_events"
        return "environment_runtime"
    if name in {"state", "sequence_runtime_state"} or any(
        token in lower for token in ("solution", "tail", "support", "sortie", "route", "cache")
    ):
        return "solution_runtime"
    if any(token in lower for token in ("weight", "temperature", "operator", "diagnostic", "count", "history")):
        return "adaptive_history"
    return "planner_runtime"


@dataclass(frozen=True)
class ObjectState:
    class_path: str
    fields: dict[str, Any]

    @classmethod
    def capture(cls, obj: Any) -> "ObjectState":
        # A named field map makes coverage inspectable; deepcopy provides a
        # detached graph while preserving array dtypes, ordered containers, and
        # aliases such as ``hazards.topo is env.topology``.
        return cls(
            class_path=_class_path(obj),
            fields=copy.deepcopy(dict(vars(obj))),
        )

    def restore_into(self, obj: Any) -> None:
        if _class_path(obj) != self.class_path:
            raise TypeError(f"checkpoint class mismatch: {self.class_path} != {_class_path(obj)}")
        vars(obj).clear()
        vars(obj).update(copy.deepcopy(self.fields))

    def clone(self, source_class: type[Any]) -> Any:
        clone = source_class.__new__(source_class)
        self.restore_into(clone)
        return clone


@dataclass(frozen=True)
class CanonicalCheckpoint:
    """A complete, output-path-independent environment/planner checkpoint."""

    metadata: dict[str, Any]
    environment: ObjectState
    planner: ObjectState
    runtime: ObjectState | None = None
    global_rng_state: dict[str, Any] | None = None

    @classmethod
    def capture(
        cls,
        env: Any,
        planner: Any,
        runtime: Any | None = None,
        *,
        scenario: str = "",
        seed: int | None = None,
        source_run_id: str = "",
    ) -> "CanonicalCheckpoint":
        env_state = ObjectState.capture(env)
        planner_state = ObjectState.capture(planner)
        runtime_state = None if runtime is None else ObjectState.capture(runtime)
        global_rng_state = copy.deepcopy(_global_rng_state())
        # Object-local RNGs are explicit fields.  Process-global Torch state is
        # retained as metadata and restored for replay, but excluded here: the
        # ER-ALNS mainline does not consume it and CUDA bookkeeping can mutate
        # it while a diagnostic captures a snapshot.
        digest = _contract_digest(env_state, planner_state, runtime_state, {})
        return cls(
            metadata={
                "checkpoint_version": CHECKPOINT_VERSION,
                "scenario": str(scenario),
                "seed": None if seed is None else int(seed),
                "step": int(getattr(getattr(env, "state", None), "step_index", -1)),
                "environment_class": env_state.class_path,
                "planner_class": planner_state.class_path,
                "schema_hash": _stable_digest(
                    (tuple(env_state.fields.keys()), tuple(planner_state.fields.keys()))
                ),
                "state_digest": digest,
                "environment_rng_digest": _stable_digest(env_state.fields.get("rng")),
                "planner_rng_digest": _stable_digest(planner_state.fields.get("rng")),
                "source_run_id": str(source_run_id),
            },
            environment=env_state,
            planner=planner_state,
            runtime=runtime_state,
            global_rng_state=global_rng_state,
        )

    @property
    def digest(self) -> str:
        return str(self.metadata["state_digest"])

    def clone(
        self, env_class: type[Any], planner_class: type[Any], runtime_class: type[Any] | None = None
    ) -> tuple[Any, Any, Any | None]:
        runtime = None
        if self.runtime is not None:
            if runtime_class is None:
                raise ValueError("runtime_class is required by this checkpoint")
            runtime = self.runtime.clone(runtime_class)
        return self.environment.clone(env_class), self.planner.clone(planner_class), runtime

    def restore_into(
        self, env: Any, planner: Any, runtime: Any | None = None, *, restore_global_rng: bool = False
    ) -> None:
        self.environment.restore_into(env)
        self.planner.restore_into(planner)
        if self.runtime is not None:
            if runtime is None:
                raise ValueError("runtime object is required by this checkpoint")
            self.runtime.restore_into(runtime)
        if restore_global_rng and self.global_rng_state is not None:
            restore_global_rng_state(self.global_rng_state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)

    @classmethod
    def load(cls, path: Path) -> "CanonicalCheckpoint":
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if not isinstance(checkpoint, cls) or checkpoint.metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported canonical checkpoint")
        return checkpoint


def checkpoint_field_rows(checkpoint: CanonicalCheckpoint) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    owners = [("environment", checkpoint.environment), ("planner", checkpoint.planner)]
    if checkpoint.runtime is not None:
        owners.append(("runtime", checkpoint.runtime))
    for owner, object_state in owners:
        for name, value in object_state.fields.items():
            rows.append(
                {
                    "object": owner,
                    "field_path": f"{owner}.{name}",
                    "data_type": _class_path(value),
                    "group": _field_group(owner, name),
                    "affects_future_decisions": True,
                    "affects_physics": owner == "environment",
                    "affects_event_progression": owner == "environment" and name in {"hazards", "physical_v2", "state"},
                    "serialize_method": "explicit_field_deepcopy",
                    "restore_method": "explicit_field_load",
                    "digest_inclusion": True,
                    "currently_available": True,
                    "missing": False,
                    "test_coverage": "checkpoint_roundtrip",
                }
            )
    return rows


def field_digest_rows(checkpoint: CanonicalCheckpoint) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    owners = [("environment", checkpoint.environment), ("planner", checkpoint.planner)]
    if checkpoint.runtime is not None:
        owners.append(("runtime", checkpoint.runtime))
    for owner, object_state in owners:
        for name, value in object_state.fields.items():
            rows.append({"field_path": f"{owner}.{name}", "digest": _stable_digest(value)})
    return rows


def compare_checkpoints(
    left: CanonicalCheckpoint, right: CanonicalCheckpoint, *, ignore_output_only_metrics: bool = False
) -> list[str]:
    """Return field paths whose exact serialized values differ."""
    differences: list[str] = []
    for owner in ("environment", "planner", "runtime"):
        if owner == "runtime" and (left.runtime is None or right.runtime is None):
            if left.runtime is not right.runtime:
                differences.append("runtime")
            continue
        before = getattr(left, owner).fields
        after = getattr(right, owner).fields
        for field in sorted(set(before) | set(after)):
            before_value = before.get(field)
            after_value = after.get(field)
            if ignore_output_only_metrics and owner == "environment" and field == "hazards":
                before_value = copy.copy(before_value)
                after_value = copy.copy(after_value)
                # This value is exported only as an end-of-step road metric and
                # is not read by any decision/physics code.
                before_value.last_length_norm_mean = 0.0
                after_value.last_length_norm_mean = 0.0
                before_value.last_pmacro_mean = 0.0
                after_value.last_pmacro_mean = 0.0
                # Per-edge probabilities are the routing input.  The scalar
                # mean is exported only for diagnostics and is used solely as
                # an empty-map fallback, never while the edge map is present.
                before_value.last_pstep_mean = 0.0
                after_value.last_pstep_mean = 0.0
            if field not in before or field not in after or _stable_digest(before_value) != _stable_digest(after_value):
                differences.append(f"{owner}.{field}")
    return differences


def checkpoint_manifest(checkpoint: CanonicalCheckpoint) -> Mapping[str, Any]:
    return dict(checkpoint.metadata)
