from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional


class TruckCommandKind(str, Enum):
    MOVE_TO_TASK = "move_to_task"
    SUPPORT_UAV = "support_uav"
    SAFETY_RECOVERY = "safety_recovery"
    HOLD = "hold"


class UAVCommandKind(str, Enum):
    LAUNCH_TC = "launch_tc"
    RETURN_OR_RTH = "return_or_rth"
    HOLD = "hold"


@dataclass(frozen=True)
class TruckCommand:
    agent_id: str
    kind: TruckCommandKind
    task_id: Optional[str] = None
    target_node: Optional[int] = None
    support_uav_id: Optional[str] = None
    support_point: Optional[int] = None
    launch_anchor: Optional[int] = None
    recovery_anchor: Optional[int] = None
    ttl_steps: int = 0
    expected_launch_step: Optional[int] = None
    expected_delivery_step: Optional[int] = None
    expected_recovery_margin: float = 0.0
    safety_reason: str = ""
    reason: str = ""


@dataclass(frozen=True)
class UAVCommand:
    agent_id: str
    kind: UAVCommandKind
    task_id: Optional[str] = None
    bind_truck_id: Optional[str] = None
    support_command_id: Optional[str] = None
    sortie_id: Optional[str] = None
    launch_anchor: Optional[int] = None
    recovery_anchor: Optional[int] = None
    reason: str = ""


@dataclass
class CommandBatch:
    truck_commands: Dict[str, TruckCommand] = field(default_factory=dict)
    uav_commands: Dict[str, UAVCommand] = field(default_factory=dict)

    def add_truck(self, command: TruckCommand) -> None:
        self.truck_commands[str(command.agent_id)] = command

    def add_uav(self, command: UAVCommand) -> None:
        self.uav_commands[str(command.agent_id)] = command

    def goals(self) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for aid, cmd in self.truck_commands.items():
            out[str(aid)] = cmd.task_id
        for aid, cmd in self.uav_commands.items():
            out[str(aid)] = cmd.task_id
        return out

    def commands(self) -> Iterable[object]:
        yield from self.truck_commands.values()
        yield from self.uav_commands.values()


class CommandValidator:
    @staticmethod
    def is_truck_support_authorized(batch: Optional[CommandBatch], agent_id: str, mode: str = "") -> bool:
        if batch is None:
            return False
        cmd = batch.truck_commands.get(str(agent_id))
        if cmd is None:
            return False
        if str(mode) == "recovery":
            return cmd.kind in {TruckCommandKind.SAFETY_RECOVERY, TruckCommandKind.SUPPORT_UAV}
        return cmd.kind == TruckCommandKind.SUPPORT_UAV

    @staticmethod
    def is_uav_launch_authorized(batch: Optional[CommandBatch], agent_id: str, task_id: Optional[str]) -> bool:
        if batch is None:
            return False
        cmd = batch.uav_commands.get(str(agent_id))
        if cmd is None or cmd.kind != UAVCommandKind.LAUNCH_TC:
            return False
        return task_id is None or str(cmd.task_id) == str(task_id)
