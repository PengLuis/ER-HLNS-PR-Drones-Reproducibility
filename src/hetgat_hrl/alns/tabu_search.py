from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

from hetgat_hrl.alns.objective import minimization_delta
from hetgat_hrl.alns.sequence import construct_k2_solution, evaluate_sequence_feasibility
from hetgat_hrl.alns.solution import ALNSSolution, StableId
from hetgat_hrl.core.mdp_spec import TaskStatus
from hetgat_hrl.hrl.event_responsive_alns_planner import EventResponsiveALNSPlanner


class TabuSearchK2Planner(EventResponsiveALNSPlanner):
    """K2 tabu-search baseline using the shared objective and feasibility checker.

    This is intentionally not an ALNS alias: it does not call destroy/repair
    operators. Neighbors are generated directly from K2 sequences using relocate,
    swap, tail replacement, and agent reassignment moves.
    """

    def __init__(self, *args, tabu_tenure: int = 7, candidate_limit: int = 96, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tabu_tenure = int(max(tabu_tenure, 1))
        self.tabu_candidate_limit = int(max(candidate_limit, 8))
        self._operator_pool_name = "tabu_k2"

    def _pending_task_ids(self, env) -> Tuple[str, ...]:
        out = []
        for task in getattr(env.state, "tasks", {}).values():
            if getattr(task, "status", None) == TaskStatus.PENDING:
                tid = str(getattr(task, "task_id", ""))
                if tid:
                    out.append(tid)
        return tuple(sorted(out))

    def _record_objective_safe_evaluation_counts(
        self,
        *,
        objective_evaluations: int,
        feasibility_evaluations: int,
    ) -> None:
        self.alns_diagnostics.objective_evaluation_count += int(
            max(objective_evaluations, 0)
        )
        self.alns_diagnostics.feasibility_evaluation_count += int(
            max(feasibility_evaluations, 0)
        )

    def _solution_agents(self, env) -> Tuple[Tuple[str, str], ...]:
        return tuple(self._agent_sequence_pairs(env))

    def _seq_dicts(self, solution: ALNSSolution) -> tuple[dict[StableId, tuple[StableId, ...]], dict[StableId, tuple[StableId, ...]]]:
        return dict(solution.truck_sequences), dict(solution.uav_sequences)

    def _with_agent_seq(self, solution: ALNSSolution, aid: str, atype: str, seq: Iterable[str]) -> ALNSSolution:
        return solution.with_sequence(str(aid), str(atype), tuple(str(x) for x in seq)[:2])

    def _assigned_tasks(self, solution: ALNSSolution) -> set[str]:
        out: set[str] = set()
        for _aid, seq in tuple(solution.truck_sequences) + tuple(solution.uav_sequences):
            out.update(str(x) for x in seq)
        return out

    def _candidate_if_feasible(self, env, solution: ALNSSolution):
        safe, evaluation, _changed = self._objective_safe_k2_solution(env, solution)
        if not bool(evaluation.feasible):
            return None
        return safe, evaluation

    def _tabu_neighbors(
        self,
        env,
        solution: ALNSSolution,
        allowed_moves: Optional[set[str]] = None,
    ):
        agents = self._solution_agents(env)
        pending = self._pending_task_ids(env)
        assigned = self._assigned_tasks(solution)
        produced = 0

        # task relocate: move an assigned task to a different feasible slot.
        for src_aid, src_type in agents:
            src_seq = tuple(str(x) for x in solution.sequence_for(src_aid, src_type))
            for src_pos, task_id in enumerate(src_seq):
                src_after = tuple(x for i, x in enumerate(src_seq) if i != src_pos)
                base = self._with_agent_seq(solution, src_aid, src_type, src_after)
                for dst_aid, dst_type in agents:
                    dst_seq = tuple(str(x) for x in base.sequence_for(dst_aid, dst_type))
                    if len(dst_seq) >= 2:
                        continue
                    for dst_pos in range(len(dst_seq) + 1):
                        new_seq = tuple(dst_seq[:dst_pos] + (task_id,) + dst_seq[dst_pos:])
                        if len(set(new_seq)) != len(new_seq):
                            continue
                        trial = self._with_agent_seq(base, dst_aid, dst_type, new_seq)
                        cand = self._candidate_if_feasible(env, trial)
                        if cand is not None and (
                            allowed_moves is None or "task_relocate" in allowed_moves
                        ):
                            yield "task_relocate", (task_id, src_aid, dst_aid), cand[0], cand[1]
                            produced += 1
                            if produced >= self.tabu_candidate_limit:
                                return

        # task swap: exchange tasks between two sequence positions.
        pairs = [(aid, atype, tuple(str(x) for x in solution.sequence_for(aid, atype))) for aid, atype in agents]
        for i, (aid_a, type_a, seq_a) in enumerate(pairs):
            for aid_b, type_b, seq_b in pairs[i + 1 :]:
                for pos_a, task_a in enumerate(seq_a):
                    for pos_b, task_b in enumerate(seq_b):
                        new_a = tuple(task_b if j == pos_a else x for j, x in enumerate(seq_a))
                        new_b = tuple(task_a if j == pos_b else x for j, x in enumerate(seq_b))
                        trial = self._with_agent_seq(self._with_agent_seq(solution, aid_a, type_a, new_a), aid_b, type_b, new_b)
                        cand = self._candidate_if_feasible(env, trial)
                        if cand is not None and (
                            allowed_moves is None or "task_swap" in allowed_moves
                        ):
                            yield "task_swap", (task_a, task_b), cand[0], cand[1]
                            produced += 1
                            if produced >= self.tabu_candidate_limit:
                                return

        # tail replacement: replace a tail with an unassigned pending task.
        for aid, atype in agents:
            seq = tuple(str(x) for x in solution.sequence_for(aid, atype))
            if len(seq) < 1:
                continue
            for task_id in pending:
                if task_id in assigned and (len(seq) < 2 or task_id != seq[-1]):
                    continue
                new_seq = (seq[0], task_id) if task_id != seq[0] else (seq[0],)
                if new_seq == seq:
                    continue
                self.alns_diagnostics.feasibility_evaluation_count += 1
                if not evaluate_sequence_feasibility(env, aid, new_seq).feasible:
                    continue
                trial = self._with_agent_seq(solution, aid, atype, new_seq)
                cand = self._candidate_if_feasible(env, trial)
                if cand is not None and (
                    allowed_moves is None or "tail_replacement" in allowed_moves
                ):
                    yield "tail_replacement", (aid, task_id), cand[0], cand[1]
                    produced += 1
                    if produced >= self.tabu_candidate_limit:
                        return

        # agent reassignment: assign an unassigned task to an agent with capacity.
        unassigned = [tid for tid in pending if tid not in assigned]
        for task_id in unassigned:
            for aid, atype in agents:
                seq = tuple(str(x) for x in solution.sequence_for(aid, atype))
                if len(seq) >= 2:
                    continue
                new_seq = tuple(seq + (task_id,))
                self.alns_diagnostics.feasibility_evaluation_count += 1
                if not evaluate_sequence_feasibility(env, aid, new_seq).feasible:
                    continue
                trial = self._with_agent_seq(solution, aid, atype, new_seq)
                cand = self._candidate_if_feasible(env, trial)
                if cand is not None and (
                    allowed_moves is None or "agent_reassignment" in allowed_moves
                ):
                    yield "agent_reassignment", (task_id, aid), cand[0], cand[1]
                    produced += 1
                    if produced >= self.tabu_candidate_limit:
                        return

    def _alns_optimize_goals(self, env, base_goals: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        start_time = time.perf_counter()
        self.alns_diagnostics.replan_count += 1
        current_goals = self._repair_goals(env, dict(base_goals))
        current_solution = construct_k2_solution(env, current_goals)
        current_solution, current_eval, _ = self._objective_safe_k2_solution(env, current_solution)
        current_goals = {str(k): (None if v is None else str(v)) for k, v in current_goals.items()}
        best_solution = current_solution
        best_eval = current_eval
        tabu_until: dict[tuple[str, tuple[str, ...]], int] = {}

        for it in range(int(self.alns_iterations)):
            self.alns_iteration_count_total += 1
            self.alns_diagnostics.iteration_count += 1
            best_neighbor = None
            best_neighbor_eval = None
            best_move_key = None
            attempted = 0
            feasible = 0
            for move_name, attrs, candidate_solution, candidate_eval in self._tabu_neighbors(env, current_solution):
                attempted += 1
                move_key = (str(move_name), tuple(str(x) for x in attrs))
                is_tabu = int(tabu_until.get(move_key, -1)) > it
                aspiration = float(candidate_eval.breakdown.total_cost) < float(best_eval.breakdown.total_cost) - 1e-12
                if is_tabu and not aspiration:
                    continue
                feasible += 1
                if best_neighbor_eval is None or float(candidate_eval.breakdown.total_cost) < float(best_neighbor_eval.breakdown.total_cost):
                    best_neighbor = candidate_solution
                    best_neighbor_eval = candidate_eval
                    best_move_key = move_key
            self.alns_diagnostics.repair_attempt_count += int(max(attempted, 1))
            self.alns_diagnostics.repair_feasible_count += int(feasible)
            if best_neighbor is None or best_neighbor_eval is None or best_move_key is None:
                self.alns_diagnostics.noop_iteration_count += 1
                continue
            self.alns_diagnostics.feasible_nonidentical_candidate_count += int(
                str(best_neighbor.digest()) != str(current_solution.digest())
            )
            delta = float(minimization_delta(float(current_eval.breakdown.total_cost), float(best_neighbor_eval.breakdown.total_cost)))
            current_solution = best_neighbor
            current_eval = best_neighbor_eval
            tabu_until[best_move_key] = int(it + self.tabu_tenure)
            self.alns_accepted_count_total += 1
            self.alns_diagnostics.accepted_count += 1
            if delta <= 0.0:
                self.alns_diagnostics.accepted_improving_count += 1
            else:
                self.alns_diagnostics.accepted_worsening_count += 1
            if float(current_eval.breakdown.total_cost) < float(best_eval.breakdown.total_cost) - 1e-12:
                best_solution = current_solution
                best_eval = current_eval
                self.alns_improvement_count_total += 1
                self.alns_diagnostics.improvement_count += 1

        self.alns_diagnostics.best_objective_gain = float(
            max(float(current_eval.breakdown.total_cost) - float(best_eval.breakdown.total_cost), 0.0)
        )
        self.alns_diagnostics.wall_clock_time_s += float(time.perf_counter() - start_time)
        return {str(k): (None if v is None else str(v)) for k, v in self._restore_protected_goals(env, current_goals, solution_to_goals(best_solution)).items()}


def solution_to_goals(solution: ALNSSolution) -> Dict[str, Optional[str]]:
    goals: Dict[str, Optional[str]] = {}
    for aid, seq in tuple(solution.truck_sequences) + tuple(solution.uav_sequences):
        goals[str(aid)] = None if not seq else str(seq[0])
    return dict(sorted(goals.items(), key=lambda kv: str(kv[0])))
