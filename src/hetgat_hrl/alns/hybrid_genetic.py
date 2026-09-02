from __future__ import annotations

import time
from typing import Dict, Optional

from hetgat_hrl.alns.sequence import construct_k2_solution
from hetgat_hrl.alns.solution import ALNSSolution
from hetgat_hrl.alns.tabu_search import TabuSearchK2Planner, solution_to_goals


class HybridGeneticK2Planner(TabuSearchK2Planner):
    """Population-based K2 baseline with crossover and neighborhood mutation.

    The method shares the public objective and feasibility oracle with the other
    metaheuristics, but it does not use ALNS destroy/repair selection or tabu
    tenure. Local neighborhood mutation makes it a hybrid genetic search rather
    than a renamed ALNS variant.
    """

    def __init__(
        self,
        *args,
        population_size: int = 8,
        elite_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.population_size = int(max(population_size, 4))
        self.elite_size = int(max(min(elite_size, self.population_size - 1), 1))
        self._operator_pool_name = "hybrid_genetic_k2"

    def _crossover(
        self, env, parent_a: ALNSSolution, parent_b: ALNSSolution
    ) -> ALNSSolution:
        child = parent_a
        for aid, atype in self._solution_agents(env):
            if float(self.rng.random()) >= 0.5:
                continue
            seq = tuple(str(x) for x in parent_b.sequence_for(aid, atype))
            child = self._with_agent_seq(child, aid, atype, seq)

        # Deterministic duplicate repair: keep the first occurrence in stable
        # agent order. The shared feasibility oracle performs the remaining
        # role/capacity/energy repair.
        seen: set[str] = set()
        for aid, atype in self._solution_agents(env):
            seq = []
            for task_id in child.sequence_for(aid, atype):
                tid = str(task_id)
                if tid in seen:
                    continue
                seen.add(tid)
                seq.append(tid)
            child = self._with_agent_seq(child, aid, atype, tuple(seq))
        safe, _evaluation, _changed = self._objective_safe_k2_solution(env, child)
        return safe

    def _mutate(self, env, solution: ALNSSolution) -> ALNSSolution:
        # Mutation is sampled, not an exhaustive best-neighbor search. This
        # keeps the baseline genetically distinct from VNS/Tabu and prevents
        # one mutation from evaluating the full K2 neighborhood on L maps.
        neighbors = []
        for row in self._tabu_neighbors(env, solution):
            neighbors.append(row)
            if len(neighbors) >= 4:
                break
        if not neighbors:
            return solution
        idx = int(self.rng.integers(0, len(neighbors)))
        return neighbors[idx][2]

    def _evaluate_population(self, env, solutions):
        unique = {}
        for solution in solutions:
            safe, evaluation, _changed = self._objective_safe_k2_solution(
                env, solution
            )
            unique[str(safe.digest())] = (safe, evaluation)
        return sorted(
            unique.values(), key=lambda row: float(row[1].breakdown.total_cost)
        )

    def _alns_optimize_goals(
        self, env, base_goals: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        started = time.perf_counter()
        self.alns_diagnostics.replan_count += 1
        current_goals = self._repair_goals(env, dict(base_goals))
        seed_solution = construct_k2_solution(env, current_goals)
        initial = [seed_solution]
        for _name, _attrs, candidate, _evaluation in self._tabu_neighbors(
            env, seed_solution
        ):
            initial.append(candidate)
            if len(initial) >= self.population_size:
                break
        population = self._evaluate_population(env, initial)
        if not population:
            population = self._evaluate_population(env, [seed_solution])

        for _ in range(int(self.alns_iterations)):
            self.alns_iteration_count_total += 1
            self.alns_diagnostics.iteration_count += 1
            elites = population[: self.elite_size]
            offspring = [row[0] for row in elites]
            attempts = 0
            while len(offspring) < self.population_size:
                attempts += 1
                pa = population[int(self.rng.integers(0, len(population)))][0]
                pb = population[int(self.rng.integers(0, len(population)))][0]
                child = self._crossover(env, pa, pb)
                if float(self.rng.random()) < 0.75:
                    child = self._mutate(env, child)
                offspring.append(child)
            candidates = self._evaluate_population(env, offspring)
            self.alns_diagnostics.repair_attempt_count += int(max(attempts, 1))
            self.alns_diagnostics.repair_feasible_count += int(len(candidates))
            combined = self._evaluate_population(
                env, [row[0] for row in population] + [row[0] for row in candidates]
            )
            old_best = float(population[0][1].breakdown.total_cost)
            population = combined[: self.population_size]
            new_best = float(population[0][1].breakdown.total_cost)
            if new_best < old_best - 1e-12:
                self.alns_accepted_count_total += 1
                self.alns_improvement_count_total += 1
                self.alns_diagnostics.accepted_count += 1
                self.alns_diagnostics.accepted_improving_count += 1
                self.alns_diagnostics.improvement_count += 1
            else:
                self.alns_diagnostics.noop_iteration_count += 1

        best_solution = population[0][0]
        self.alns_diagnostics.wall_clock_time_s += float(
            time.perf_counter() - started
        )
        goals = solution_to_goals(best_solution)
        return {
            str(k): (None if v is None else str(v))
            for k, v in self._restore_protected_goals(
                env, current_goals, goals
            ).items()
        }
