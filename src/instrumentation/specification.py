from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence, TypeAlias

from banquo import HybridPredicate, hybrid_distance
from bsa.branches import BranchTree, Condition, Comparison
from bsa.kripke import Kripke, State
from staliro.core.specification import Specification

from thermostat import controller

from .model import InstrumentedOutput


def _coverage_requirement(predicate_names: Sequence[str]) -> str:
    if len(predicate_names) == 0:
        raise ValueError("Cannot construct coverage formula with no predicates")

    predicate_name = predicate_names[0]

    if len(predicate_names) == 1:
        return f"<> {predicate_name}"

    left_subformula = _coverage_requirement(predicate_names[1:])
    return rf"(<> {predicate_name}) /\ ({left_subformula})"


def active_state(kripke: Kripke[Condition], variables: dict[str, float]) -> str:
    matching_states = [
        state for state in kripke.states 
        if all(label.is_true(variables) for label in kripke.labels_for(state))
    ]

    assert len(matching_states) == 1, f"More than one state active given variables {variables}"
    return str(matching_states[0])


def _condition_into_str(cond: Condition) -> str:
    if cond.comparison == Comparison.LTE:
        return f"{cond.variable} <= {cond.bound}"

    if cond.comparison == Comparison.GTE:
        return f"{cond.bound} <= {cond.variable}"

    raise ValueError(f"{cond.comparison} is not a Comparison")


def _edge_guards(kripke: Kripke[Condition], start: State, end: State) -> list[str]:
    start_variables = set(v for l in kripke.labels_for(start) for v in l.variables)
    valid_guards = filter(lambda l: l.variables <= start_variables, kripke.labels_for(end))

    return [_condition_into_str(g) for g in valid_guards]


_States: TypeAlias = Sequence[InstrumentedOutput]
_Times: TypeAlias = Sequence[float]
_VariableMap: TypeAlias = dict[str, float]
_HybridTrace: TypeAlias = dict[float, tuple[_VariableMap, str]]


@dataclass(frozen=True)
class SystemCoverage:
    remaining_states: int
    hybrid_distance: tuple[float, float]


class ThermostatSpecification(Specification[InstrumentedOutput, SystemCoverage]):
    """Hybrid distance specification."""

    @property
    def failure_cost(self) -> SystemCoverage:
        return SystemCoverage(0, (-inf, -inf))

    def __init__(self) -> None:
        trees = BranchTree.from_function(controller)

        assert len(trees) == 1

        tree = trees[0]
        kripkes = tree.as_kripke()

        assert len(kripkes) == 1

        self.kripke = kripkes[0]
        self.uncovered_states = set(str(state) for state in self.kripke.states)
        self.guards = {
            (str(s1), str(s2)): _edge_guards(self.kripke, s1, s2)
            for s1 in self.kripke.states
            for s2 in self.kripke.states_from(s1)
        }

    def evaluate(self, state: _States, timestamps: _Times) -> SystemCoverage:
        trace: _HybridTrace = {
            time: (output.variables, active_state(self.kripke, output.variables))
            for time, output in zip(timestamps, state)
        }

        covered_states = set(state for _, state in trace.values())
        self.uncovered_states -= covered_states

        predicates = {
            f"s{i}": HybridPredicate(None, state) for i, state in enumerate(self.uncovered_states)
        }

        if len(predicates) > 0:
            formula = _coverage_requirement(list(predicates.keys()))
            distance = hybrid_distance(formula, predicates, self.guards, trace)
        else:
            distance = (0, inf)

        return SystemCoverage(len(self.uncovered_states), distance)
