from __future__ import annotations

from math import inf
from pprint import pprint
from typing import Sequence, TypeAlias

from banquo import HybridPredicate, hybrid_distance
from bsa.branches import BranchTree, Condition, Comparison
from bsa.kripke import Kripke
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


def _active_state(kripke: Kripke[Condition], variables: dict[str, float]) -> str:
    matching_states = [
        state for state in kripke.states 
        if all(label.is_true(variables) for label in kripke.labels_for(state))
    ]

    assert len(matching_states) == 1, f"More than one state active given variables {variables}"

    return str(matching_states[0])


def _condition_into_str(cond: Condition) -> str:
    if cond.comparison == Comparison.LTE:
        return f"{cond.variable} <= {cond.bound}"
    elif cond.comparison == Comparison.GTE:
        return f"{cond.bound} <= {cond.variable}"
    else:
        raise ValueError(f"{cond.comparison} is not a Comparison")


_States: TypeAlias = Sequence[InstrumentedOutput]
_Times: TypeAlias = Sequence[float]
_VariableMap: TypeAlias = dict[str, float]
_HybridTrace: TypeAlias = dict[float, tuple[_VariableMap, str]]

HybridDistance: TypeAlias = tuple[float, float]


class ThermostatSpecification(Specification[InstrumentedOutput, HybridDistance]):
    """Hybrid distance specification."""

    @property
    def failure_cost(self) -> HybridDistance:
        return (-inf, -inf)

    def __init__(self) -> None:
        trees = BranchTree.from_function(controller)

        assert len(trees) == 1

        tree = trees[0]
        kripkes = tree.as_kripke()

        assert len(kripkes) == 1

        self.kripke = kripkes[0]
        self.predicates = {
            f"s{i}": HybridPredicate(None, str(state)) for i, state in enumerate(self.kripke.states)
        }
        self.guards = {
            (str(s1), str(s2)): [_condition_into_str(c) for c in self.kripke.labels_for(s2)]
            for s1 in self.kripke.states
            for s2 in self.kripke.states_from(s1)
        }

        predicate_names = list(self.predicates.keys())
        self.formula = _coverage_requirement(predicate_names)

        print(self.formula)
        pprint(self.guards)

    def evaluate(self, state: _States, timestamps: _Times) -> HybridDistance:
        trace: _HybridTrace = {
            time: (output.variables, _active_state(self.kripke, output.variables))
            for time, output in zip(timestamps, state)
        }

        return hybrid_distance(self.formula, self.predicates, self.guards, trace)

