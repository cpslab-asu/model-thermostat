from __future__ import annotations

from itertools import count
from math import inf
from operator import itemgetter
from pprint import pprint
from typing import Sequence, TypedDict

import numpy
from bsa.branches import BranchTree, Condition, Comparison
from bsa.kripke import Kripke, State
from numpy.typing import NDArray
from staliro.core.specification import Specification
from staliro.specifications import TaliroPredicate, TpTaliro
from typing_extensions import TypeAlias

from thermostat import controller

from .model import InstrumentedOutput

HybridDistance: TypeAlias = tuple[float, float]
_States: TypeAlias = Sequence[InstrumentedOutput]


class _Guard(TypedDict):
    a: NDArray[numpy.float_]
    b: NDArray[numpy.float_]


def _row_from_condition(condition: Condition, indices: dict[str, int]) -> tuple[list[float], float]:
    coefficients = [0.0 for _ in indices]
    bound = 0.0
    cmp = condition.comparison
    var_index = indices[condition.variable]

    if isinstance(condition.bound, int):
        if cmp is Comparison.LTE:
            coefficients[var_index] = 1.0
            bound = condition.bound
        elif cmp is Comparison.GTE:
            coefficients[var_index] = -1.0
            bound = -condition.bound

    if isinstance(condition.bound, str):
        bnd_index = indices[condition.bound]

        if cmp is Comparison.LTE:
            coefficients[var_index] = 1.0
            coefficients[bnd_index] = -1.0
        elif cmp is Comparison.GTE:
            coefficients[var_index] = -1.0
            coefficients[bnd_index] = 1.0

    return coefficients, bound


def _guard_from_conditions(conditions: Sequence[Condition], indices: dict[str, int]) -> _Guard:
    rows = [_row_from_condition(condition, indices) for condition in conditions]
    guard: _Guard = {
        "a": numpy.array([row[0] for row in rows]),
        "b": numpy.array([row[1] for row in rows]),
    }

    return guard


def _active_state(kripke: Kripke[Condition], variables: dict[str, float]) -> State:
    matching_states = [
        state for state in kripke.states 
        if all(label.is_true(variables) for label in kripke.labels_for(state))
    ]

    assert len(matching_states) == 1, f"More than one state active given variables {variables}"

    return matching_states[0]


def _location(kripke: Kripke[Condition], variables: dict[str, float], state_map: dict[State, int]) -> float:
    state = _active_state(kripke, variables)
    location = state_map[state]

    return float(location)


def _map_variables(variables: dict[str, float], index_map: dict[str, int]) -> list[float]:
    assert all(name in index_map for name in variables), "Missing variable name in index map"

    row = [0.0] * len(index_map)

    for var_name, var_index in index_map.items():
        try:
            var_value = variables[var_name]
        except KeyError:
            continue
        else:
            row[var_index] = var_value

    return row


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

        self.kripke = kripkes[0]
        self.state_index_map: dict[State, int] = dict(zip(self.kripke.states, count(start=1)))

        predicates = [
            TaliroPredicate(
                name=f"s{i}",
                A=numpy.array([0] * len(tree.variables), ndmin=2, dtype=numpy.double),
                b=numpy.array([0], ndmin=2, dtype=numpy.double),
                l=numpy.array([i], ndmin=2, dtype=numpy.double),
            )
            for i in self.state_index_map.values()
        ]

        self.spec = TpTaliro(r"<> s1 /\ <> s2 /\ <> s3 /\ <> s4 /\ <> s5", predicates)
        self.adj_list: dict[str, list[str]] = {
            str(s1): [str(s2) for s2 in self.kripke.states_from(s1)]
            for s1 in self.state_index_map.keys()
        }
        self.variable_index_map: dict[str, int] = dict(zip(tree.variables, count(start=0)))
        self.guards: dict[tuple[str, str], _Guard] = {
            (str(s1), str(s2)): _guard_from_conditions(self.kripke.labels_for(s2), self.variable_index_map)
            for s1 in self.kripke.states
            for s2 in self.kripke.states_from(s1)
        }

        pprint(self.variable_index_map)
        pprint(self.guards)

        raise RuntimeError()

    def evaluate(self, state: _States, timestamps: Sequence[float]) -> HybridDistance:
        states_: list[list[float]] = [
            _map_variables(output.variables, self.variable_index_map) for output in state
        ]

        locations: list[float] = [
            _location(self.kripke, output.variables, self.state_index_map) for output in state
        ]

        get_distances = itemgetter("ds", "dl")
        robustness = self.spec.hybrid(states_, timestamps, locations, self.adj_list, self.guards)

        return get_distances(robustness)

