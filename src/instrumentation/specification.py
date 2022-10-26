from __future__ import annotations

from itertools import repeat
from math import inf
from operator import itemgetter
from typing import Sequence, TypedDict

import numpy
from bsa.branches import BranchTree, Condition
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


def _row_from_condition(condition: Condition, indicies: dict[str, int]) -> tuple[list[float], float]:
    coefficients = list(repeat(0.0, len(indicies)))
    bound = 0.0

    for variable in condition.variables:
        index = indicies[variable]
        coefficients[index] = 1.0

    return coefficients, bound


def _guard_from_conditions(conditions: Sequence[Condition], indices: dict[str, int]) -> _Guard:
    rows = [_row_from_condition(condition, indices) for condition in conditions]
    guard: _Guard = {
        "a": numpy.array(row[0] for row in rows),
        "b": numpy.array(row[1] for row in rows)
    }

    return guard


class ThermostatSpecification(Specification[InstrumentedOutput, HybridDistance]):
    """Hybrid distance specification."""

    @property
    def failure_cost(self) -> HybridDistance:
        return (-inf, -inf)

    def __init__(self) -> None:
        trees = BranchTree.from_function(controller)

        assert len(trees) == 1

        variable_indices = dict(zip(trees[0].variables, range(1)))
        predicates = [
            TaliroPredicate(name="s2", A=None, b=None),  # TODO: Determine A and b values
            TaliroPredicate(name="temp", A=None, b=None),  # TODO: Determine A and b values
        ]
        kripkes = trees[0].as_kripke()

        self.kripke = kripkes[0]
        self.spec = TpTaliro("(not s2) U temp", predicates)
        self.adj_list: dict[str, list[str]] = {
            str(s1): [str(s2) for s2 in self.kripke.states_from(s1)] for s1 in self.kripke.states
        }
        self.guards: dict[tuple[str, str], _Guard] = {
            (str(s1), str(s2)): _guard_from_conditions(self.kripke.labels_for(s2), variable_indices)
            for s1 in self.kripke.states
            for s2 in self.kripke.states_from(s1)
        }

    def evaluate(self, state: _States, timestamps: Sequence[float]) -> HybridDistance:
        states_ = [(output.state.temp1, output.state.temp2) for output in state]
        locations: list[float] = []
        get_distances = itemgetter("ds", "dl")
        robustness = self.spec.hybrid(states_, timestamps, locations, self.adj_list, self.guards)

        return get_distances(robustness)

