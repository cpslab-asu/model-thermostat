from __future__ import annotations

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


def _guard_from_conditions(conditions: Sequence[Condition], indices: dict[str, int]) -> _Guard:
    pass


class ThermostatSpecification(Specification[InstrumentedOutput, HybridDistance]):
    """Hybrid distance specification."""

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

    def evaluate(self, states: _States, times: Sequence[float]) -> HybridDistance:
        states_ = [(output.state.temp1, output.state.temp2) for output in states]
        locations: list[float] = []
        get_distances = itemgetter("ds", "dl")
        robustness = self.spec.hybrid(states_, times, locations, self.adj_list, self.guards)

        return get_distances(robustness)
