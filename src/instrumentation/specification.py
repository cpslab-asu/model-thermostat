from __future__ import annotations

import logging
from dataclasses import dataclass
from math import inf
from typing import Sequence, TypeAlias

from banquo import HybridPredicate, hybrid_distance, robustness  # pylint: disable=no-name-in-module
from bsa.branches import BranchTree, Comparison, Condition
from bsa.kripke import Kripke, State
from staliro.core.specification import Specification

from thermostat import Controller

from .model import InstrumentedOutput


def _coverage_requirement(predicate_names: Sequence[str]) -> str:
    """Generate a coverage formula given a set of predicate names.

    This function requires that the set of predicate names is non-empty. Each predicate represents
    reaching a state of the system. The generated formula has the form
    "<> p1 and <> p2 and ... <> pn" which computes the minimum distance from every system state
    of the system represented in the formula.

    Args:
        predicate_names: The names of the predicates to put into the formula

    Returns:
        formula: A formula that computes the minimum distance to any state in the formula

    Raises:
        ValueError: If the set of predicate names is empty
    """
    if len(predicate_names) == 0:
        raise ValueError("Cannot construct coverage formula with no predicates")

    predicate_name = predicate_names[0]

    if len(predicate_names) == 1:
        return f"<> {predicate_name}"

    left_subformula = _coverage_requirement(predicate_names[1:])
    return rf"(<> {predicate_name}) /\ ({left_subformula})"


def active_state(kripke: Kripke[Condition], variables: dict[str, float]) -> str:
    """Compute the active Kripke state from a set of variables.

    A Kripke state is active if all of its boolean labels are true. To find the active state, all
    states of the Kripke structure are analyzed - moving onto the next state when the first false
    label is found. For a Kripke structure from an function we assume that only one world can be
    active at a time since only one condition can be true at a time.

    Args:
        kripke: The Kripke structure to analyzed
        variables: The set of variables to use to evaluate the state labels

    Returns:
        name: The name of the state that is active

    Raises:
        AssertionError: If there is more than one state where all labels are true
    """

    matching_states = [
        state
        for state in kripke.states
        if all(label.is_true(variables) for label in kripke.labels_for(state))
    ]

    assert len(matching_states) == 1, f"More than one state active given variables {variables}"
    return str(matching_states[0])


def _condition_into_str(cond: Condition) -> str:
    """Convert a Condition into a formula.

    Args:
        cond: The condition to convert

    Returns:
        formula: A formula representing the condition

    Raises:
        ValueError: If the cond value is not a Comparison
    """

    if cond.comparison == Comparison.LTE:
        return f"{cond.variable} <= {cond.bound}"

    if cond.comparison == Comparison.GTE:
        return f"{cond.bound} <= {cond.variable}"

    raise ValueError(f"{cond.comparison} is not a Comparison")


def _edge_guards(kripke: Kripke[Condition], start: State, end: State) -> list[str]:
    """Create a list of formulas representing a Kripke transition.

    To compute the list of formulas, this function finds the intersection of the labels for start
    state and the end state. This is an implementation detail that is necessary because variable
    values are only saved inside a conditional block so it is possible to have transition
    conditions that reference non-existent values.

    Args:
        kripke: The kripke structure
        start: The state to start from
        end: The state to end at

    Returns:
        formulas: A list of formulas representing the necessary conditions to transition from start
                  to end
    """
    start_variables = set(v for l in kripke.labels_for(start) for v in l.variables)
    valid_guards = filter(lambda l: l.variables <= start_variables, kripke.labels_for(end))

    return [_condition_into_str(g) for g in valid_guards]


_States: TypeAlias = Sequence[InstrumentedOutput]
_Times: TypeAlias = Sequence[float]
_VariableMap: TypeAlias = dict[str, float]
_HybridTrace: TypeAlias = dict[float, tuple[_VariableMap, str]]


@dataclass(frozen=True)
class CoverageSafety:
    """Specification output type

    Attributes:
        remaining_states: Number of function states that have not been covered in an execution
        hybrid_distance: Minimum distance to transition to an unvisited state
    """

    remaining_states: int
    safety: float
    coverage: float


class ThermostatSpecification(Specification[InstrumentedOutput, CoverageSafety]):
    """Hybrid distance specification.

    This specification maintains a set of unvisited states of the instrumented function. When a
    trace is evaluated, it is scanned to determine which states where visited during execution
    and the visited states are removed from the unvisited states. It is important to note that
    the states referenced here are the *function* states, not the states of the system the function
    controls.

    After shrinking the unvisited set of states, a hybrid distance formula is generated from the
    unvisited set. This formula will compute the shortest distance to for the function to visit
    one of the uncovered states. This distance is returned as a cost value, along with an integer
    representing the size of the unvisited set to communicate to the optimizer when full coverage
    has beed achieved and it should stop generating samples.

    Attributes:
        kripke: The kripke structure generated from the input function
        uncovered_states: The set of states that have not been visited in any execution
        guards: Edges and guards representing the transitions of the function kripke system
    """

    @property
    def failure_cost(self) -> CoverageSafety:
        return CoverageSafety(0, -inf, -inf)

    def __init__(self, controller: Controller) -> None:
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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Controller states to cover: {len(self.uncovered_states)}")

    def evaluate(self, state: _States, timestamps: _Times) -> CoverageSafety:
        trace: _HybridTrace = {
            time: (output.variables, active_state(self.kripke, output.variables))
            for time, output in zip(timestamps, state)
        }

        covered_states = set(state for _, state in trace.values())
        n_prev_uncovered = len(self.uncovered_states)
        self.uncovered_states -= covered_states
        n_uncovered = len(self.uncovered_states)

        if n_uncovered < n_prev_uncovered:
            self.logger.debug(f"Controller states remaining: {n_uncovered}")

        predicates = {
            f"s{i}": HybridPredicate(None, state) for i, state in enumerate(self.uncovered_states)
        }

        if len(predicates) > 0:
            formula = _coverage_requirement(list(predicates.keys()))
            safety = 1.0
            _, coverage = hybrid_distance(formula, predicates, self.guards, trace)
        else:
            safety = 0.0
            coverage = inf

        return CoverageSafety(len(self.uncovered_states), safety, coverage)


def _state_dict(state: InstrumentedOutput) -> dict[str, float]:
    return {
        "room1": state.state.room1,
        "room2": state.state.room2,
        "room3": state.state.room3,
        "room4": state.state.room4,
    }


class ThermostatRequirement(Specification[InstrumentedOutput, CoverageSafety]):
    def __init__(self, formula: str, controller: Controller):
        self.formula = formula
        self.spec = ThermostatSpecification(controller)

    @property
    def failure_cost(self) -> CoverageSafety:
        return self.spec.failure_cost

    @property
    def kripke(self) -> Kripke[Condition]:
        return self.spec.kripke

    def evaluate(self, state: _States, timestamps: _Times) -> CoverageSafety:
        result = self.spec.evaluate(state, timestamps)
        trace = {time: _state_dict(s) for (time, s) in zip(timestamps, state)}
        safety = robustness(self.formula, trace)

        return CoverageSafety(result.remaining_states, safety, result.coverage)
