"""Instrumentation of the thermostat controller."""

from dataclasses import dataclass
from operator import itemgetter
from typing import Sequence, TypedDict

import numpy
from bsa import BranchTree, instrument_function
from bsa.branches import Condition
from numpy.typing import NDArray
from staliro.core import Interval, Sample, Trace
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult
from staliro.core.optimizer import ObjectiveFn, Optimizer
from staliro.core.specification import Specification
from staliro.options import Options
from staliro.specifications import TaliroPredicate, TpTaliro
from staliro.staliro import staliro
from typing_extensions import TypeAlias

from thermostat import CoolingCooling, RoomParameters, State, SystemParameters, controller


@dataclass
class InstrumentedOutput:
    """Output of instrumented controller."""

    variables: dict[str, float]
    state: State


_ModelResult = ModelResult[InstrumentedOutput, None]


class ThermostatModel(Model[InstrumentedOutput, None]):
    """Thermostat controller model."""

    def __init__(self) -> None:
        self.instr_fn = instrument_function(controller)

    def simulate(self, inputs: ModelInputs, tspan: Interval) -> _ModelResult:
        t_step = 0.1
        time = tspan.lower
        state = CoolingCooling(inputs.static[0], inputs.static[1])
        rm1_params = RoomParameters(heat=5.0, cool=inputs.static[2], bias=0.0)
        rm2_params = RoomParameters(heat=5.0, cool=inputs.static[3], bias=0.0)
        sys_params = SystemParameters([rm1_params, rm2_params], [(19.0, 22.0), (19.0, 22.0)])
        timed_states = []

        while time < tspan.upper:
            variables, state = self.instr_fn(state, t_step, sys_params)
            time = time + t_step
            timed_states.append((time, InstrumentedOutput(variables, state)))

        times = [time for time, _ in timed_states]
        states = [state for _, state in timed_states]
        trace = Trace(times, states)

        return BasicResult(trace)


_Cost: TypeAlias = tuple[float, float]
_States: TypeAlias = Sequence[InstrumentedOutput]


class _Guard(TypedDict):
    a: NDArray[numpy.float_]
    b: NDArray[numpy.float_]


def _guard_from_conditions(conditions: Sequence[Condition], indices: dict[str, int]) -> _Guard:
    pass


class ThermostatSpecification(Specification[InstrumentedOutput, _Cost]):
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

    def evaluate(self, states: _States, times: Sequence[float]) -> _Cost:
        states_ = [(output.state.temp1, output.state.temp2) for output in states]
        locations: list[float] = []
        get_distances = itemgetter("ds", "dl")
        robustness = self.spec.hybrid(states_, times, locations, self.adj_list, self.guards)

        return get_distances(robustness)


class UniformRandom(Optimizer[_Cost, None]):
    """Uniform random optimizer specialized to consume hybrid distance cost values."""

    def optimize(
        self,
        func: ObjectiveFn[_Cost],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> None:
        def _randinterval(rng: numpy.random.Generator, interval: Interval) -> float:
            return interval.lower + rng.random() * interval.length

        def _randsample(rng: numpy.random.Generator, intervals: Sequence[Interval]) -> Sample:
            return Sample([_randinterval(rng, interval) for interval in intervals])

        rng = numpy.random.default_rng(seed)
        samples = [_randsample(rng, bounds) for _ in range(budget)]

        for sample in samples:
            _ = func.eval_sample(sample)


options = Options(
    static_parameters=[(19.0, 22.0), (19.0, 22.0), (0.0, 1.0), (0.0, 1.0)],
    iterations=1000,
    runs=1,
)
result = staliro(ThermostatModel(), ThermostatSpecification(), UniformRandom(), options)
