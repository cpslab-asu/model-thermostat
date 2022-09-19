"""Instrumentation of the thermostat controller."""

from dataclasses import dataclass
from typing import Sequence, TypedDict

import numpy
from bsa import BranchTree, instrument_function
from numpy.typing import NDArray
from staliro.core import Interval, Trace
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult
from staliro.core.optimizer import ObjectiveFn, Optimizer
from staliro.core.specification import Specification
from staliro.options import Options
from staliro.staliro import staliro
from taliro import tptaliro
from typing_extensions import TypeAlias

from thermostat import CoolingCooling, RoomParameters, State, SystemParameters, controller


@dataclass
class InstrumentedOutput:
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
        rm_params = RoomParameters(heat=5.0, cool=0.1, bias=0.0)
        sys_params = SystemParameters([rm_params, rm_params], [(19.0, 22.0), (19.0, 22.0)])
        timed_states = []

        while time < tspan.upper:
            variables, state = self.instr_fn(state, t_step, sys_params)
            time = time + t_step
            timed_states.append((time, InstrumentedOutput(variables, state)))

        times = [time for time, _ in timed_states]
        states = [state for _, state in timed_states]
        trace = Trace(times, states)

        return BasicResult(trace)


HybridDistance: TypeAlias = tuple[float, float]
_States: TypeAlias = Sequence[InstrumentedOutput]
_Edge: TypeAlias = tuple[str, str]


class _Guard(TypedDict):
    a: NDArray[numpy.float_]
    b: NDArray[numpy.float_]


class ThermostatSpecification(Specification[InstrumentedOutput, HybridDistance]):
    """Hybrid distance specification."""

    def __init__(self) -> None:
        trees = BranchTree.from_function(controller)
        variable_indices = dict(zip(trees[0].variables, range(1)))

        # TODO
        self.adj_list: dict[str, list[str]] = {}
        self.guards: dict[_Edge, _Guard] = {}

    def evaluate(self, states: _States, times: Sequence[float]) -> HybridDistance:
        # TODO
        pass


class OptimizerResult:
    pass


class UniformRandom(Optimizer[HybridDistance, OptimizerResult]):
    def optimize(
        self,
        func: ObjectiveFn[HybridDistance],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> OptimizerResult:
        # TODO

        return OptimizerResult()


options = Options(
    static_parameters=[(19.0, 22.0), (19.0, 22.0)],
    iterations=1000,
    runs=1,
)
result = staliro(ThermostatModel(), ThermostatSpecification(), UniformRandom(), options)
