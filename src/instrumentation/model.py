from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bsa import instrument_function
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace

from thermostat import Controller, Cooling, RoomParameters, State, SystemParameters


@dataclass
class InstrumentedOutput:
    """Output of instrumented controller."""

    variables: dict[str, float]
    state: State


_ModelResult = ModelResult[InstrumentedOutput, None]


class ThermostatModel(Model[InstrumentedOutput, None], ABC):
    """Thermostat controller model with 2 rooms.

    Simulate the thermostat model over the interval tspan. Create a trace of instrumented outputs
    containing both the state of the system and the conditional variables used during execution.
    """

    def __init__(self, controller: Controller) -> None:
        self.instr_fn = instrument_function(controller)

    @abstractmethod
    def setup(self, inputs: ModelInputs) -> tuple[State, SystemParameters]:
        raise NotImplementedError()

    def simulate(self, inputs: ModelInputs, interval: Interval) -> _ModelResult:
        t_step = 0.1
        time = interval.lower
        state, sys_params = self.setup(inputs)
        timed_states: list[tuple[float, InstrumentedOutput]] = []

        while time < interval.upper:
            variables, state = self.instr_fn(state, t_step, sys_params)
            time = time + t_step
            timed_states.append((time, InstrumentedOutput(variables, state)))

        times = [time for time, _ in timed_states]
        states = [state for _, state in timed_states]
        trace = Trace(times, states)

        return BasicResult(trace)


class Thermostat2Rooms(ThermostatModel):
    """Thermostat controller model with 2 rooms.

    Simulate the thermostat model over the interval tspan. Create a trace of instrumented outputs
    containing both the state of the system and the conditional variables used during execution.
    """

    def setup(self, inputs: ModelInputs) -> tuple[State, SystemParameters]:
        state = Cooling(room1=inputs.static[0], room2=inputs.static[1], room3=0.0, room4=0.0)
        bounds = (19.0, 22.0)
        params = SystemParameters(
            room1=RoomParameters(heat=5.0, cool=inputs.static[2], bias=0.0, bounds=bounds),
            room2=RoomParameters(heat=5.0, cool=inputs.static[3], bias=0.0, bounds=bounds),
            room3=RoomParameters(heat=0.0, cool=0.0, bias=0.0, bounds=bounds),
            room4=RoomParameters(heat=0.0, cool=0.0, bias=0.0, bounds=bounds),
        )

        return state, params


class Thermostat4Rooms(ThermostatModel):
    """Thermostat controller model with 2 rooms.

    Simulate the thermostat model over the interval tspan. Create a trace of instrumented outputs
    containing both the state of the system and the conditional variables used during execution.
    """

    def setup(self, inputs: ModelInputs) -> tuple[State, SystemParameters]:
        state = Cooling(
            room1=inputs.static[0],
            room2=inputs.static[1],
            room3=inputs.static[2],
            room4=inputs.static[3],
        )
        bounds = (19.0, 22.0)
        params = SystemParameters(
            room1=RoomParameters(heat=5.0, cool=inputs.static[4], bias=0.0, bounds=bounds),
            room2=RoomParameters(heat=5.0, cool=inputs.static[5], bias=0.0, bounds=bounds),
            room3=RoomParameters(heat=0.0, cool=inputs.static[6], bias=0.0, bounds=bounds),
            room4=RoomParameters(heat=0.0, cool=inputs.static[7], bias=0.0, bounds=bounds),
        )

        return state, params
