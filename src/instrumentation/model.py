from __future__ import annotations

from dataclasses import dataclass

from bsa import instrument_function
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace

from thermostat import Cooling, RoomParameters, State, SystemParameters, controller


@dataclass
class InstrumentedOutput:
    """Output of instrumented controller."""

    variables: dict[str, float]
    state: State


_ModelResult = ModelResult[InstrumentedOutput, None]


class ThermostatModel(Model[InstrumentedOutput, None]):
    """Thermostat controller model.

    Simulate the thermostat model over the interval tspan. Create a trace of instrumented outputs
    containing both the state of the system and the conditional variables used during execution.
    """

    def __init__(self) -> None:
        self.instr_fn = instrument_function(controller)

    def simulate(self, inputs: ModelInputs, tspan: Interval) -> _ModelResult:
        t_step = 0.1
        time = tspan.lower
        state = Cooling(inputs.static[0], inputs.static[1], inputs.static[2], inputs.static[3])
        params = SystemParameters(
            room1=RoomParameters(heat=5.0, cool=inputs.static[4], bias=0.0, bounds=(19.0, 22.0)),
            room2=RoomParameters(heat=5.0, cool=inputs.static[5], bias=0.0, bounds=(19.0, 22.0)),
            room3=RoomParameters(heat=5.0, cool=inputs.static[6], bias=0.0, bounds=(19.0, 22.0)),
            room4=RoomParameters(heat=5.0, cool=inputs.static[7], bias=0.0, bounds=(19.0, 22.0)),
        )
        timed_states: list[tuple[float, InstrumentedOutput]] = []

        while time < tspan.upper:
            variables, state = self.instr_fn(state, t_step, params)
            time = time + t_step
            timed_states.append((time, InstrumentedOutput(variables, state)))

        times = [time for time, _ in timed_states]
        states = [state for _, state in timed_states]
        trace = Trace(times, states)

        return BasicResult(trace)
