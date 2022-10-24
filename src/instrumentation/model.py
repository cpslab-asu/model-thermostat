from __future__ import annotations

from dataclasses import dataclass

from bsa import instrument_function
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace

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
