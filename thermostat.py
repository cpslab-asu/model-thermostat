"""
Representation of a thermostat controller and two rooms. The controller can only heat one room at a
time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")
Trace = List[Tuple[float, T]]


@dataclass()
class RoomParameters:
    """Parameters representing the heating and cooling dynamics of a room.

    Properties:
        heat: The heating constant of the room
        cool: The cooling coefficient of the room
        bias: Constant representing an external heating/cooling effect on the room
    """

    heat: float
    cool: float
    bias: float


def _heat(params: RoomParameters, step_size: float, temp: float) -> float:
    deriv = params.heat - params.cool * temp
    temp_ = temp + step_size * deriv
    return temp_ + params.bias


def _cool(params: RoomParameters, step_size: float, temp: float) -> float:
    deriv = params.cool * temp
    temp_ = temp - step_size * deriv
    return temp_ + params.bias


@dataclass()
class SystemParameters:
    """Parameters representing the heating and cooling dynamics of the entire system.

    Properties:
        rooms: The parameters of each room
    """

    rooms: Sequence[RoomParameters]
    op_range: Sequence[Tuple[float, float]]


class State(ABC):
    """Abstract representation of a 2-room thermostat controller state.

    Properties:
        temp1: The temperature of room 1
        temp2: The temperature of room 2
    """

    temp1: float
    temp2: float

    @abstractmethod
    def step(self: T, step_size: float, params: SystemParameters) -> T:
        """Simulate the temperature of the system for a given duration.

        Arguments:
            step_size: The size of the simulation interval
            params: The heating/cooling parameters of the system

        Returns:
            A new instance of the state with updated temperatures
        """
        raise NotImplementedError()


class HeatingCooling(State):
    """Thermostat controller state that represents heating room 1 and cooling room 2."""

    def __init__(self, temp1: float, temp2: float):
        self.temp1 = temp1
        self.temp2 = temp2

    def __str__(self) -> str:
        return f"Heating(temp1={self.temp1}, temp2={self.temp2})"

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> HeatingCooling:
        return HeatingCooling(
            _heat(params.rooms[0], step_size, self.temp1),
            _cool(params.rooms[1], step_size, self.temp2),
        )


class CoolingHeating(State):
    """Thermostat controller state that represents cooling room 1 and heating room 2."""

    def __init__(self, temp1: float, temp2: float):
        self.temp1 = temp1
        self.temp2 = temp2

    def __str__(self) -> str:
        return f"Heating(temp1={self.temp1}, temp2={self.temp2})"

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> CoolingHeating:
        return CoolingHeating(
            _cool(params.rooms[0], step_size, self.temp1),
            _heat(params.rooms[1], step_size, self.temp2),
        )


class CoolingCooling(State):
    """Thermostat controller state representing cooling rooms 1 and 2."""

    def __init__(self, temp1: float, temp2: float):
        self.temp1 = temp1
        self.temp2 = temp2

    def __str__(self) -> str:
        return f"Cooling(temp1={self.temp1}, temp2={self.temp2})"

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> CoolingCooling:
        return CoolingCooling(
            _cool(params.rooms[0], step_size, self.temp1),
            _cool(params.rooms[1], step_size, self.temp2),
        )


def _heating_cooling_step(state: State) -> State:
    if isinstance(state, HeatingCooling):
        return state

    return HeatingCooling(state.temp1, state.temp2)


def _cooling_heating_step(state: State) -> State:
    if isinstance(state, CoolingHeating):
        return state

    return CoolingHeating(state.temp1, state.temp2)


def _cooling_cooling_step(state: State) -> State:
    if isinstance(state, CoolingCooling):
        return state

    return CoolingCooling(state.temp1, state.temp2)


def controller(state: State, step_size: float, params: SystemParameters) -> State:
    """Thermostat controller that handles switching between states.

    Arguments:
        state: The current state
        step_size: The length of the simulation interval

    Returns:
        The next state
    """

    room1_lower, room1_upper = params.op_range[0]
    room2_lower, room2_upper = params.op_range[1]

    if state.temp1 <= room1_lower:
        next_state = _heating_cooling_step(state)
    elif state.temp1 >= room1_upper:
        if state.temp2 <= room2_lower:
            next_state = _cooling_heating_step(state)
        elif state.temp2 >= room2_upper:
            next_state = _cooling_cooling_step(state)
        else:
            next_state = state
    else:
        next_state = state

    next_step = next_state.step(step_size, params)

    return next_step


def run(init: State, t_stop: float, t_start: float = 0.0, t_step: float = 1.0) -> Trace[State]:
    """Run the thermostat controller over the interval (t_start, t_stop) in increments of t_step.

    Arguments:
        init: The state for the controller to start in
        t_stop: The time limit for the simulation
        t_step: How long each simulation interval is

    Returns:
        A sequence of times and states representing the trajectory of the system
    """

    time = t_start
    state = init
    trace = [(time, state)]
    rm_params = RoomParameters(5.0, 0.1, 0.0)
    sys_params = SystemParameters([rm_params, rm_params], [(19.0, 22.0), (19.0, 22.0)])

    while time < t_stop:
        state = controller(state, t_step, sys_params)
        time = time + t_step
        trace.append((time, state))

    return trace
