"""
Representation of a thermostat controller and two rooms. The controller can only heat one room at a
time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Tuple, TypeVar

from typing_extensions import Self

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
    bounds: tuple[float, float]

    def step_heat(self, step_size: float, temp: float) -> float:
        deriv = self.heat - self.cool * temp
        temp_ = temp + step_size * deriv
        return temp_ + self.bias

    def step_cool(self, step_size: float, temp: float) -> float:
        deriv = self.cool * temp
        temp_ = temp - step_size * deriv
        return temp_ + self.bias


@dataclass()
class SystemParameters:
    """Parameters representing the heating and cooling dynamics of the entire system.

    Properties:
        rooms: The parameters of each room
    """

    room1: RoomParameters
    room2: RoomParameters
    room3: RoomParameters
    room4: RoomParameters


class State(ABC):
    """Abstract representation of a 2-room thermostat controller state.

    Properties:
        room1: The temperature of room 1
        room2: The temperature of room 2
        room3: The temperature of room 2
        room4: The temperature of room 2
    """

    room1: float
    room2: float
    room3: float
    room4: float

    def __init__(self, room1: float, room2: float, room3: float, room4: float):
        self.room1 = room1
        self.room2 = room2
        self.room3 = room3
        self.room4 = room4

    def _stringify(self, name: str) -> str:
        return f"{name}(room1={self.room1}, room2={self.room2}, room3={self.room3}, room4={self.room4})"

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

    @classmethod
    def from_state(cls, state: State) -> Self:
        if isinstance(state, cls):
            return state

        return cls(state.room1, state.room2, state.room3, state.room4)


class Heating1(State):
    """Thermostat controller state that represents heating room 1 and cooling room 2."""

    def __str__(self) -> str:
        return self._stringify("Heating1")

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> Heating1:
        return Heating1(
            params.room1.step_heat(step_size, self.room1),
            params.room2.step_cool(step_size, self.room2),
            params.room3.step_cool(step_size, self.room3),
            params.room4.step_cool(step_size, self.room4),
        )


class Heating2(State):
    """Thermostat controller state that represents cooling room 1 and heating room 2."""

    def __str__(self) -> str:
        return self._stringify("Heating2")

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> Heating2:
        return Heating2(
            params.room1.step_cool(step_size, self.room1),
            params.room2.step_heat(step_size, self.room2),
            params.room3.step_cool(step_size, self.room3),
            params.room4.step_cool(step_size, self.room4),
        )


class Heating3(State):
    """Thermostat controller state representing cooling rooms 1 and 2."""

    def __str__(self) -> str:
        return self._stringify("Heating3")

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> Heating3:
        return Heating3(
            params.room1.step_cool(step_size, self.room1),
            params.room2.step_cool(step_size, self.room2),
            params.room3.step_heat(step_size, self.room3),
            params.room4.step_cool(step_size, self.room4),
        )


class Heating4(State):
    """Thermostat controller state representing cooling rooms 1 and 2."""

    def __str__(self) -> str:
        return self._stringify("Heating4")

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> Heating3:
        return Heating3(
            params.room1.step_cool(step_size, self.room1),
            params.room2.step_cool(step_size, self.room2),
            params.room3.step_cool(step_size, self.room3),
            params.room4.step_heat(step_size, self.room4),
        )


class Cooling(State):
    """Thermostat controller state representing cooling rooms 1 and 2."""

    def __str__(self) -> str:
        return self._stringify("Cooling")

    def __repr__(self) -> str:
        return str(self)

    def step(self, step_size: float, params: SystemParameters) -> Heating3:
        return Heating3(
            params.room1.step_cool(step_size, self.room1),
            params.room2.step_cool(step_size, self.room2),
            params.room3.step_cool(step_size, self.room3),
            params.room4.step_cool(step_size, self.room4),
        )


def controller_2rooms(state: State, step_size: float, params: SystemParameters) -> State:
    """Thermostat controller that handles switching between states.

    Arguments:
        state: The current state
        step_size: The length of the simulation interval

    Returns:
        The next state
    """

    lr1, hr1 = params.room1.bounds
    lr2, hr2 = params.room2.bounds
    temp1 = state.room1
    temp2 = state.room2

    if temp1 <= lr1:
        next_state = Heating1.from_state(state)
    elif temp1 >= hr1:
        if temp2 <= lr2:
            next_state = Heating2.from_state(state)
        elif temp2 >= hr2:
            next_state = Cooling.from_state(state)
        else:
            next_state = state
    else:
        next_state = state

    next_step = next_state.step(step_size, params)

    return next_step


def controller_4rooms(state: State, step_size: float, params: SystemParameters) -> State:
    """Thermostat controller that handles switching between states.

    Arguments:
        state: The current state
        step_size: The length of the simulation interval

    Returns:
        The next state
    """

    lr1, hr1 = params.room1.bounds
    lr2, hr2 = params.room2.bounds
    lr3, hr3 = params.room3.bounds
    lr4, hr4 = params.room4.bounds

    temp1 = state.room1
    temp2 = state.room2
    temp3 = state.room3
    temp4 = state.room4

    if temp1 <= lr1:
        next_state = Heating1.from_state(state)
    elif temp1 >= hr1:
        if temp2 <= lr2:
            next_state = Heating2.from_state(state)
        elif temp2 >= hr2:
            if temp3 <= lr3:
                next_state = Heating3.from_state(state)
            elif temp3 >= hr3:
                if temp4 <= lr4:
                    next_state = Heating4.from_state(state)
                elif temp4 >= hr4:
                    next_state = Cooling.from_state(state)
                else:
                    next_state = state
            else:
                next_state = state
        else:
            next_state = state
    else:
        next_state = state

    next_step = next_state.step(step_size, params)
    return next_step


class Controller(Protocol):
    def __call__(self, state: State, step_size: float, params: SystemParameters) -> State:
        ...


def run(
    controller: Controller,
    init: State,
    t_stop: float,
    t_start: float = 0.0,
    t_step: float = 1.0,
) -> Trace[State]:
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
    rm_params = RoomParameters(heat=5.0, cool=0.1, bias=0.0, bounds=(19.0, 22.0))
    sys_params = SystemParameters(
        room1=rm_params, room2=rm_params, room3=rm_params, room4=rm_params
    )

    while time < t_stop:
        state = controller(state, t_step, sys_params)
        time = time + t_step
        trace.append((time, state))

    return trace
