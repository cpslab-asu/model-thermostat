"""
Representation of a thermostat controller and two rooms. The controller can only heat one room at a
time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, TypeVar

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

    room1: RoomParameters
    room2: RoomParameters
    room3: RoomParameters
    room4: RoomParameters


class StateDict(TypedDict):
    t1: float
    t2: float
    t3: float
    t4: float


class State(ABC):
    """Abstract representation of a 2-room thermostat controller state.

    Properties:
        temp1: The temperature of room 1
        temp2: The temperature of room 2
    """

    def __init__(self, t1: float, t2: float, t3: float, t4: float):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4

    def to_dict(self) -> StateDict:
        """Convert state to dictionary."""
        return StateDict(t1=self.t1, t2=self.t2, t3=self.t3, t4=self.t4)

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


class Heating1(State):
    """Thermostat controller state that represents heating room 1 and cooling room 2."""

    def __str__(self) -> str:
        return f"Heating(room=1, t1={self.t1}, t2={self.t2}, t3={self.t3}, t4={self.t4})"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, step_size: float, params: SystemParameters) -> Heating1:
        return Heating1(
            _heat(params.room1, step_size, self.t1),
            _cool(params.room2, step_size, self.t2),
            _cool(params.room3, step_size, self.t3),
            _cool(params.room4, step_size, self.t4),
        )


class Heating2(State):
    """Thermostat controller state that represents cooling room 1 and heating room 2."""

    def __str__(self) -> str:
        return f"Heating(room=2, t1={self.t1}, t2={self.t2}, t3={self.t3}, t4={self.t4})"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, step_size: float, params: SystemParameters) -> Heating2:
        return Heating2(
            _cool(params.room1, step_size, self.t1),
            _heat(params.room2, step_size, self.t2),
            _cool(params.room3, step_size, self.t3),
            _cool(params.room4, step_size, self.t4),
        )


class Heating3(State):
    """Thermostat controller state that represents cooling room 1 and heating room 2."""

    def __str__(self) -> str:
        return f"Heating(room=3, t1={self.t1}, t2={self.t2}, t3={self.t3}, t4={self.t4})"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, step_size: float, params: SystemParameters) -> Heating3:
        return Heating3(
            _cool(params.room1, step_size, self.t1),
            _cool(params.room2, step_size, self.t2),
            _heat(params.room3, step_size, self.t3),
            _cool(params.room4, step_size, self.t4),
        )


class Heating4(State):
    """Thermostat controller state that represents cooling room 1 and heating room 2."""

    def __str__(self) -> str:
        return f"Heating(room=4, t1={self.t1}, t2={self.t2}, t3={self.t3}, t4={self.t4})"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, step_size: float, params: SystemParameters) -> Heating4:
        return Heating4(
            _cool(params.room1, step_size, self.t1),
            _cool(params.room2, step_size, self.t2),
            _cool(params.room3, step_size, self.t3),
            _heat(params.room4, step_size, self.t4),
        )


class Cooling(State):
    """Thermostat controller state representing cooling rooms 1 and 2."""

    def __str__(self) -> str:
        return f"Cooling(t1={self.t1}, t2={self.t2}, t3={self.t3}, t4={self.t4})"

    def __repr__(self) -> str:
        return self.__str__()

    def step(self, step_size: float, params: SystemParameters) -> Cooling:
        return Cooling(
            _cool(params.room1, step_size, self.t1),
            _cool(params.room2, step_size, self.t2),
            _cool(params.room3, step_size, self.t3),
            _cool(params.room4, step_size, self.t4),
        )


def _heating1(state: State) -> State:
    if isinstance(state, Heating1):
        return state

    return Heating1(state.t1, state.t2, state.t3, state.t4)


def _heating2(state: State) -> State:
    if isinstance(state, Heating2):
        return state

    return Heating2(state.t1, state.t2, state.t3, state.t4)


def _heating3(state: State) -> State:
    if isinstance(state, Heating3):
        return state

    return Heating3(state.t1, state.t2, state.t3, state.t4)


def _heating4(state: State) -> State:
    if isinstance(state, Heating4):
        return state

    return Heating4(state.t1, state.t2, state.t3, state.t4)


def _cooling(state: State) -> State:
    if isinstance(state, Cooling):
        return state

    return Cooling(state.t1, state.t2, state.t3, state.t4)


def controller(state: State, step_size: float, params: SystemParameters) -> State:
    """Thermostat controller that handles switching between states.

    Arguments:
        state: The current state
        step_size: The length of the simulation interval

    Returns:
        The next state
    """

    t1 = state.t1
    lowerr1, upperr1 = params.room1.bounds

    if t1 <= lowerr1:
        next_state = _heating1(state)
    elif t1 >= upperr1:
        t2 = state.t2
        lowerr2, upperr2 = params.room2.bounds

        if t2 <= lowerr2:
            next_state = _heating2(state)
        elif t2 >= upperr2:
            t3 = state.t3
            lowerr3, upperr3 = params.room3.bounds

            if t3 <= lowerr3:
                next_state = _heating3(state)
            elif t3 >= upperr3:
                t4 = state.t4
                lowerr4, upperr4 = params.room4.bounds

                if t4 <= lowerr4:
                    next_state = _heating4(state)
                elif t4 >= upperr4:
                    next_state = _cooling(state)
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
    rm_params = RoomParameters(heat=5.0, cool=0.1, bias=0.0, bounds=(19.0, 22.0))
    sys_params = SystemParameters(
        room1=rm_params, room2=rm_params, room3=rm_params, room4=rm_params
    )

    while time < t_stop:
        state = controller(state, t_step, sys_params)
        time = time + t_step
        trace.append((time, state))

    return trace
