from staliro.options import Options
from staliro.staliro import staliro

from .model import ThermostatModel
from .optimizer import UniformRandom
from .specification import ThermostatSpecification

options = Options(
    static_parameters=[(19.0, 22.0), (19.0, 22.0), (0.0, 1.0), (0.0, 1.0)],
    iterations=1000,
    runs=1,
)
result = staliro(ThermostatModel(), ThermostatSpecification(), UniformRandom(), options)
