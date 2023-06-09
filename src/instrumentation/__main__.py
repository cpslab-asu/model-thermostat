from staliro.options import Options
from staliro.staliro import staliro

from .model import ThermostatModel
from .optimizer import SOAR, UniformRandom
from .specification import ThermostatSpecification

spec = ThermostatSpecification()
model = ThermostatModel()
options = Options(
    static_parameters=[
        (19.0, 22.0),
        (19.0, 22.0),
        (19.0, 22.0),
        (19.0, 22.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ],
    iterations=1000,
    runs=1,
)

result_ur = staliro(model, ThermostatSpecification(), UniformRandom(), options)
run_ur = result_ur.runs[0]


result_soar = staliro(model, ThermostatSpecification(), SOAR(), options)
run_soar = result_soar.runs[0]

print(f"Function branches: {len(spec.kripke.states)}")
print(f"UR coverage achieved in: {len(run_ur.history)}")
print(f"SOAR coverage achieved in: {len(run_soar.history)}")
