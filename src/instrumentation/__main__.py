from staliro.options import Options
from staliro.staliro import staliro, simulate_model

from .model import ThermostatModel
from .optimizer import UniformRandom
from .specification import ThermostatSpecification, active_state

model = ThermostatModel()
specification = ThermostatSpecification()
options = Options(
    static_parameters=[(19.0, 22.0), (19.0, 22.0), (0.0, 1.0), (0.0, 1.0)],
    iterations=1000,
    runs=1,
)

result = staliro(model, specification, UniformRandom(), options)
run = result.runs[0]

print(f"Function branches: {len(specification.kripke.states)}")
print(f"Coverage achieved in: {len(run.history)}")
