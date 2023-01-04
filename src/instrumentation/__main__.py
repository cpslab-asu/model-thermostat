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
best_eval = min((eval for run in result.runs for eval in run.history), key=lambda e: e.cost)
model_output_best = simulate_model(model, options, best_eval.sample)
most_states_visited = {
    active_state(specification.kripke, instr_output.variables)
    for instr_output in model_output_best.trace.states
}

worst_eval = max((eval for run in result.runs for eval in run.history), key=lambda e: e.cost)
model_output_worst = simulate_model(model, options, worst_eval.sample)
fewest_states_visited = {
    active_state(specification.kripke, instr_output.variables)
    for instr_output in model_output_worst.trace.states
}

print(f"Function branches: {len(specification.kripke.states)}")
print("========================")
print(f"Maximum Distance: {best_eval.cost}")
print(f"Maximum branches covered: {len(most_states_visited)}")
print("========================")
print(f"Minimum Distance: {worst_eval.cost}")
print(f"Minimum branches covered: {len(fewest_states_visited)}")
