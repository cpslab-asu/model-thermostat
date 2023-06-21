import logging

from staliro.core import Model
from staliro.options import Options
from staliro.staliro import staliro

from cli import parser
from instrumentation.model import InstrumentedOutput, Thermostat2Rooms, Thermostat4Rooms
from instrumentation.optimizer import SOAR, UniformRandom
from instrumentation.specification import ThermostatRequirement
from thermostat import Controller, controller_2rooms, controller_4rooms

logger = logging.getLogger("safety")


def main():
    args = parser.parse_args()

    init_temp_bounds = (19.0, 22.0)
    cool_coeff_bounds = (0.0, 1.0)

    if args.rooms == 2:
        controller: Controller = controller_2rooms
        model: Model[InstrumentedOutput, None] = Thermostat2Rooms(controller)
        formula: str = r"[] (-1.0 * room1 <= -19.0 /\ (1.0 * room1 <= 22.0 /\ (-1.0 * room2 <= -19.0 /\ 1.0 * room2 <= 22.0)))"
        static_params = [init_temp_bounds] * 2 + [cool_coeff_bounds] * 2
    elif args.rooms == 4:
        controller = controller_4rooms
        model = Thermostat4Rooms(controller)
        formula = r"[] (-1.0 * room1 <= -19.0 /\ (1.0 * room1 <= 22.0 /\ (-1.0 * room2 <= -19.0 /\ (1.0 * room2 <= 22.0 /\ -1.0 * room3 <= -19.0 /\ (1.0 * room3 <= 22.0 /\ (-1.0 * room4 <= -19.0 /\ 1.0 * room4 <= 22.0))))))"
        static_params = [init_temp_bounds] * 4 + [cool_coeff_bounds] * 4
    else:
        raise ValueError(f"Unsupported number of rooms {args.rooms}")

    options = Options(
        static_parameters=static_params,
        iterations=1000,
        runs=1,
    )

    logger.debug("====Options====")
    logger.debug(f"Rooms: {args.rooms}")
    logger.debug(f"Runs: {options.runs}")
    logger.debug(f"Iterations: {options.iterations}")
    logger.debug(f"Static parameters: {static_params}")

    ur_spec = ThermostatRequirement(formula, controller)

    logger.debug("Beginning uniform random execution")
    ur_result = staliro(model, ur_spec, UniformRandom(), options)
    logger.debug("Finished uniform random execution")

    ur_run = ur_result.runs[0]
    ur_best = min((e.cost for e in ur_run.history), key=lambda c: c.remaining_states)

    soar_spec = ThermostatRequirement(formula, controller)

    logger.debug("Beginning SOAR execution")
    soar_result = staliro(model, soar_spec, SOAR(), options)
    logger.debug("Finished SOAR execution")

    soar_run = soar_result.runs[0]
    soar_best = min((e.cost for e in soar_run.history), key=lambda c: c.remaining_states)

    n_branches = len(soar_spec.kripke.states)

    print(f"Function branches: {n_branches}")

    if ur_best.remaining_states == 0:
        print(f"UR coverage achieved in: {len(ur_run.history)}")
    else:
        print(f"UR coverage not achieved in: {len(ur_run.history)}")

    if soar_best.remaining_states == 0:
        print(f"SOAR coverage achieved in: {len(soar_run.history)}")
    else:
        print(f"SOAR coverage not achieved in: {len(soar_run.history)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("staliro").setLevel(logging.INFO)
    logging.getLogger("PySOAR-C").setLevel(logging.INFO)
    main()
