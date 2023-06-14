from staliro.options import Options
from staliro.staliro import staliro

from instrumentation.model import ThermostatModel
from instrumentation.optimizer import SOAR, UniformRandom
from instrumentation.specification import ThermostatRequirement


def _temp_req(name: str) -> str:
    return rf"-1.0 * {name} <= -19.0 /\ 1.0 * {name} <= 22.0"


def _temp_reqs(count: int) -> tuple[str, ...]:
    return tuple(_temp_req(f"t{n}") for n in range(1, count + 1))


def main():
    req1, req2, req3, req4 = _temp_reqs(4)
    req = rf"[] (-1.0 * t1 <= -19.0 /\ (1.0 * t1 <= 22.0 /\ (-1.0 * t2 <= -19.0 /\ (1.0 * t2 <= 22.0 /\ (-1.0 * t3 <= -19.0 /\ (1.0 * t3 <= 22.0 /\ (-1.0 * t4 <= -19.0 /\ 1.0 * t4 <= 22.0)))))))"
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

    spec_ur = ThermostatRequirement(req)
    result_ur = staliro(model, spec_ur, UniformRandom(), options)
    run_ur = result_ur.runs[0]

    spec_soar = ThermostatRequirement(req)
    result_soar = staliro(model, spec_soar, SOAR(), options)
    run_soar = result_soar.runs[0]

    print(f"Function branches: {len(spec_soar.kripke.states)}")
    print(f"UR coverage achieved in: {len(run_ur.history)}")
    print(f"SOAR coverage achieved in: {len(run_soar.history)}")


if __name__ == "__main__":
    main()
