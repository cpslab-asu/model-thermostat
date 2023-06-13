from argparse import ArgumentParser

from staliro.options import Options
from staliro.staliro import staliro

from .model import ThermostatModel
from .optimizer import SOAR, UniformRandom
from .specification import ThermostatRequirement, ThermostatSpecification


def coverage():
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


def _temp_req(name: str) -> str:
    return rf"-1.0 * {name} <= -19.0 /\ 1.0 * {name} <= 22.0"


def _temp_reqs() -> tuple[str, str, str, str]:
    return _temp_req("t1"), _temp_req("t1"), _temp_req("t1"), _temp_req("t1")


def coverage_safety():
    req1, req2, req3, req4 = _temp_reqs()
    spec = ThermostatRequirement(rf"[] {req1} /\ {req2} /\ {req3} /\ {req4}")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        default="coverage",
        choices=["coverage", "safety_coverage"],
        help="Evaluation mode",
    )

    args = parser.parse_args()

    if args.mode == "coverage":
        coverage()
    elif args.mode == "safety_coverage":
        coverage_safety()
    else:
        raise ValueError(f"Unknown mode {args.mode}")
