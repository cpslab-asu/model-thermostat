from __future__ import annotations

from typing import Generator, Sequence

import numpy as np
import numpy.random as rand
from numpy.typing import NDArray
from pysoarc import PySOARC  # pyright: ignore[reportMissingTypeStubs]
from pysoarc.coreAlgorithm import Behavior  # pyright: ignore[reportMissingTypeStubs]
from pysoarc.gprInterface import InternalGPR  # pyright: ignore[reportMissingTypeStubs]
from staliro.core import Interval, Sample
from staliro.core.optimizer import ObjectiveFn, Optimizer

from .specification import CoverageSafety


class UniformRandom(Optimizer[CoverageSafety, None]):
    """Uniform random optimizer specialized to consume hybrid distance cost values.

    This optimizer will stop generating samples when the number of uncovered states reaches zero.
    Samples are generated randomly from the input space.
    """

    def optimize(
        self,
        func: ObjectiveFn[CoverageSafety],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> None:
        def _randinterval(rng: rand.Generator, interval: Interval) -> float:
            return interval.lower + rng.random() * interval.length

        def _randsample(rng: rand.Generator, intervals: Sequence[Interval]) -> Sample:
            return Sample(tuple(_randinterval(rng, interval) for interval in intervals))

        def _randsamples(
            rng: rand.Generator, intervals: Sequence[Interval]
        ) -> Generator[Sample, None, None]:
            for _ in range(budget):
                yield _randsample(rng, intervals)

        rng = rand.default_rng(seed)
        samples = _randsamples(rng, bounds)
        evaluations = map(func.eval_sample, samples)

        for evaluation in evaluations:
            if evaluation.remaining_states == 0:
                break


class SOAR(Optimizer[CoverageSafety, None]):
    def optimize(
        self,
        func: ObjectiveFn[CoverageSafety],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> None:
        def _test_fn(input: NDArray[np.double]) -> tuple[float, float] | None:
            if input.ndim != 1:
                raise ValueError("Input array must be 1-d")

            sample = Sample(tuple(input.tolist()))
            result = func.eval_sample(sample)

            if result.remaining_states == 0:
                return None

            return (result.safety, abs(result.coverage))

        _ = PySOARC(
            n_0=20,
            nSamples=budget,
            trs_max_budget=5,
            max_loc_iter=5,
            inpRanges=np.array([bound.astuple() for bound in bounds]),
            alpha_lvl_set=0.05,
            eta0=0.25,
            eta1=0.75,
            delta=0.75,
            gamma=1.25,
            eps_tr=0.01,
            min_tr_size=10.0,
            TR_threshold=0.05,
            test_fn=_test_fn,  # type: ignore
            gpr_model=InternalGPR(),
            seed=seed,
            local_search="gp_local_search",
            behavior=Behavior.COVERAGE,
        )

        return None
