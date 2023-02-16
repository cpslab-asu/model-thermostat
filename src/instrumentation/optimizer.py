from __future__ import annotations

from itertools import islice
from typing import Generator, Sequence

import numpy.random as rand
from staliro.core import Interval, Sample
from staliro.core.optimizer import ObjectiveFn, Optimizer

from .specification import SystemCoverage


class UniformRandom(Optimizer[SystemCoverage, None]):
    """Uniform random optimizer specialized to consume hybrid distance cost values."""

    def optimize(
        self,
        func: ObjectiveFn[SystemCoverage],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> None:
        def _randinterval(rng: rand.Generator, interval: Interval) -> float:
            return interval.lower + rng.random() * interval.length

        def _randsample(rng: rand.Generator, intervals: Sequence[Interval]) -> Sample:
            return Sample(tuple(_randinterval(rng, interval) for interval in intervals))

        def _randsamples(rng: rand.Generator, intervals: Sequence[Interval]) -> Generator[Sample, None, None]:
            while True:
                yield _randsample(rng, intervals)

        rng = rand.default_rng(seed)
        samples = islice(_randsamples(rng, bounds), budget)
        evaluations = map(func.eval_sample, samples)

        for evaluation in evaluations:
            if evaluation.remaining_states == 0:
                break

