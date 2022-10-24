from __future__ import annotations

from typing import Sequence

import numpy.random as rand
from staliro.core import Interval, Sample
from staliro.core.optimizer import ObjectiveFn, Optimizer

from .specification import HybridDistance


class UniformRandom(Optimizer[HybridDistance, None]):
    """Uniform random optimizer specialized to consume hybrid distance cost values."""

    def optimize(
        self,
        func: ObjectiveFn[HybridDistance],
        bounds: Sequence[Interval],
        budget: int,
        seed: int,
    ) -> None:
        def _randinterval(rng: rand.Generator, interval: Interval) -> float:
            return interval.lower + rng.random() * interval.length

        def _randsample(rng: rand.Generator, intervals: Sequence[Interval]) -> Sample:
            return Sample([_randinterval(rng, interval) for interval in intervals])

        rng = rand.default_rng(seed)
        samples = [_randsample(rng, bounds) for _ in range(budget)]

        for sample in samples:
            _ = func.eval_sample(sample)
