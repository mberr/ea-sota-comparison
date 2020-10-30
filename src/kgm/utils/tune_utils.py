"""Utilities for ray.tune."""
import logging
import math
import random
from enum import Enum
from typing import Any, Callable, Mapping, Type, Sequence

from ray import tune
from ray.tune.schedulers import MedianStoppingRule, TrialScheduler

from kgm.modules import BaseLoss, ContrastiveLoss, MarginLoss
from kgm.modules.losses import BCELoss, base_loss_name_normalizer, ActivationEnum
from kgm.utils.common import get_all_subclasses, enum_values

logger = logging.getLogger(name=__name__)


class MedianStoppingRuleNoPreemption(MedianStoppingRule):
    """A modified median stopping rule, which does not pause any trial."""

    def __init__(
        self,
        time_attr="time_total_s",
        reward_attr=None,
        metric="episode_reward_mean",
        mode="max",
        grace_period=60.0,
        min_samples_required=3,
    ):  # noqa: D107
        super().__init__(
            time_attr=time_attr,
            reward_attr=reward_attr,
            metric=metric,
            mode=mode,
            grace_period=grace_period,
            min_samples_required=min_samples_required,
            hard_stop=True,  # no pausing
        )

    def _on_insufficient_samples(self, trial_runner, trial, time):  # noqa: D102
        # No preemption
        return TrialScheduler.CONTINUE


def _log_uniform_int(
    low: int,
    high: int,
    q: int,
) -> int:
    return int(round(math.exp(random.uniform(math.log(low), math.log(high))))) // q * q


def tune_log_uniform_int(
    low: int,
    high: int,
    q: int,
) -> Callable[[Mapping], int]:
    """
    Create a sampler for integer log uniform sampling between bounds.

    :param low:
        The lower bound.
    :param high:
        The upper bound.
    :param q:
        The sampled values are multiples of q.
    """

    def sample(_spec):
        return _log_uniform_int(low=low, high=high, q=q)

    return sample


def tune_enum(enum: Type[Enum]):
    """Tune enum values."""
    return tune.choice(enum_values(enum_cls=enum))


#: Shorthand for chosing a boolean
tune_bool = tune.choice([False, True])

#: HPO for margin activation ("hard" vs "soft" margin)
tune_margin_activation = tune_enum(ActivationEnum)

# default HPO ranges for pairwise losses
default_pairwise_loss_hpo: Mapping[Type[BaseLoss], Mapping[str, Any]] = {
    MarginLoss: dict(
        margin=tune.loguniform(1.0e-03, 1.0e+01),
        activation=tune_margin_activation,
    ),
    ContrastiveLoss: dict(
        positive_margin=tune.loguniform(1.0e-04, 1.0),
        negative_margin=tune.loguniform(1.0e-04, 1.0e-02),
        normalize_negatives=tune_bool,
        activation=tune_margin_activation,
    ),
    BCELoss: dict(
        balance=tune_bool,
    )
}


def _sample_base_loss(_spec):
    """Sample a pairwise loss with appropriate HPO sub-space."""
    cls = random.choice(list(get_all_subclasses(BaseLoss)))
    kwargs = default_pairwise_loss_hpo.get(cls, dict())
    kwargs['cls'] = base_loss_name_normalizer(name=cls.__name__)
    return kwargs


default_sampled_matching_loss_hpo = dict(
    pairwise_loss=tune.sample_from(_sample_base_loss),
    num_negatives=tune.sample_from(tune_log_uniform_int(1, 256, q=1)),
    self_adversarial_weighting=tune_bool,
)


def tune_linear_quantized_range(high: int, low: int = None, q: int = 1) -> Sequence[int]:
    if low is None:
        low = q
    low = low // q
    high = high // q + 1
    low = min(1, low)
    return tune.choice([q * i for i in range(low, high)])
