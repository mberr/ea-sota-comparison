"""Sampling methods for negative samples."""
from abc import abstractmethod
from typing import Optional, Tuple

import torch

from kgm.utils.types import NodeIDs


class NegativeSampler:
    """Abstract class encapsulating a logic of choosing negative examples."""

    @abstractmethod
    def sample(
        self,
        size: Tuple[int, ...],
        device: torch.device,
        max_id: Optional[int] = None,
        candidates: Optional[NodeIDs] = None,
    ) -> NodeIDs:
        """Choose negative samples.

        If a set of candidates is provided, the samples are chosen from them. Otherwise, the max_id parameter will be
        used to sample from [0, max_id-1].

        :param size:
            Expected shape of the output tensor of indices.
        :param device:
            Device of the output tensor.
        :param max_id: >0
            The maximum ID (exclusive).
        :param candidates: shape: (num_of_candidates,)
            Tensor containing candidates for negative examples to choose from.
        """
        raise NotImplementedError


class UniformRandomSampler(NegativeSampler):
    """NegativeExamplesSampler implementation using uniform random distribution to choose negative samples."""

    def sample(
        self,
        size: Tuple[int, ...],
        device: torch.device,
        max_id: Optional[int] = None,
        candidates: Optional[NodeIDs] = None,
    ) -> NodeIDs:  # noqa: D102
        if candidates is not None:
            return candidates[torch.randint(candidates.shape[0], size=size, device=candidates.device)]
        return torch.randint(max_id, size=size, dtype=torch.long, device=device)
