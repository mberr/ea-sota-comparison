# coding=utf-8
"""Embedding normalization."""
import enum
from abc import abstractmethod
from typing import Union

import torch
from torch.nn import functional

from ...utils.common import get_subclass_by_name


class EmbeddingNormalizer:
    """Embedding normalization."""

    @abstractmethod
    def normalize(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Normalize a batch of embeddings, e.g. during forward pass.

        :param x: shape: (batch_size, dim)
            The tensor of embeddings.
        """
        raise NotImplementedError


class LpNormalization(EmbeddingNormalizer):
    """Normalize the unit L_p norm."""

    def __init__(self, p: int):
        """
        Initialize the normalizer.

        :param p: >0
            The parameter p of the Lp distance.
        """
        self.p = p

    def normalize(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return functional.normalize(x, p=self.p, dim=-1)


def norm_method_normalizer(name: str):
    """Normalize the name of a normalization method."""
    return name.lower().replace('_', '').replace('embeddingnormalizer', '')


class L2EmbeddingNormalizer(LpNormalization):
    """L2 normalization."""

    def __init__(self):
        """Initialize the normalizer."""
        super().__init__(p=2)


class L1EmbeddingNormalizer(LpNormalization):
    """L1 normalization."""

    def __init__(self):
        """Initialize the normalizer."""
        super().__init__(p=1)


class NoneEmbeddingNormalizer(EmbeddingNormalizer):
    """Dummy normalization which does not actually change anything."""

    def normalize(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        return x


@enum.unique
class EmbeddingNormalizationMethod(str, enum.Enum):
    """An enum for embedding normalizations."""

    none = 'none'
    l2 = 'l2'
    l1 = 'l1'


def get_normalizer_by_name(name: Union[EmbeddingNormalizationMethod, str]) -> EmbeddingNormalizer:
    """Get an embedding normalizer by name."""
    if isinstance(name, EmbeddingNormalizationMethod):
        name = name.value
    norm_class = get_subclass_by_name(
        base_class=EmbeddingNormalizer,
        name=name,
        normalizer=norm_method_normalizer,
    )
    return norm_class()
