# coding=utf-8
"""Node embedding initialization."""

import pathlib
from typing import Any, Optional, Union

import torch
from torch import nn

from ....data import MatchSideEnum


class NodeEmbeddingInitializer:
    """Initialization methods."""

    def init_one_(
        self,
        embedding: torch.FloatTensor,
    ) -> None:
        """
        Initialize embedding in-place.

        :param embedding:
            The embedding.
        """
        raise NotImplementedError

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimension."""
        return None


class RandomNodeEmbeddingInitializer(NodeEmbeddingInitializer):
    """Initialize nodes i.i.d. with random vectors drawn from the given distribution."""

    def __init__(
        self,
        random_distribution=nn.init.normal_,
        **kwargs: Any,
    ):
        """
        Initialize the initializers.

        :param random_distribution:
            The random distribution to use for initialization.
        """
        self.random_dist_ = random_distribution
        self.kwargs = kwargs

    def init_one_(
        self,
        embedding: torch.FloatTensor,
    ) -> None:  # noqa: D102
        self.random_dist_(embedding, **self.kwargs)


class PretrainedNodeEmbeddingInitializer(NodeEmbeddingInitializer):
    """Load pretrained node embeddings."""

    def __init__(
        self,
        embeddings: torch.FloatTensor,
    ):
        """
        Initialize the initializer.

        :param embeddings: shape: (n, d)
            The pretrained embeddings.
        """
        super().__init__()
        self.pretrained = embeddings

    @staticmethod
    def from_path(directory: Union[pathlib.Path, str], side: MatchSideEnum) -> 'PretrainedNodeEmbeddingInitializer':
        """Construct initializer from pretrained embeddings stored under a path."""
        return PretrainedNodeEmbeddingInitializer(
            embeddings=torch.load(
                PretrainedNodeEmbeddingInitializer.output_file_path(
                    directory=directory,
                    side=side,
                )
            )
        )

    @staticmethod
    def output_file_path(directory: Union[pathlib.Path, str], side: MatchSideEnum) -> pathlib.Path:
        """Return the canonical file path."""
        return pathlib.Path(directory) / f'{side.value}.pt'

    def save_to_path(self, directory: Union[pathlib.Path, str], side: MatchSideEnum) -> pathlib.Path:
        """Save pretrained node embedding into a file."""
        output_path = PretrainedNodeEmbeddingInitializer.output_file_path(directory=directory, side=side)
        torch.save(obj=self.pretrained, f=output_path)
        return output_path

    def init_one_(
        self,
        embedding: torch.FloatTensor,
    ) -> None:  # noqa: D102
        embedding.data.copy_(self.pretrained, non_blocking=True)

    @property
    def embedding_dim(self) -> Optional[int]:  # noqa: D102
        return self.pretrained.shape[-1]
