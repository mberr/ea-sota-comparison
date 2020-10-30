# coding=utf-8
"""API for models for knowledge graph matching."""
import logging
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Type, Union

import torch
from frozendict import frozendict
from torch import nn

from ...data import EntityAlignment, KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ...data.reduction import KnowledgeGraphToGraphReduction
from ...modules import Similarity
from ...modules.embeddings import get_embedding_pair
from ...modules.embeddings.base import EmbeddingNormalizationMode, NodeEmbeddingInitMethod
from ...modules.embeddings.norm import EmbeddingNormalizationMethod
from ...utils.common import get_subclass_by_name, kwargs_or_empty
from ...utils.torch_utils import ExtendedModule, maximize_memory_utilization
from ...utils.types import EntityIDs

logger = logging.getLogger(name=__name__)

__all__ = [
    'GraphBasedKGMatchingModel',
    'IndependentSideMixin',
    'KGMatchingModel',
    'PureEmbeddingModel',
    'get_matching_model_by_name',
]


class KGMatchingModel(ExtendedModule):
    """
    Generic class for (knowledge) graph matching models of a specific form.

    The models produce vector representation for each node, and the matching is done by comparing these representations
    by some similarity measure.
    """

    #: The number of nodes on each side.
    num_nodes: Mapping[MatchSideEnum, int]

    def __init__(
        self,
        num_nodes: Mapping[MatchSideEnum, int],
    ):
        """
        Initialize the model.

        :param num_nodes:
            The number of nodes on each side.
        """
        super().__init__()
        self.num_nodes = frozendict(num_nodes)
        self.batch_size = sum(num_nodes.values())

    # pylint: disable=arguments-differ
    @abstractmethod
    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, EntityIDs]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """Return embeddings for nodes on both sides.

        :param indices:
            If provided only return representations for these indices.

        :return: a mapping side -> representations
            where
            representations: shape: (num_nodes_on_side, embedding_dim)
        """
        raise NotImplementedError

    def _get_node_representations(
        self,
        indices: Mapping[MatchSideEnum, EntityIDs],
        batch_size: int,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """
        Batched calculation of node representations.

        :param indices:
            The indices for each side.
        :param batch_size:
            The batch size.

        :return:
            A mapping from side to node representations on side.
        """
        result = defaultdict(list)
        total_num_nodes = sum(v.shape[0] for v in indices.values())
        num_first_side = indices[SIDES[0]].shape[0]
        for start in range(0, total_num_nodes, batch_size):
            # construct indices
            batch_indices = dict()
            for i_side, side in enumerate(SIDES):
                start_side = max(start - i_side * num_first_side, 0)
                end_side = min(max(start + batch_size - i_side * num_first_side, 0), self.num_nodes[side])
                if end_side - start_side > 0:
                    batch_indices[side] = indices[side][start_side:end_side].to(self.device)

            # update result
            for side, partial_node_repr in self(indices=batch_indices).items():
                result[side].append(partial_node_repr)

        # combine result
        return {
            side: torch.cat(partial_node_repr)
            for side, partial_node_repr in result.items()
        }

    def get_node_representations(
        self,
        indices: Optional[Mapping[MatchSideEnum, EntityIDs]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:
        """
        Calculate node representations for all nodes using batching.

        :param indices:
            Optional restriction to some indices.

        :return:
            The node representations.
        """
        if indices is None:
            indices = {
                side: torch.arange(num, device=self.device)
                for side, num in self.num_nodes.items()
            }
        result, self.batch_size = maximize_memory_utilization(
            self._get_node_representations,
            parameter_name='batch_size',
            parameter_max_value=self.batch_size,
            indices=indices,
        )
        return result


class IndependentSideMixin(KGMatchingModel):
    """Mix-in for models which compute independent representations on each side."""

    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, EntityIDs]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        if indices is None:
            indices = {
                side: None
                for side in SIDES
            }

        return {
            side: self._forward_side(side=side, indices=indices_on_side)
            for side, indices_on_side in indices.items()
        }

    @abstractmethod
    def _forward_side(
        self,
        side: MatchSideEnum,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute node representations on one side.

        :param side:
            The side.
        :param indices:
            The indices. None means to compute all representations.

        :return: shape: (num_indices, embedding_dim)
            The node representations.
        """
        raise NotImplementedError


# pylint: disable=abstract-method
class GraphBasedKGMatchingModel(KGMatchingModel, ABC):
    """A knowledge graph matching model explicitly using the graph structure."""

    #: The reductions to adjacency matrices.
    reductions: Mapping[MatchSideEnum, KnowledgeGraphToGraphReduction]

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        reduction_cls: Type[KnowledgeGraphToGraphReduction],
        reduction_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """
        Initialize the model.

        :param dataset:
            The dataset.
        :param reduction_cls:
            The reduction strategy to obtain a (weighted) adjacency matrix from a knowledge graph.
        :param reduction_kwargs:
            Optional key-word based arguments to pass to the reduction.
        """
        super().__init__(num_nodes=dataset.num_nodes)
        reduction_kwargs = kwargs_or_empty(reduction_kwargs)
        self.reductions = nn.ModuleDict({
            side: reduction_cls(knowledge_graph=graph, **reduction_kwargs)
            for side, graph in dataset.graphs.items()
        })


def get_matching_model_by_name(
    name: str,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Type[KGMatchingModel]:
    """
    Get a matching model class by name.

    :param name:
        The name.
    :param normalizer:
        An optional custom name normalization method.

    :return:
        The matching class.
    """
    if normalizer is None:
        normalizer = str.lower
    return get_subclass_by_name(base_class=KGMatchingModel, name=name, normalizer=normalizer, exclude={GraphBasedKGMatchingModel})


class PureEmbeddingModel(IndependentSideMixin, KGMatchingModel):
    """A knowledge graph matching model with learned node representations without interaction between the nodes."""

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        embedding_dim: int = 3,
        node_embedding_init_method: NodeEmbeddingInitMethod = NodeEmbeddingInitMethod.random,
        node_embedding_init_config: Optional[Mapping[str, Any]] = None,
        node_embedding_normalization_method: EmbeddingNormalizationMethod = EmbeddingNormalizationMethod.none,
        node_embedding_normalization_mode: EmbeddingNormalizationMode = EmbeddingNormalizationMode.none,
        dropout: Optional[float] = None,
    ):
        """
        Initialize the model.

        :param embedding_dim: > 0
            The dimensionality of the embedding.
        :param node_embedding_init_method:
            The embedding initialization method used for the node embeddings.
        :param node_embedding_init_config:
            Additional keyword based arguments for the initializer.
        :param node_embedding_normalization_method:
            The node embedding normalization method.
        :param node_embedding_normalization_mode:
            The node embedding normalization mode.
        :param dropout:
            If present, apply dropout to the node embeddings.
        """
        super().__init__(num_nodes=dataset.num_nodes)
        self.embeddings = get_embedding_pair(
            init=node_embedding_init_method,
            dataset=dataset,
            embedding_dim=embedding_dim,
            dropout=dropout,
            trainable=True,
            init_config=node_embedding_init_config,
            norm=node_embedding_normalization_method,
            normalization_mode=node_embedding_normalization_mode,
        )
        self.reset_parameters()

    def _forward_side(
        self,
        side: MatchSideEnum,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        return self.embeddings[side](indices=indices)


@torch.no_grad()
def store_similarity_matrix(
    model: KGMatchingModel,
    alignment: Union[KnowledgeGraphAlignmentDataset, EntityAlignment, Mapping[str, torch.LongTensor]],
    similarity: Similarity,
    output_path: pathlib.Path,
    **kwargs,
) -> None:
    """
    Compute and store pairwise similarity matrices.

    :param model:
        The model for the entity representations.
    :param alignment:
        The alignment.
    :param similarity:
        The similarity to use.
    :param output_path:
        The output path.
    :param kwargs:
        Additional key-word based arguments stored as meta-data in the file.
    """
    # Input normalization
    if isinstance(alignment, KnowledgeGraphAlignmentDataset):
        alignment = alignment.alignment
    if isinstance(alignment, EntityAlignment):
        alignment = alignment.to_dict()

    # get all node representations
    node_repr = model.forward()

    result = dict(**kwargs)
    for key, indices in alignment.items():
        left, right = [
            node_repr[side][idx]
            for side, idx in zip(SIDES, indices)
        ]
        sim = similarity.all_to_all(left=left, right=right)
        result[key] = sim.to("cpu")

    # save to file
    torch.save(result, output_path)
