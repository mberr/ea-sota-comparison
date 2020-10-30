"""Models from Deep Graph Matching Consensus."""
from typing import Any, Mapping, Optional, Type, Union

import torch
from torch import nn
from torch.nn import functional

from .base import GraphBasedKGMatchingModel, IndependentSideMixin
from ...data import KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ...data.reduction import DropRelationInformationKnowledgeGraphToGraphReduction, KnowledgeGraphToGraphReduction, symmetric_normalization
from ...modules.embeddings import get_embedding_pair
from ...modules.embeddings.base import NodeEmbeddingInitMethod, EmbeddingNormalizationMode
from ...modules.embeddings.init import NodeEmbeddingInitializer
from ...modules.embeddings.norm import EmbeddingNormalizationMethod
from ...utils.torch_utils import ExtendedModule, SparseCOOMatrix


# pylint: disable=abstract-method
class RelConv(ExtendedModule):
    """GCN variant used by GCN-Align implementation of DGMC."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Initialize the module.

        :param in_channels:
            The number of input channels.
        :param out_channels:
            The number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.forward_trans = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        self.backward_trans = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        self.root = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        adjacency: SparseCOOMatrix,
    ) -> torch.FloatTensor:
        r"""
        Enrich node embeddings by bi-directional message passing with explicit self-loop.

        .. math ::
            Y = AXW_1 + A^{T}XW_2 + XW_3

        :param x: shape: (num_nodes, in_channels)
            The node features.
        :param adjacency: shape: (num_nodes, num_nodes)
            The node adjacency matrix.

        :return: shape: (num_nodes, out_channels)
            The new node features.
        """
        return self.root(x) + self._send_messages(
            x=x,
            adjacency=adjacency,
            transformation=self.forward_trans,
        ) + self._send_messages(
            x=x,
            adjacency=adjacency.t(),
            transformation=self.backward_trans,
        )

    def _send_messages(
        self,
        x: torch.FloatTensor,
        adjacency: SparseCOOMatrix,
        transformation: nn.Module,
    ) -> torch.FloatTensor:
        """Send messages with mean aggregation."""
        return adjacency @ transformation(x) / (adjacency.without_weights() @ x.new_ones(x.shape[0], 1)).clamp_min_(1)


class DGMCGCNAlign(IndependentSideMixin, GraphBasedKGMatchingModel):
    """
    GCN-Align model implementation according to variant from Deep Graph Matching Consensus.

    .. seealso ::
        https://github.com/rusty1s/deep-graph-matching-consensus/blob/a25f89751f4a3a0d509baa6bbada8b4153c635f6/dgmc/models/rel.py
    """

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        output_dim: int,
        num_layers: int,
        batch_norm: bool = False,
        cat: bool = True,
        final_linear_projection: bool = True,
        dropout: float = 0.5,
        node_embedding_init_method: Union[NodeEmbeddingInitMethod, Type[NodeEmbeddingInitializer], NodeEmbeddingInitializer] = NodeEmbeddingInitMethod.random,
        node_embedding_init_config: Optional[Mapping[str, Any]] = None,
        embedding_dim: Optional[int] = None,
        trainable_node_embeddings: bool = True,
        node_embedding_norm: EmbeddingNormalizationMethod = EmbeddingNormalizationMethod.none,
        node_embedding_mode: EmbeddingNormalizationMode = EmbeddingNormalizationMode.none,
        reduction_cls: Type[KnowledgeGraphToGraphReduction] = DropRelationInformationKnowledgeGraphToGraphReduction,
        reduction_kwargs: Optional[Mapping[str, Any]] = None,
        vertical_sharing: bool = False,
        horizontal_sharing: bool = True,
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
        if reduction_kwargs is None:
            reduction_kwargs = dict(
                unique=True,
                normalization=symmetric_normalization,
            )
        super().__init__(
            dataset=dataset,
            reduction_cls=reduction_cls,
            reduction_kwargs=reduction_kwargs,
        )

        self.embeddings = get_embedding_pair(
            init=node_embedding_init_method,
            dataset=dataset,
            embedding_dim=embedding_dim,
            trainable=trainable_node_embeddings,
            init_config=node_embedding_init_config,
            norm=node_embedding_norm,
            normalization_mode=node_embedding_mode,
        )

        if embedding_dim is None:
            embedding_dim = self.embeddings[MatchSideEnum.left].embedding_dim

        self.in_channels = embedding_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.final_linear_projection = final_linear_projection
        self.dropout = dropout
        self.vertical_sharing = vertical_sharing
        self.horizontal_sharing = horizontal_sharing

        convs = []  # nn.ModuleList()
        batch_norms = []  # nn.ModuleList()

        if self.vertical_sharing:
            rel_conv = RelConv(embedding_dim, output_dim)
            batch_norm = nn.BatchNorm1d(num_features=output_dim) if self.batch_norm else None
            for _ in range(num_layers):
                convs.append(rel_conv)
                batch_norms.append(batch_norm)

        else:
            for _ in range(num_layers):
                convs.append(RelConv(embedding_dim, output_dim))
                batch_norms.append(nn.BatchNorm1d(num_features=output_dim) if self.batch_norm else None)
                embedding_dim = output_dim

        if self.horizontal_sharing:
            _convs = nn.ModuleList(convs)
            _norms = nn.ModuleList(batch_norms)
            side_to_convs = {
                side: _convs
                for side in SIDES
            }
            side_to_norms = {
                side: _norms
                for side in SIDES
            }

        else:
            side_to_convs = {
                side: nn.ModuleList(convs)
                for side in SIDES
            }
            side_to_norms = {
                side: nn.ModuleList(batch_norms)
                for side in SIDES
            }

        self.convs = nn.ModuleDict(modules=side_to_convs)
        self.batch_norms = nn.ModuleDict(modules=side_to_norms)

        if self.cat:
            embedding_dim = self.in_channels + num_layers * output_dim
        else:
            embedding_dim = output_dim

        if self.final_linear_projection:
            self.final = nn.Linear(in_features=embedding_dim, out_features=output_dim, bias=True)

        self.reset_parameters()

    def _forward_side(
        self,
        side: MatchSideEnum,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        x = self.embeddings[side](indices=None)
        adjacency = self.reductions[side]()
        out = [x]

        convs = self.convs[side]
        batch_norms = self.batch_norms[side]

        for conv, batch_norm in zip(convs, batch_norms):
            x = conv(x=x, adjacency=adjacency).relu()
            if batch_norm is not None:
                assert isinstance(batch_norm, nn.BatchNorm1d)
                x = batch_norm(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
            if self.cat:
                out.append(x)
            else:
                out = [x]

        x = torch.cat(out, dim=-1)
        if indices is not None:
            x = x[indices]
        return self.final(x) if self.final_linear_projection else x
