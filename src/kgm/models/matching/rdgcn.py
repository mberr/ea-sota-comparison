"""An implementation of RDGCN as described in https://arxiv.org/abs/1908.08210 ."""

import logging
import math
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional

from .base import GraphBasedKGMatchingModel
from ...data import KnowledgeGraph, KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ...data.reduction import DropRelationInformationKnowledgeGraphToGraphReduction, symmetric_normalization
from ...modules.embeddings import get_embedding_pair
from ...modules.embeddings.base import Embedding, EmbeddingNormalizationMode, NodeEmbeddingInitMethod
from ...modules.embeddings.init.base import NodeEmbeddingInitializer
from ...modules.embeddings.norm import EmbeddingNormalizationMethod
from ...utils.torch_utils import SparseCOOMatrix, simple_sparse_softmax
from ...utils.types import EntityIDs, Triples

logger = logging.getLogger(name=__name__)


def get_relation_dual_graph_dense(
    knowledge_graph: KnowledgeGraph,
) -> torch.FloatTensor:
    r"""
    Generate the dense adjacency matrix of the dual relation graph, given the sparse triple representation of one KG.

    The construction follows the dual relation graph definition from https://arxiv.org/abs/1908.08210 .

    .. math ::
        A[i, j] = Jaccard(H[i], H[j]) + Jaccard(T[i], T[j])

    where

    .. math ::
        H[i] := \{h \in \mathcal{E} \mid \exists t: (h, r_i, t) \in \mathcal{T}\}

        T[i] := \{t \in \mathcal{E} \mid \exists h: (h, r_i, t) \in \mathcal{T}\}

    denote the set of entities which are head/tail entities for relation :math:`r_i`.

    :param knowledge_graph:
        The knowledge graph.

    :return: shape: (num_relations, num_relations)
        The dense relation graph adjacency matrix.
    """
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t in knowledge_graph.triples.tolist():
        heads[r].add(h)
        tails[r].add(t)

    weights = 2 * torch.eye(knowledge_graph.num_relations, dtype=torch.float32)
    for i in range(knowledge_graph.num_relations):
        for j in range(i + 1, knowledge_graph.num_relations):
            a_h = len(heads[i] & heads[j]) / len(heads[i] | heads[j])
            a_t = len(tails[i] & tails[j]) / len(tails[i] | tails[j])
            weights[j, i] = weights[i, j] = a_h + a_t

    return weights


def get_sparse_entity_to_relation_matrix(
    graph: KnowledgeGraph,
    entity_col: int,
) -> SparseCOOMatrix:
    r"""
    Create a entity to relation adjacency matrix.

    The adjacency matrix is a column normalized matrix

    .. math ::
        A[i, j] = \hat{A}_{ij} / \sum_{j} \hat{A}_{ij}

    derived from the entity-relation co-occurence matrix

    .. math ::
        \hat{A}_{ij} = \mathbb{I}[\exists T \in \mathcal{T}: T[entity\_col] = j \land T[1] = i]

    :param graph:
        The knowledge graph.
    :param entity_col:
        The column for entities (0: head, 2: tail).

    :return:
        A sparse matrix S, such that R = S @ E.
    """
    return SparseCOOMatrix.from_indices_values_pair(
        indices=graph.triples[:, [1, entity_col]].unique(dim=0).t(),
        size=(graph.num_relations, graph.num_entities),
    ).normalize(dim=1)


# pylint: disable=abstract-method
class SparseAttentionLayer(nn.Module):
    """
    The primal GAT layer.

    .. seealso ::
        * Paper: https://www.ijcai.org/Proceedings/2019/0733.pdf, eq. (6) and (7)
        + Code: https://github.com/StephanieWyt/RDGCN/blob/ebbf6e7585c0acf31f9b0e59ae38b2cbf212fb18/include/Model.py#L100-L114
    """

    def __init__(
        self,
        input_dim: int,
        activation: nn.Module,
        wu2019_init: bool = False,
    ):
        """
        Initialize the module.

        :param input_dim: >0
            The input dimension.
        :param activation:
            The activation function.
        """
        super().__init__()
        self.trans = nn.Linear(in_features=input_dim, out_features=1, bias=True)
        self.activation = activation
        self.wu2019_init = wu2019_init

    # pylint: disable=arguments-differ
    def forward(
        self,
        entities: torch.FloatTensor,
        relations: torch.FloatTensor,
        triples: Triples,
    ) -> torch.FloatTensor:
        """
        Compute entity representations.

        :param entities: shape: (num_entities, entity_dim)
            The entity representations.
        :param relations: shape: (num_relations, relation_dim)
            The relation representations.
        :param triples: shape: (num_triples, 3)
            The triples.

        :return: shape: (num_entities, entity_dim)
            The new entity representations.
        """
        h, r, t = triples.t()
        dual_transform = functional.leaky_relu(self.trans(relations).squeeze(dim=-1)).index_select(dim=0, index=r)
        # logits = torch.sparse_coo_tensor(indices=r_mat_ind, values=dual_transform, size=[inlayer.shape[0]] * 2).coalesce()
        # coefs = torch.sparse.softmax(logits, dim=1)
        logits = simple_sparse_softmax(edge_weights=dual_transform, index=h, num_nodes=entities.shape[0], method='cpu_max')
        coefs = torch.sparse_coo_tensor(indices=torch.stack([h, t]), values=logits, size=[entities.shape[0]] * 2)
        vals = torch.sparse.mm(mat1=coefs, mat2=entities)
        if self.activation is not None:
            vals = self.activation(vals)
        return vals

    def reset_parameters(self):
        """Reset the layer's parameters."""
        # comment: since the original code base uses Conv1d with reshaping instead of linear layers, we need to adjust
        # the initialization manually
        if self.wu2019_init:
            nn.init.kaiming_uniform_(self.trans.weight, a=math.sqrt(5))
            fan_in = self.trans.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trans.bias, -bound, bound)


# pylint: disable=abstract-method
class DenseAttentionLayer(nn.Module):
    """
    Dense dual attention layer.

    .. seealso ::
        * Paper: https://www.ijcai.org/Proceedings/2019/0733.pdf, eq. (3) and (4)
        + Code: https://github.com/StephanieWyt/RDGCN/blob/adb2ec056fd4d3df92983583c1e2d3b18cf76d53/include/Model.py#L117-L132
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_func: nn.Module,
        wu2019_init: bool = False,
    ):
        """
        Initialize the module.

        :param input_dim: >0
            The input dimension.
        :param hidden_dim: >0
            The output dimension.
        :param act_func:
            The activation function.
        """
        super().__init__()
        self.common = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.left = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)
        self.right = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)
        self.act_func = act_func
        self.wu2019_init = wu2019_init

    # pylint: disable=arguments-differ
    def forward(
        self,
        relation: torch.FloatTensor,
        context: torch.FloatTensor,
        adj_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""
        Compute relation representations.

        :param context: (num_relations, 2 * entity_dim)
            The relation context representations obtained from the (primal) entity representations.
        :param relation: shape: (num_relations, relation_dim)
            The relation representations ("dual vertex representations").
        :param adj_tensor:
            The relation-relation adjacency matrix.

        :return: shape: (num_relations, relation_dim)
            The new relation representations.
        """
        in_fts = self.common(context)
        logits = self.left(in_fts) + self.right(in_fts).t()
        logits = functional.leaky_relu(logits, negative_slope=0.2)
        # masking
        logits = logits.masked_fill(mask=adj_tensor <= 0., value=float('-inf'))
        coefs = functional.softmax(logits, dim=-1)
        x = coefs @ relation
        if self.act_func is not None:
            x = self.act_func(x)
        return x

    def reset_parameters(self):
        """Reset the layer's parameters."""
        # comment: since the original code base uses Conv1d with reshaping instead of linear layers, we need to adjust
        # the initialization manually
        if self.wu2019_init:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in = module.weight.shape[1]
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(module.bias, -bound, bound)


# pylint: disable=abstract-method
class DiagLayer(nn.Module):
    """GCN layer with diagonal weight matrix."""

    def __init__(
        self,
        dimension: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        """
        Initialize the module.

        :param dimension: > 0
            The dimension.
        :param activation:
            The activation function.
        :param dropout:
            An input dropout value.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, dimension))
        self.dropout = dropout
        self.act_func = activation

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        adjacency: torch.sparse.Tensor,
    ) -> torch.FloatTensor:  # noqa: D102
        if self.dropout > 0:
            x = functional.dropout(x, p=self.dropout)
        x = x * self.weight
        x = torch.sparse.mm(mat1=adjacency, mat2=x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

    def reset_parameters(self):
        """Reset the module's parameters."""
        nn.init.ones_(self.weight)


# pylint: disable=abstract-method
class HighwayLayer(nn.Module):
    r"""
    Implementation of Highway Layer.

    .. math ::
        g = \sigma(Wx_1 + b)

        y = g \cdot x_1 + (1 - g) \cdot x_2
    """

    def __init__(
        self,
        dimension: int,
        wu2019_init: bool = False,
    ):
        """
        Initialize the module.

        :param dimension: >0
            The dimension.
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features=dimension, out_features=dimension, bias=True),
            nn.Sigmoid(),
        )
        self.wu2019_init = wu2019_init

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        transform_gate = self.gate(x)
        carry_gate = 1.0 - transform_gate
        return transform_gate * y + carry_gate * x

    def reset_parameters(self):
        """Reset the module's parameters."""
        if self.wu2019_init:
            nn.init.xavier_normal_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)


class RDGCN(GraphBasedKGMatchingModel):
    """Relation-aware Dual-Graph Convolutional Network (RDGCN)."""

    #: The entity embeddings
    embeddings: Mapping[MatchSideEnum, Embedding]

    #: The model's layers
    layers: Sequence[nn.Module]

    def __init__(
        self,
        dataset: KnowledgeGraphAlignmentDataset,
        embedding_dim: int = 300,
        interaction_round_weights: Sequence[float] = (0.1, 0.3),
        embedding_norm: EmbeddingNormalizationMethod = EmbeddingNormalizationMethod.l2,
        embedding_norm_mode: EmbeddingNormalizationMode = EmbeddingNormalizationMode.every_forward,
        trainable_node_embeddings: bool = True,
        num_interactions: int = 2,
        num_gcn_layers: int = 2,
        node_embedding_init_method: Union[NodeEmbeddingInitMethod, Mapping[MatchSideEnum, NodeEmbeddingInitializer]] = None,
        node_embedding_init_config: Optional[Mapping[str, Any]] = None,
        wu2019_init: bool = False,
    ):
        """
        Initialize the model.

        :param dataset:
            The dataset.
        :param embedding_dim:
            The embedding dimension.
        """
        super().__init__(
            dataset=dataset,
            reduction_cls=DropRelationInformationKnowledgeGraphToGraphReduction,
            reduction_kwargs=dict(
                unique=True,
                normalization=symmetric_normalization,
                add_self_loops=True,
                add_inverse=True,
            )
        )

        if node_embedding_init_method is None:
            node_embedding_init_method = NodeEmbeddingInitMethod.rdgcn_precomputed

        self.beta = interaction_round_weights

        # Use L2 normalization, cf. https://github.com/StephanieWyt/RDGCN/blob/ebbf6e7585c0acf31f9b0e59ae38b2cbf212fb18/include/Model.py#L194
        # openea does not not use L2: https://github.com/nju-websoft/OpenEA/blob/master/src/openea/approaches/rdgcn.py#L282-L288
        self.embeddings = get_embedding_pair(
            init=node_embedding_init_method,
            dataset=dataset,
            embedding_dim=embedding_dim,
            norm=embedding_norm,
            normalization_mode=embedding_norm_mode,
            trainable=trainable_node_embeddings,
            init_config=node_embedding_init_config,
        )
        embedding_dim = next(iter(self.embeddings.values())).embedding_dim

        # Precompute adjacency matrices

        # GCN adjacency -> see self.reductions

        # head_r, tail_r: (E x R); used for obtaining relation representations from entity representations
        self.head_to_rel = nn.ModuleDict({
            side: get_sparse_entity_to_relation_matrix(graph=graph, entity_col=0)
            for side, graph in dataset.graphs.items()
        })
        self.tail_to_rel = nn.ModuleDict({
            side: get_sparse_entity_to_relation_matrix(graph=graph, entity_col=2)
            for side, graph in dataset.graphs.items()
        })

        # primal adjacency: triples ( called r_mat / r_val in RDGCN code base)
        # adjacency matrix of the dual graph; (R x R), dense
        for side in SIDES:
            self.register_buffer(
                name=f'triples_{side.value}',
                tensor=dataset.graphs[side].triples,
            )
            self.register_buffer(
                name=f'dual_adjacency_{side.value}',
                tensor=get_relation_dual_graph_dense(knowledge_graph=dataset.graphs[side])
            )

        relation_dim = 2 * embedding_dim
        self.dual_attentions = nn.ModuleList([
            DenseAttentionLayer(input_dim=relation_dim, hidden_dim=relation_dim, act_func=nn.ReLU(), wu2019_init=wu2019_init)
            for _ in range(num_interactions)
        ])
        self.primal_attentions = nn.ModuleList([
            SparseAttentionLayer(input_dim=relation_dim, activation=nn.ReLU(), wu2019_init=wu2019_init)
            for _ in range(num_interactions)
        ])
        self.gcn_layers = nn.ModuleList([
            DiagLayer(dimension=embedding_dim, activation=nn.ReLU(), dropout=0.0)
            for _ in range(num_gcn_layers)
        ])
        self.highway_layers = nn.ModuleList([
            HighwayLayer(dimension=embedding_dim, wu2019_init=wu2019_init)
            for _ in range(num_gcn_layers)
        ])

        # Important: call this in the constructor.
        self.reset_parameters()

    def _get_relation_representation(
        self,
        entities: torch.FloatTensor,
        side: MatchSideEnum,
    ) -> torch.FloatTensor:
        r"""
        Get relation representations as concatenation of averaged head and tail embeddings.

        .. math ::
            r_i = \left[ \frac{\sum_{k \in H_i} e_k}{|H_i|} \| \frac{\sum_{k \in T_i} e_k}{|T_i|} \right]

        where :math:`H_i` denotes the set of head entities co-occurring with this relation, and :math:`T_i` analogously
        the set of tail entities. :math:`\|` denotes the vector concatenation.

        .. seealso ::
            https://www.ijcai.org/Proceedings/2019/0733.pdf, eq. (5)

        :param entities: shape: (num_entities, dim)
            The entity representations.
        :param side:
            The side.

        :return: shape: (num_relations, 2 * dim)
            The relation representation.
        """
        return torch.cat([
            self.head_to_rel[side] @ entities,
            self.tail_to_rel[side] @ entities,
        ], dim=-1)

    def forward(
        self,
        indices: Optional[Mapping[MatchSideEnum, EntityIDs]] = None,
    ) -> Mapping[MatchSideEnum, torch.FloatTensor]:  # noqa: D102
        result = dict()
        if indices is None:
            indices = dict()

        for side in SIDES:
            e = e_0 = self.embeddings[side](indices=None)
            r = None

            dual_A = getattr(self, f'dual_adjacency_{side.value}')
            triples = getattr(self, f'triples_{side.value}')

            # interactions
            for dual_att, primal_att in zip(self.dual_attentions, self.primal_attentions):
                c = self._get_relation_representation(entities=e, side=side)
                if r is None:
                    r = c
                r = dual_att(relation=r, context=r, adj_tensor=dual_A)
                e = primal_att(entities=e, relations=r, triples=triples)
                e = e_0 + self.beta[0] * e

            # gcn layers
            adjacency = self.reductions[side]().sparse_matrix
            for gcn, highway in zip(self.gcn_layers, self.highway_layers):
                y = gcn(x=e, adjacency=adjacency)
                e = highway(x=e, y=y)

            # If we are only interested in some entities, select them.
            ind = indices.get(side)
            if ind is not None:
                e = e[ind]

            result[side] = e

        return result
