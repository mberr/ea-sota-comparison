# coding=utf-8
"""Various knowledge graph related data structures."""
import enum
import json
import logging
import lzma
import pathlib
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch

from ..utils.torch_utils import split_tensor
from ..utils.types import EntityIDs, IDAlignment, Triples

logger = logging.getLogger(__name__)

__all__ = [
    'EntityAlignment',
    'KnowledgeGraph',
    'KnowledgeGraphAlignmentDataset',
    'MatchSideEnum',
    'SIDES',
    'exact_self_alignment',
    'get_erdos_renyi',
    'get_other_side',
    'get_synthetic_math_graph',
    'sub_graph_alignment',
    'validation_split',
]


@enum.unique
class MatchSideEnum(str, enum.Enum):
    """The graph selection for a entity alignment dataset."""

    #: The left side
    left = 'left'

    #: The right side
    right = 'right'


# The canonical order of match sides
SIDES = (MatchSideEnum.left, MatchSideEnum.right)


def get_other_side(side: MatchSideEnum) -> MatchSideEnum:
    """Get the enum of the other side."""
    return MatchSideEnum.left if side == MatchSideEnum.right else MatchSideEnum.right


def add_self_loops(
    triples: Triples,
    entity_label_to_id: Mapping[str, int],
    relation_label_to_id: Mapping[str, int],
    self_loop_relation_name: Optional[str] = None,
) -> Tuple[Triples, Mapping[str, int]]:
    """Add self loops with dummy relation.

    For each entity e, add (e, self_loop, e).

    :param triples: shape: (n, 3)
         The triples.
    :param entity_label_to_id:
        The mapping from entity labels to ids.
    :param relation_label_to_id:
        The mapping from relation labels to ids.
    :param self_loop_relation_name:
        The name of the self-loop relation. Must not exist.

    :return:
        cat(triples, self_loop_triples)
        updated mapping
    """
    if self_loop_relation_name is None:
        self_loop_relation_name = 'self_loop'
    p = triples[:, 1]

    # check if name clashes might occur
    if self_loop_relation_name in relation_label_to_id.keys():
        raise AssertionError(f'There exists a relation "{self_loop_relation_name}".')

    # Append inverse relations to translation table
    max_relation_id = max(relation_label_to_id.values())
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    self_loop_relation_id = max_relation_id + 1
    updated_relation_label_to_id.update({self_loop_relation_name: self_loop_relation_id})
    assert len(updated_relation_label_to_id) == len(relation_label_to_id) + 1

    # create self-loops triples
    assert (p <= max_relation_id).all()
    e = torch.tensor(sorted(entity_label_to_id.values()), dtype=torch.long)  # pylint: disable=not-callable
    p_self_loop = torch.ones_like(e) * self_loop_relation_id
    self_loop_triples = torch.stack([e, p_self_loop, e], dim=1)

    all_triples = torch.cat([triples, self_loop_triples], dim=0)

    return all_triples, updated_relation_label_to_id


def add_inverse_triples(
    triples: Triples,
    relation_label_to_id: Mapping[str, int],
    inverse_relation_postfix: Optional[str] = None,
) -> Tuple[Triples, Mapping[str, int]]:
    """Create and append inverse triples.

    For each triple (s, p, o), an inverse triple (o, p_inv, s) is added.

    :param triples: shape: (n, 3)
        The triples.
    :param relation_label_to_id:
        The mapping from relation labels to ids.
    :param inverse_relation_postfix:
        A postfix to use for creating labels for the inverse relations.

    :return: cat(triples, inverse_triples)
    """
    if inverse_relation_postfix is None:
        inverse_relation_postfix = '_inv'
    assert len(inverse_relation_postfix) > 0

    s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]

    # check if name clashes might occur
    suspicious_relations = sorted(k for k in relation_label_to_id.keys() if k.endswith('_inv'))
    if len(suspicious_relations) > 0:
        raise AssertionError(
            f'Some of the inverse relations did already exist! Suspicious relations: {suspicious_relations}')

    # Append inverse relations to translation table
    num_relations = len(relation_label_to_id)
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    updated_relation_label_to_id.update({r_label + inverse_relation_postfix: r_id + num_relations for r_label, r_id in relation_label_to_id.items()})
    assert len(updated_relation_label_to_id) == 2 * num_relations

    # create inverse triples
    assert (p < num_relations).all()
    p_inv = p + num_relations
    inverse_triples = torch.stack([o, p_inv, s], dim=1)

    all_triples = torch.cat([triples, inverse_triples], dim=0)

    return all_triples, updated_relation_label_to_id


@dataclass
class KnowledgeGraph:
    """A knowledge graph, a multi-relational graph, represented by triples."""

    #: The triples, shape: (n, 3)
    triples: Triples

    #: The mapping from entity labels to IDs
    entity_label_to_id: Optional[Mapping[str, int]]

    #: The mapping from relations labels to IDs
    relation_label_to_id: Optional[Mapping[str, int]]

    #: Language code of the knowledge graph (e.g. zh, en, ...)
    lang_code: Optional[str] = None

    #: Dataset name
    dataset_name: Optional[str] = None

    #: Dataset subset name
    subset_name: Optional[str] = None

    #: Whether inverse triples have been added
    inverse_triples: bool = False

    #: Whether self-loops have been added.
    self_loops: bool = False

    @property
    def num_triples(self) -> int:
        """Return the number of triples."""
        return self.triples.shape[0]

    @property
    def num_entities(self) -> int:
        """Return the number of entities."""
        return len(set(self.entity_label_to_id.values()))

    @property
    def num_relations(self) -> int:
        """Return the number of relations."""
        return len(set(self.relation_label_to_id.values()))

    def with_inverse_triples(
        self,
        inverse_relation_postfix: Optional[str] = None,
    ) -> 'KnowledgeGraph':
        """Return a KG with added inverse triples, if not already contained. Otherwise return reference to self."""
        assert not self.self_loops
        if self.inverse_triples:
            return self
        else:
            enriched_triples, enriched_relation_label_to_id = add_inverse_triples(
                triples=self.triples,
                relation_label_to_id=self.relation_label_to_id,
                inverse_relation_postfix=inverse_relation_postfix,
            )
            return KnowledgeGraph(
                triples=enriched_triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=enriched_relation_label_to_id,
                inverse_triples=True,
                self_loops=False,
                lang_code=self.lang_code,
                dataset_name=self.dataset_name,
                subset_name=self.subset_name
            )

    def with_self_loops(
        self,
        self_loop_relation_name: Optional[str] = None,
    ) -> 'KnowledgeGraph':
        """Return a KG with added self-loops, if not already contained. Otherwise return reference to self."""
        if self.self_loops:
            return self
        else:
            enriched_triples, enriched_relation_label_to_id = add_self_loops(
                triples=self.triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=self.relation_label_to_id,
                self_loop_relation_name=self_loop_relation_name,
            )
            return KnowledgeGraph(
                triples=enriched_triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=enriched_relation_label_to_id,
                inverse_triples=self.inverse_triples,
                self_loops=True,
                lang_code=self.lang_code,
                dataset_name=self.dataset_name,
                subset_name=self.subset_name
            )

    def __str__(self):  # noqa: D105
        return f'{self.__class__.__name__}(num_triples={self.num_triples}, num_entities={self.num_entities}, num_relations={self.num_relations}, inverse_triples={self.inverse_triples}, self_loops={self.self_loops})'

    def get_relation_label_by_id(self, relation_id: int) -> Optional[str]:
        """Lookup a relation label for a given ID."""
        matches = [label for (label, id_) in self.relation_label_to_id.items() if id_ == relation_id]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise ValueError(f'More than one relation with ID {relation_id}')
        return matches[0]

    def save(self, directory: pathlib.Path) -> None:
        """Save the KG to a directory."""
        # ensure the directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # save triples
        torch.save(self.triples, directory / 'triples.pth')
        assert not self.inverse_triples
        assert not self.self_loops

        # save label-to-id
        with lzma.open(directory / 'metadata.json.xz', 'wt') as json_file:
            json.dump(
                obj=dict(
                    entity_label_to_id=self.entity_label_to_id,
                    relation_label_to_id=self.relation_label_to_id,
                    lang_code=self.lang_code,
                    dataset_name=self.dataset_name,
                    subset_name=self.subset_name,
                ),
                fp=json_file,
                sort_keys=True,
                indent=2,
            )

    @staticmethod
    def load(directory: pathlib.Path) -> 'KnowledgeGraph':
        """Load the KG from a directory."""
        triples = torch.load(directory / 'triples.pth')
        with lzma.open(directory / 'metadata.json.xz', 'r') as json_file:
            meta = json.load(json_file)
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=meta['entity_label_to_id'],
            relation_label_to_id=meta['relation_label_to_id'],
            lang_code=meta['lang_code'],
            dataset_name=meta['dataset_name'],
            subset_name=meta['subset_name'],
        )


@dataclass
class EntityAlignment:
    """An entity alignment between two knowledge graphs."""

    #: The entity alignment used for training, shape: (2, num_train_alignments)
    train: IDAlignment

    #: The entity alignment used for testing, shape: (2, num_test_alignments)
    test: IDAlignment

    #: The entity alignment used for validation, shape: (2, num_validation_alignments)
    _validation: Optional[IDAlignment] = None

    @property
    def validation(self) -> IDAlignment:
        """
        Return the validation alignment.

        :return: shape: (2, num_val_alignments), dtype=long
            The validation alignment.
        """
        if self._validation is None:
            return torch.empty(2, 0, dtype=torch.long, device=self.train.device)
        return self._validation

    @property
    def num_train(self) -> int:
        """Return the number of training alignment pairs."""
        return self.train.shape[1]

    @property
    def num_validation(self) -> int:
        """Return the number of validation alignment pairs."""
        return self.validation.shape[1]

    @property
    def num_test(self) -> int:
        """Return the number of test alignment pairs."""
        return self.test.shape[1]

    @property
    def all(self) -> IDAlignment:
        """
        Return the concatenation of all alignments parts.

        :return: shape: (2, num_total_alignments), dtype=long
            All alignments (train, validation, test)
        """
        return torch.cat([self.train, self.validation, self.test], dim=1)

    def to_dict(self) -> Mapping[str, IDAlignment]:
        """Convert the alignment to a dictionary with keys {'train', 'test'}, and optionally 'validation'."""
        return {
            key: value
            for key, value in zip(
                ('train', 'test', 'validation'),
                (self.train, self.test, self.validation)
            )
            if value.numel() > 0
        }

    def validation_split(self, train_ratio: float, seed: Optional[int] = None) -> 'EntityAlignment':
        """Return a new alignment object where the training alignments have been split to train, and validation."""
        if train_ratio <= 0. or train_ratio >= 1.:
            raise ValueError(f'ratio must be in (0, 1), but is {train_ratio}')
        return validation_split(alignment=self, train_ratio=train_ratio, seed=seed)

    def __str__(self):  # noqa: D105
        return f'{self.__class__.__name__}(num_train={self.num_train}, num_test={self.num_test}, num_val={self.num_validation})'

    @staticmethod
    def from_full_alignment(
        alignment: IDAlignment,
        train_test_split: Optional[float],
        train_validation_split: Optional[float],
        seed: Optional[int] = 42,
    ) -> 'EntityAlignment':
        """
        Create an entity alignment by splitting a given alignment tensor.

        If requested the alignment is first split into a train and test part. Afterwards, if requested, the train part
        is split to train and validation.

        :param alignment: shape: (2, total_num_alignments)
            The ID-based alignment.
        :param train_test_split:
            The train-test split ratio.
        :param train_validation_split:
            The train-validation split ratio.
        :param seed:
            The seed to be used for splitting.

        :return:
            An entity alignment.
        """
        if train_test_split is None:
            train_test_split = 1.
        if train_validation_split is None:
            train_validation_split = 1.
        test_train_split = 1. - train_test_split
        # pylint: disable=unbalanced-tuple-unpacking
        test, train, validation = split_tensor(alignment, ratios=[test_train_split, train_validation_split], shuffle=True, dim=1, seed=seed)
        return EntityAlignment(
            train=train,
            test=test,
            _validation=validation,
        )

    def __getitem__(self, item: str) -> IDAlignment:  # noqa: D105
        if item == 'train':
            return self.train
        elif item == 'test':
            return self.test
        elif item == 'validation':
            return self.validation
        else:
            raise KeyError(item)


class KnowledgeGraphAlignmentDataset:
    """A knowledge graph alignment data set, comprising a pair of graphs, and a (partial) alignment of their entities."""

    #: The first knowledge graph
    left_graph: KnowledgeGraph

    #: The second knowledge graph
    right_graph: KnowledgeGraph

    #: The alignment
    alignment: EntityAlignment

    def __init__(
        self,
        left_graph: KnowledgeGraph,
        right_graph: KnowledgeGraph,
        alignment: EntityAlignment,
    ):
        """
        Initialize the alignment dataset.

        :param left_graph:
            The left graph.
        :param right_graph:
            The right graph.
        :param alignment:
            The alignment between the graphs.
        """
        self.left_graph = left_graph
        self.right_graph = right_graph
        self.alignment = alignment

    def validation_split(self, train_ratio: float, seed: Optional[int] = None) -> 'KnowledgeGraphAlignmentDataset':
        """Return the dataset, where the training alignment part has been split into train and validation part."""
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph,
            right_graph=self.right_graph,
            alignment=self.alignment.validation_split(train_ratio=train_ratio, seed=seed),
        )

    @property
    def triples(self) -> Mapping[MatchSideEnum, Triples]:
        """Return a dictionary of the side to the corresponding triples on this side."""
        return {
            MatchSideEnum.left: self.left_graph.triples,
            MatchSideEnum.right: self.right_graph.triples,
        }

    @property
    def graphs(self) -> Mapping[MatchSideEnum, KnowledgeGraph]:
        """Return a dictionary of the side to KG on this side."""
        return {
            MatchSideEnum.left: self.left_graph,
            MatchSideEnum.right: self.right_graph,
        }

    @property
    def num_nodes(self) -> Mapping[MatchSideEnum, int]:
        """Return a dictionary of side to number of entities."""
        return {
            MatchSideEnum.left: self.left_graph.num_entities,
            MatchSideEnum.right: self.right_graph.num_entities,
        }

    @property
    def num_exclusives(self) -> Mapping[MatchSideEnum, int]:
        """Return a dictionary of side to number of exclusive nodes."""
        return {
            side: self.num_nodes[side] - len(set(aligned_on_side.tolist()))
            for side, aligned_on_side in zip(SIDES, self.alignment.all)
        }

    @property
    def exclusives(self) -> Mapping[MatchSideEnum, EntityIDs]:
        """Return a dictionary of side to ID of exclusive entities."""
        return {
            side: torch.as_tensor(
                data=sorted(set(range(self.graphs[side].num_entities)).difference(aligned_on_side.tolist())),
                dtype=torch.long,
            )
            for side, aligned_on_side in zip(
                [MatchSideEnum.left, MatchSideEnum.right],
                self.alignment.all,
            )
        }

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        return self.left_graph.dataset_name

    @property
    def subset_name(self) -> str:
        """Return the name of the subset."""
        return self.left_graph.subset_name

    def __str__(self):  # noqa: D105
        return f'{self.__class__.__name__}(left={self.left_graph}, right={self.right_graph}, align={self.alignment})'

    def with_inverse_triples(self) -> 'KnowledgeGraphAlignmentDataset':
        """Return the dataset where both sides have been extended by inverse triples."""
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph.with_inverse_triples(),
            right_graph=self.right_graph.with_inverse_triples(),
            alignment=self.alignment,
        )

    def with_self_loops(self) -> 'KnowledgeGraphAlignmentDataset':
        """Return the dataset where both sides have been extended by self-loops."""
        return KnowledgeGraphAlignmentDataset(
            left_graph=self.left_graph.with_self_loops(),
            right_graph=self.right_graph.with_self_loops(),
            alignment=self.alignment,
        )


def validation_split(
    alignment: EntityAlignment,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> EntityAlignment:
    """
    Split the train part of an entity alignment into train and validation.

    :param alignment:
        The alignment.
    :param train_ratio: 0 < x < 1
        The ratio of alignments to use for the train part.
    :param seed:
        The seed to use for randomisation.

    :return:
        An entity alignment with the updated train and validation part.
    """
    # Check input
    if not (0. < train_ratio < 1.):
        raise ValueError(f'train_ratio must be between 0 and 1, but is {train_ratio}')

    # re-combine train and validation, if already split
    num_total = alignment.num_train
    pool = alignment.train
    if alignment.num_validation > 0:
        num_total += alignment.num_validation
        pool = torch.cat([pool, alignment.validation], dim=1)

    # Delegate to tensor-based split.
    # pylint: disable=unbalanced-tuple-unpacking
    train_alignments, validation_alignments = split_tensor(tensor=pool, ratios=train_ratio, dim=1, seed=seed)

    # Construct new alignment object.
    return EntityAlignment(
        train=train_alignments,
        _validation=validation_alignments,
        test=alignment.test,
    )


def exact_self_alignment(
    graph: KnowledgeGraph,
    train_percentage: float = 0.5,
) -> KnowledgeGraphAlignmentDataset:
    """
    Create a alignment between a graph a randomly permuted version of it.

    :param graph: The graph.
    :param train_percentage: The percentage of training alignments.

    :return: A knowledge graph alignment dataset.
    """
    # Create a random permutation as alignment
    full_alignment = torch.stack([
        torch.arange(graph.num_entities, dtype=torch.long),
        torch.randperm(graph.num_entities)
    ], dim=0)

    # shuffle
    full_alignment = full_alignment[:, torch.randperm(graph.num_entities)]

    # create mapping
    mapping = {int(a): int(b) for a, b in full_alignment.t()}

    # translate triples
    h, r, t = graph.triples.t()
    h_new, t_new = [torch.tensor([mapping[int(e)] for e in es], dtype=torch.long) for es in (h, t)]  # pylint: disable=not-callable
    r_new = r.detach().clone()
    new_triples = torch.stack([h_new, r_new, t_new], dim=-1)

    # compose second KG
    second_graph = KnowledgeGraph(
        triples=new_triples,
        entity_label_to_id={k: mapping[v] for k, v in graph.entity_label_to_id.items()},
        relation_label_to_id=graph.relation_label_to_id.copy(),
        inverse_triples=False,
        self_loops=False,
    )
    second_graph.inverse_triples = graph.inverse_triples
    second_graph.self_loops = graph.self_loops

    # split alignment
    split_id = int(train_percentage * graph.num_entities)
    alignment = EntityAlignment(
        train=full_alignment[:, :split_id],
        test=full_alignment[:, split_id:],
    )

    return KnowledgeGraphAlignmentDataset(
        left_graph=graph,
        right_graph=second_graph,
        alignment=alignment,
    )


def sub_graph_alignment(
    graph: KnowledgeGraph,
    overlap: float = 0.5,
    ratio: float = 0.7,
    train_test_split: float = 0.5,
    train_validation_split: Optional[float] = 0.8,
) -> KnowledgeGraphAlignmentDataset:
    """
    Create a synthetic entity alignment dataset, where both sides are random subgraphs from a larger one.

    :param graph:
        The source KG.
    :param overlap:
        The percentage of overlapping entities.
    :param ratio:
        The ratio of entities between the two KG.
    :param train_test_split:
        The ratio for train-test splitting the aligned entities.

    :return:
        A entity alignment dataset.
    """
    # split entities
    entities = torch.arange(graph.num_entities)
    # pylint: disable=unbalanced-tuple-unpacking
    common, left, right = split_tensor(tensor=entities, ratios=[overlap, ratio])
    left = torch.cat([common, left])
    right = torch.cat([common, right])

    # create alignment
    alignment = EntityAlignment.from_full_alignment(
        alignment=torch.arange(common.shape[0]).unsqueeze(dim=0).repeat(2, 1),
        train_test_split=train_test_split,
        train_validation_split=train_validation_split,
    )

    # induced subgraph
    graphs = []
    for ent in [left, right]:
        ent = set(ent.tolist())
        entity_label_to_id = {
            str(old_id): new_id
            for new_id, old_id in enumerate(ent)
        }
        triples = torch.as_tensor(data=[
            (entity_label_to_id[str(h)], r, entity_label_to_id[str(t)])
            for h, r, t in graph.triples.tolist()
            if (h in ent and t in ent)
        ], dtype=torch.long)
        graphs.append(KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=graph.relation_label_to_id,
        ))

    return KnowledgeGraphAlignmentDataset(
        left_graph=graphs[0],
        right_graph=graphs[1],
        alignment=alignment,
    )


def get_erdos_renyi(
    num_entities: int,
    num_relations: int,
    num_triples: int,
) -> KnowledgeGraph:
    """
    Generate a synthetic KG using Erdos-Renyi, and random edge typing.

    :param num_entities: >0
        The number of entities.
    :param num_relations: >0
        The number of relations.
    :param num_triples:
        The number of triples. If present, ignore p.

    :return:
        A KG.
    """
    head, tail = torch.randint(num_entities, size=(2, num_triples))
    relation = torch.randint(num_relations, size=(num_triples,))
    triples = torch.stack([head, relation, tail], dim=1)
    return KnowledgeGraph(
        triples=triples,
        entity_label_to_id={str(i): i for i in range(num_entities)},
        relation_label_to_id={str(i): i for i in range(num_relations)},
        dataset_name='erdos_renyi',
        subset_name=f'{num_entities}-{num_relations}-{num_triples}',
    )


def get_synthetic_math_graph(
    num_entities: int,
) -> KnowledgeGraph:
    """
    Generate a synthetic KG of positive integers, linked by modulo relations.

    :param num_entities:
        The number of entities.

    :return:
        A KG.
    """
    entities = list(range(num_entities))
    relations = list(range(num_entities))
    triples = [(e, r, (e + r) % num_entities) for r in relations for e in entities]
    return KnowledgeGraph(
        triples=torch.as_tensor(triples, dtype=torch.long),
        entity_label_to_id={str(e): e for e in entities},
        relation_label_to_id={'+' + str(r): r for r in relations},
    )
