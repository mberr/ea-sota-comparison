# coding=utf-8
"""Data loading and representation."""
from .knowledge_graph import (
    EntityAlignment,
    KnowledgeGraph,
    KnowledgeGraphAlignmentDataset,
    MatchSideEnum,
    SIDES,
    exact_self_alignment,
    get_erdos_renyi,
    get_other_side,
    get_synthetic_math_graph,
    sub_graph_alignment,
    validation_split,
)
from .loaders import available_datasets, get_dataset_by_name, get_dataset_loader_by_name

__all__ = [
    'EntityAlignment',
    'KnowledgeGraph',
    'KnowledgeGraphAlignmentDataset',
    'MatchSideEnum',
    'SIDES',
    'available_datasets',
    'exact_self_alignment',
    'get_dataset_by_name',
    'get_dataset_loader_by_name',
    'get_erdos_renyi',
    'get_other_side',
    'get_synthetic_math_graph',
    'sub_graph_alignment',
    'validation_split',
]
