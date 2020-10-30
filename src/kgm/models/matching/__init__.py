# coding=utf-8
"""Models for (knowledge) graph matching."""
from .base import GraphBasedKGMatchingModel, KGMatchingModel, PureEmbeddingModel, get_matching_model_by_name
from .dgmc import DGMCGCNAlign
from .rdgcn import RDGCN

__all__ = [
    'GraphBasedKGMatchingModel',
    'DGMCGCNAlign',
    'KGMatchingModel',
    'PureEmbeddingModel',
    'RDGCN',
    'get_matching_model_by_name',
]
