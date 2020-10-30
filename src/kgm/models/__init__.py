# coding=utf-8
"""Entity Alignment and Link Prediction Models."""
from .matching import GraphBasedKGMatchingModel, KGMatchingModel, PureEmbeddingModel, get_matching_model_by_name

__all__ = [
    'GraphBasedKGMatchingModel',
    'KGMatchingModel',
    'PureEmbeddingModel',
    'get_matching_model_by_name',
]
