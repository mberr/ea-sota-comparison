# coding=utf-8
"""Modules for embeddings."""
from .base import get_embedding_pair
from .init.base import PretrainedNodeEmbeddingInitializer

__all__ = [
    'PretrainedNodeEmbeddingInitializer',
    'get_embedding_pair',
]
