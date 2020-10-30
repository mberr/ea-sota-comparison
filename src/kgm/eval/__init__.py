# coding=utf-8
"""Evaluation methods."""
from .common import compute_ranks
from .matching import evaluate_alignment, evaluate_matching_model

__all__ = [
    'compute_ranks',
    'evaluate_alignment',
    'evaluate_matching_model',
]
