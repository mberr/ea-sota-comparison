# coding=utf-8
"""Components for building and training models."""
from .losses import BaseLoss, ContrastiveLoss, MarginLoss, MatchingLoss, SampledMatchingLoss, get_matching_loss, get_pairwise_loss, get_activation
from .similarity import BoundInverseTransformation, CosineSimilarity, DistanceToSimilarity, DotProductSimilarity, LpSimilarity, NegativeTransformation, Similarity, SimilarityEnum, get_similarity

__all__ = [
    'BoundInverseTransformation',
    'ContrastiveLoss',
    'CosineSimilarity',
    'DistanceToSimilarity',
    'DotProductSimilarity',
    'get_matching_loss',
    'get_pairwise_loss',
    'get_activation',
    'get_similarity',
    'LpSimilarity',
    'MarginLoss',
    'MatchingLoss',
    'NegativeTransformation',
    'BaseLoss',
    'SampledMatchingLoss',
    'Similarity',
    'SimilarityEnum',
]
