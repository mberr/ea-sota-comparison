# coding=utf-8
"""Loss functions for entity alignment and link prediction."""
import enum
import logging
from typing import Any, Callable, Mapping, Optional, Union

import torch
from torch import nn
from torch.nn import functional

from .similarity import Similarity
from ..data import MatchSideEnum, SIDES
from ..utils.common import get_subclass_by_name
from ..utils.types import IDAlignment, NodeIDs

logger = logging.getLogger(name=__name__)

__all__ = [
    'BaseLoss',
    'ContrastiveLoss',
    'MarginLoss',
    'MatchingLoss',
    'SampledMatchingLoss',
    'get_matching_loss',
    'get_pairwise_loss',
]


# pylint: disable=abstract-method
class BaseLoss(nn.Module):
    """Abstract class for losses on similarity matrices."""

    # pylint: disable=arguments-differ
    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:
        r"""
        Efficiently compute loss values from a similarity matrix.

        .. math::
            \frac{1}{n(m-1))} \sum_{b=1}^{n} \sum_{j \neq true[b]} pairloss(sim[b, true[b]], sim[b, j])

        :param similarities: shape: (n, m)
            A batch of similarity values.
        :param true_indices: shape (n,)
            The index of the unique true choice in each batch.
        """
        raise NotImplementedError


class MarginLoss(BaseLoss):
    r"""Evaluate a margin based loss.

     In particular the following form is used:

    .. math::
        baseloss(pos\_sim, neg\_sim) = g(neg\_sim + margin - pos\_sim)

    where g is an activation function, e.g. ReLU leading to the classical margin loss formulation.
    """

    def __init__(
        self,
        margin: float = 1.0,
        exact_loss_value: bool = False,
        activation: Callable[[torch.FloatTensor], torch.FloatTensor] = functional.relu,
    ):
        """
        Initialize the loss.

        :param margin: >0
             The margin which should be between positive and negative similarity values.
        :param exact_loss_value:
            Can be disabled to compute the loss up to a constant additive term for improved performance.
        :param activation:
            The activation function to use. Typical examples:
                - hard margin: torch.functional.relu
                - soft margin: torch.functional.softplus
        """
        super().__init__()
        self.margin = margin
        self.exact_loss_value = exact_loss_value
        self.activation = activation

    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        batch_size, num_choices = similarities.shape
        batch_indices = torch.arange(batch_size, device=similarities.device)
        pos_sim = similarities[batch_indices, true_indices].unsqueeze(dim=1)
        # as pos_sim + margin - pos_sim = margin, there is no gradient for comparison of positives with positives
        # as there are num_choices elements per row, with one positive, and (num_choices-1) negatives, we need to subtract
        # (margin/num_choices) to compensate for that in the loss value.
        # As this is a constant, the gradient is the same as if we would not add it, hence we only do it, if explicitly requested.
        loss_value = self.activation(similarities + self.margin - pos_sim).mean()
        if self.exact_loss_value:
            loss_value = loss_value - (self.activation(torch.as_tensor(data=self.margin, dtype=torch.float, device=loss_value.device)) / num_choices)
        return loss_value


class ContrastiveLoss(BaseLoss):
    """Similarity-based loss function, to pull positive samples together and negative samples apart."""

    def __init__(
        self,
        positive_margin: float = 1.0,
        negative_margin: float = 0.01,
        normalize_negatives: bool = True,
        balance_positive_negatives: Optional[float] = None,
        activation: Callable[[torch.FloatTensor], torch.FloatTensor] = functional.relu,
    ):
        """
        Initialize the loss.

        :param positive_margin:
            The positive margin. Similarity values for positive pairs should be above.
        :param negative_margin:
            The negative margin. Similarity values for negative pairs should be below. Should be at most the positive margin.
        :param normalize_negatives:
            Whether to normalize the loss values of negative pairs by their number.
        :param balance_positive_negatives:
            An optional weight parameter to balance the influence of positive and negative samples. Defaults to 0.5,
            i.e. equal weighting. It is highly recommended to combine it with normalize_negatives=True
        :param activation:
            The activation function to use. Typical examples:
                - hard margin: torch.functional.relu
                - soft margin: torch.functional.softplus
        """
        super().__init__()

        # Parameter validation
        if positive_margin < negative_margin:
            raise ValueError(f'Positive margin ({positive_margin}) must not be smaller than the negative margin ({negative_margin}).')

        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.normalize_negatives = normalize_negatives
        if balance_positive_negatives is not None:
            if not (0. <= balance_positive_negatives <= 1.):
                raise ValueError(f'If balance_positive_negatives is provided, it has to be between 0 and 1 (both inclusive), but is balance_positive_negatives={balance_positive_negatives}.')
            if not normalize_negatives:
                logger.warning('Providing balance_positive_negatives without setting normalize_negatives=True. This is not recommended.')
        self.balance_positive_negatives = balance_positive_negatives
        self.activation = activation

    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        batch_size, num_choices = similarities.shape
        pos_sim = similarities[torch.arange(batch_size, device=similarities.device), true_indices].unsqueeze(dim=0)
        pos_loss = self.activation(self.positive_margin - pos_sim).sum()
        neg_loss = self.activation(similarities - self.negative_margin).sum() - self.activation(pos_sim - self.negative_margin).sum()

        negative_weight = 1.

        # Normalize negative loss value by the number of negative pairs.
        if self.normalize_negatives:
            negative_weight = 1. / (num_choices - 1)

        # Balance contribution of positive and negative samples
        if self.balance_positive_negatives is not None:
            pos_loss = pos_loss * self.balance_positive_negatives
            negative_weight *= (1. - self.balance_positive_negatives)

        # apply weights (post-poned for increased efficiency)
        if negative_weight != 1.:
            neg_loss = neg_loss * negative_weight

        # normalize by batch_size
        return (pos_loss + neg_loss) / batch_size


class BCELoss(BaseLoss):
    """Point-wise classification loss."""

    def __init__(self, balance: bool = False):
        """
        Initialize the loss module.

        :param balance:
            Whether to balance the loss to account for the larger number of negative samples.
        """
        super().__init__()
        self.balance = balance

    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        target = torch.zeros_like(similarities)
        target[torch.arange(true_indices.shape[0], device=similarities.device), true_indices] = 1.0
        if self.balance:
            pos_weight = torch.as_tensor(data=[similarities.shape[1] - 1], dtype=similarities.dtype, device=similarities.device)
        else:
            pos_weight = None
        return functional.binary_cross_entropy_with_logits(input=similarities, target=target, pos_weight=pos_weight)


class CrossEntropyFromSoftmaxLoss(BaseLoss):
    """Cross entropy loss on softmax output."""

    def forward(self, similarities: torch.FloatTensor, true_indices: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        return functional.cross_entropy(input=similarities, target=true_indices)


@enum.unique
class LossDirectionEnum(str, enum.Enum):
    """An enum for specification of the direction of a matching loss."""

    #: Loss is matching entities from a left graph to a right one
    left_to_right = 'left_to_right'

    #: Loss is matching entities from a right graph to a left one
    right_to_left = 'right_to_left'

    #: Loss is averaging loss of matching entities from a left to a right graph and from the right to the left one
    symmetrical = 'symmetrical'


@enum.unique
class ActivationEnum(str, enum.Enum):
    #: ReLU Activation Function
    relu = 'relu'

    #: Softplus Activation Function
    softplus = 'softplus'


# pylint: disable=abstract-method
class MatchingLoss(nn.Module):
    """An API for graph matching losses."""

    #: The similarity
    similarity: Similarity

    #: The direction in which to compute the loss
    loss_direction: LossDirectionEnum

    def __init__(
        self,
        similarity: Similarity,
        loss_direction: LossDirectionEnum = LossDirectionEnum.symmetrical,
    ):
        """
        Initialize the loss.

        :param similarity:
            The similarity to use for comparing node representations.
        :param loss_direction:
            Defines a direction of matching, which loss is optimized during training
        """
        super().__init__()
        self.similarity = similarity
        self.loss_direction = loss_direction

    # pylint: disable=arguments-differ
    def forward(
        self,
        alignment: IDAlignment,
        representations: Mapping[MatchSideEnum, torch.FloatTensor],
        negatives: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute the loss.

        :param alignment: shape: (2, num_aligned)
            The aligned nodes in form of node ID pairs.
        :param representations:
            side -> repr, where repr is a tensor of shape (num_nodes_side, dim)
        :param negatives: shape: (2, num_aligned, num_negatives)
            Negative samples. negatives[0] has to be combined with alignment[1] for a valid pair.
        """
        partial_losses = []

        # left-to-right loss
        if self.loss_direction in {LossDirectionEnum.left_to_right, LossDirectionEnum.symmetrical}:
            source_side, target_side = SIDES
            partial_losses.append(
                self._one_side_matching_loss(
                    source=representations[source_side],
                    target=representations[target_side],
                    alignment=alignment,
                    negatives=None if negatives is None else negatives[1],
                )
            )

        # right-to-left loss
        if self.loss_direction in {LossDirectionEnum.right_to_left, LossDirectionEnum.symmetrical}:
            target_side, source_side = SIDES
            partial_losses.append(
                self._one_side_matching_loss(
                    source=representations[source_side],
                    target=representations[target_side],
                    alignment=alignment.flip(0),
                    negatives=None if negatives is None else negatives[0],
                )
            )

        assert len(partial_losses) > 0
        return sum(partial_losses) / len(partial_losses)

    def _one_side_matching_loss(
        self,
        source: torch.FloatTensor,
        target: torch.FloatTensor,
        alignment: IDAlignment,
        negatives: Optional[NodeIDs]
    ) -> torch.FloatTensor:
        """
        Compute the loss from selected nodes in source graph to the other graph.

        :param source: shape: (num_source, dim)
            Source node representations.
        :param target: shape: (num_target, dim)
            Target node representations.
        :param alignment: shape: (2, num_aligned)
            The alignment.
        :param negatives: shape: (num_aligned, num_negatives)
            The negative examples from target side.
        """
        raise NotImplementedError


class SampledMatchingLoss(MatchingLoss):
    """Apply a base loss to a similarity matrix where negative samples are used to reduce memory footprint."""

    #: The base loss
    base_loss: BaseLoss

    #: The number of negative samples
    num_negatives: int

    #: Whether to use self-adversarial weighting
    self_adversarial_weighting: bool

    def __init__(
        self,
        similarity: Similarity,
        base_loss: BaseLoss,
        loss_direction: LossDirectionEnum = LossDirectionEnum.symmetrical,
        num_negatives: int = 1,
        self_adversarial_weighting: bool = False,
    ):
        """
        Initialize the loss.

        :param similarity:
            The similarity to use for computing the similarity matrix.
        :param base_loss:
            The base loss to apply to the similarity matrix.
        :param num_negatives:
            The number of negative samples for each positive pair.
        :param self_adversarial_weighting:
            Whether to apply self-adversarial weighting.
        """
        super().__init__(
            similarity=similarity,
            loss_direction=loss_direction
        )
        self.base_loss = base_loss
        self.num_negatives = num_negatives
        self.self_adversarial_weighting = self_adversarial_weighting

    def _one_side_matching_loss(
        self,
        source: torch.FloatTensor,
        target: torch.FloatTensor,
        alignment: IDAlignment,
        negatives: Optional[NodeIDs],
    ) -> torch.FloatTensor:  # noqa: D102
        # Split mapping
        source_ind, target_ind_pos = alignment

        # Extract representations, shape: (batch_size, dim)
        anchor = source[source_ind]

        # Positive scores
        pos_scores = self.similarity.one_to_one(left=anchor, right=target[target_ind_pos])

        # Negative samples in target graph, shape: (batch_size, num_negatives)
        if negatives is None:
            negatives = torch.randint(
                target.shape[0],
                size=(target_ind_pos.shape[0], self.num_negatives),
                device=target.device,
            )

        # Negative scores, shape: (batch_size, num_negatives, dim)
        neg_scores = self.similarity.one_to_one(left=anchor.unsqueeze(1), right=target[negatives])

        # self-adversarial weighting as described in RotatE paper: https://arxiv.org/abs/1902.10197
        if self.self_adversarial_weighting:
            neg_scores = functional.softmax(neg_scores, dim=1).detach() * neg_scores

        # Evaluate base loss
        return self.base_loss(
            similarities=torch.cat([pos_scores.unsqueeze(dim=-1), neg_scores], dim=-1),
            true_indices=torch.zeros_like(target_ind_pos),
        ).mean()


def matching_loss_name_normalizer(name: str) -> str:
    """Normalize the class name of a MatchingLoss."""
    return name.lower().replace('matchingloss', '')


def base_loss_name_normalizer(name: str) -> str:
    """Normalize the class name of a base BaseLoss."""
    return name.lower().replace('loss', '')


def get_pairwise_loss(name: str, **kwargs: Any) -> BaseLoss:
    """
    Get a pairwise loss by class name.

    :param name:
        The name of the class.
    :param kwargs:
        Additional key-word based constructor arguments.

    :return:
        The base loss instance.
    """
    pairwise_loss_cls = get_subclass_by_name(base_class=BaseLoss, name=name, normalizer=base_loss_name_normalizer)
    pairwise_loss = pairwise_loss_cls(**kwargs)
    return pairwise_loss


def get_matching_loss(name: str, similarity: Similarity, **kwargs) -> MatchingLoss:
    """
    Get a matching loss by class name.

    :param name:
        The name of the class.
    :param similarity:
        The similarity to use.
    :param kwargs:
        Additional key-word based constructor arguments.

    :return:
        The matching loss instance.
    """
    matching_loss_cls = get_subclass_by_name(base_class=MatchingLoss, name=name, normalizer=matching_loss_name_normalizer)
    matching_loss = matching_loss_cls(similarity=similarity, **kwargs)
    return matching_loss


def get_activation(activation_name: str) -> Union[functional.relu, functional.softplus]:
    """
    Get an activation function by name.

    :param activation_name:
        the name of the activation function

    :return:
        The chosen activation function.
    """
    if activation_name == ActivationEnum.relu:
        activation = functional.relu
    elif activation_name == ActivationEnum.softplus:
        activation = functional.softplus
    else:
        raise ValueError(f'Activation function currently not supported: {activation_name}')

    return activation
