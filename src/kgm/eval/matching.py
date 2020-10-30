# coding=utf-8
"""Entity Alignment evaluation methods."""
from typing import Collection, Dict, Mapping, Optional, Tuple, TypeVar, Union

import torch

from .common import aggregate_ranks, get_rank
from ..data import MatchSideEnum, SIDES
from ..models import KGMatchingModel
from ..modules import Similarity
from ..utils.torch_utils import maximize_memory_utilization
from ..utils.types import IDAlignment

__all__ = [
    'evaluate_matching_model',
    'evaluate_alignment',
]

T = TypeVar('T')


def evaluate_matching_model(
    model: KGMatchingModel,
    alignments: Mapping[T, IDAlignment],
    similarity: Similarity,
    eval_batch_size: Optional[int] = None,
    ks: Collection[int] = (1, 10, 50, 100),
) -> Tuple[Mapping[T, Mapping[str, float]], int]:
    """Evaluate a model on multiple alignments.

    :param model:
        The KG matching model to evaluate.
    :param alignments:
        A mapping of key -> alignment, where alignment is a LongTensor of shape (2, num_alignments).
    :param similarity:
        The similarity.
    :param eval_batch_size:
        The evaluation batch size.
    :param ks:
        The values for which to evaluate hits@k.

    :return:
        A mapping key -> subresult, where subresult is a mapping from metric-name to metric value.
    """
    # Evaluation
    with torch.no_grad():
        # Set model in evaluation mode
        model.eval()

        result = {}
        safe_eval_batch_size = None
        for key, alignment in alignments.items():
            alignment, indices = _reduce_alignment(alignment=alignment)
            partial_repr = model.get_node_representations(indices=indices)
            partial_result, this_eval_batch_size = evaluate_alignment(
                similarity=similarity,
                alignment=alignment,
                representations=partial_repr,
                eval_batch_size=eval_batch_size,
                ks=ks,
            )
            result[key] = partial_result
            if this_eval_batch_size is not None:
                if safe_eval_batch_size is None:
                    safe_eval_batch_size = this_eval_batch_size
                else:
                    safe_eval_batch_size = min(safe_eval_batch_size, this_eval_batch_size)
        assert safe_eval_batch_size is not None

    return result, safe_eval_batch_size


def evaluate_alignment(
    similarity: Similarity,
    alignment: IDAlignment,
    representations: Mapping[MatchSideEnum, torch.FloatTensor],
    eval_batch_size: Optional[int] = None,
    ks: Collection[int] = (1, 10, 50, 100),
) -> Tuple[Dict[str, float], int]:
    """
    Evaluate an alignment.

    :param representations: side -> repr
        The node representations, a tensor of shape (num_nodes[side], d).
    :param alignment: shape: (2, num_alignments)
        The alignment.
    :param similarity:
        The similarity.
    :param eval_batch_size: int (positive, optional)
        The batch size to use for evaluation.
    :param ks:
        The values for which to compute hits@k.
    :return: A tuple with
        1) dictionary with keys 'mr, 'mrr', 'hits_at_k' for all k in ks.
        2) The maximum evaluation batch size.
    """
    num_alignments = alignment.shape[1]
    if num_alignments <= 0:
        return dict(), None

    node_repr = dict()
    for side, alignment_on_side in zip(SIDES, alignment):
        repr_on_side = representations[side]
        node_repr[side] = repr_on_side[alignment_on_side.to(repr_on_side.device)]
    left, right = [representations[side] for side in SIDES]

    # Ensure data is on correct device
    right, alignment = [t.to(device=left.device) for t in (right, alignment)]

    if eval_batch_size is None:
        eval_batch_size = num_alignments
    return maximize_memory_utilization(
        _evaluate_alignment,
        parameter_name='eval_batch_size',
        parameter_max_value=eval_batch_size,
        alignment=alignment,
        similarity=similarity,
        left=left,
        right=right,
        ks=ks,
    )


def _summarize_ranks(
    ranks: torch.LongTensor,
    n: Union[int, Tuple[int, int]],
    ks: Collection[int],
) -> Dict[str, float]:
    if isinstance(n, int):
        n = (n, n)
    # overall
    result = dict(aggregate_ranks(
        ranks=ranks,
        emr=(sum(n) / 2 + 1) / 2,
        ks=ks,
    ))
    # side-specific
    for i, side in enumerate(SIDES):
        result[side.value] = aggregate_ranks(
            ranks=ranks[i],
            emr=(n[i] + 1) / 2,
            ks=ks,
        )
    return result


def _evaluate_alignment(
    eval_batch_size: int,
    alignment: IDAlignment,
    similarity: Similarity,
    left: torch.FloatTensor,
    right: torch.FloatTensor,
    ks: Collection[int],
) -> Dict[str, float]:
    """Evaluate an entity alignment.

    :param eval_batch_size:
        The evaluation batch size.
    :param alignment: shape: (2, num_alignments)
        The alignment.
    :param similarity:
        The similarity.
    :param left: shape: (num_left, dim)
        The left aligned representations.
    :param right: shape: (num_right, dim)
        The right aligned representations.
    :param ks:
        The values for which to calculate Hits@k.

    :return:
        The evaluation results as dictionary.
    """
    num_alignments = alignment.shape[1]
    ranks = left.new_empty(2, num_alignments)
    for i in range(0, num_alignments, eval_batch_size):
        batch = alignment[:, i:i + eval_batch_size]

        # match a batch of right nodes to all left nodes
        sim_right_to_all_left = similarity.all_to_all(left, right[batch[1]]).t()
        ranks[0, i:i + eval_batch_size] = get_rank(sim=sim_right_to_all_left, true=batch[0])

        # match a batch of left nodes to all right nodes
        sim_left_to_all_right = similarity.all_to_all(left[batch[0]], right)
        ranks[1, i:i + eval_batch_size] = get_rank(sim=sim_left_to_all_right, true=batch[1])

    num_nodes = [n.shape[0] for n in (left, right)]
    return _summarize_ranks(ranks=ranks, n=num_nodes, ks=ks)


def _reduce_alignment(alignment: IDAlignment) -> Tuple[IDAlignment, Mapping[MatchSideEnum, torch.LongTensor]]:
    indices = dict()
    local_alignment = []
    for side, alignment_on_side in zip(SIDES, alignment):
        uniq, inverse = torch.unique(alignment_on_side, sorted=False, return_inverse=True)
        indices[side] = uniq
        local_alignment.append(inverse)
    alignment = torch.stack(local_alignment, dim=0)
    return alignment, indices
