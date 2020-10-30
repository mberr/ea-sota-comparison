"""Common utility methods for evaluation."""
import logging
from typing import Collection, Mapping, Optional

import torch

logger = logging.getLogger(name=__name__)

# Small constant for floating point comparison
EPSILON = 1.0e-08


def get_rank(sim: torch.FloatTensor, true: torch.LongTensor) -> torch.FloatTensor:
    """Compute the rank, exploiting that there is only one true hit."""
    batch_size = true.shape[0]
    true_sim = sim[torch.arange(batch_size), true].unsqueeze(1)
    best_rank = torch.sum(sim > true_sim, dim=1, dtype=torch.long).float() + 1
    worst_rank = torch.sum(sim >= true_sim, dim=1, dtype=torch.long).float()
    return 0.5 * (best_rank + worst_rank)


def compute_ranks(
    scores: torch.FloatTensor,
    true_indices: torch.LongTensor,
    smaller_is_better: bool = True,
    mask: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    """Compute the rank of the true hit.

    :param scores: shape: (k, n)
        The scores for each sample.
    :param true_indices: shape: (k,)
        Values between 0 (incl.) and n (excl.)
    :param smaller_is_better:
        Whether smaller of larger values are better.
    :param mask: shape: (m, 2), optional
        Optional mask for filtered setting
    :return: shape: (k,)
        The ranks, a number between 1 and n.
    """
    # Ensure that larger is better
    if smaller_is_better:
        scores = -scores

    # Get the scores of the currently considered true entity.
    batch_size = scores.shape[0]
    true_score = (scores[torch.arange(0, batch_size), true_indices.flatten()]).view(-1, 1)

    # The best rank is the rank when assuming all options with an equal score are placed behind the currently
    # considered. Hence, the rank is the number of options with better scores, plus one, as the rank is one-based.
    best_rank = (scores > true_score).sum(dim=1) + 1

    # The worst rank is the rank when assuming all options with an equal score are placed in front of the currently
    # considered. Hence, the rank is the number of options which have at least the same score minus one (as the
    # currently considered option in included in all options). As the rank is one-based, we have to add 1, which
    # nullifies the "minus 1" from before.
    worst_rank = (scores >= true_score).sum(dim=1)

    # The average rank is the average of the best and worst rank, and hence the expected rank over all permutations of
    # the elements with the same score as the currently considered option.
    # We use the double average rank to avoid precision loss due to floating point operations.
    double_avg_rank = best_rank + worst_rank

    # In filtered setting ranking another true entity higher than the currently considered one should not be punished.
    # Hence, an adjustment is computed, which is the number of other true entities ranked higher. This adjustment is
    # subtracted from the rank.
    if mask is not None:
        batch_indices, entity_indices = mask.t()
        true_scores = true_score[batch_indices, 0]
        other_true_scores = scores[batch_indices, entity_indices]
        double_other_true_in_front = -2 * (other_true_scores > true_scores).long()
        double_avg_rank.index_add_(dim=0, index=batch_indices, source=double_other_true_in_front)

    avg_rank = 0.5 * double_avg_rank.float()

    return avg_rank


def aggregate_ranks(
    ranks: torch.FloatTensor,
    emr: float,
    ks: Collection[int] = (1, 10, 50, 100),
) -> Mapping[str, float]:
    """
    Compute rank aggregation metrics.

    :param ranks:
        The individual ranks.
    :param emr:
        The expected mean rank.
    :param ks:
        The values for which to compute Hits@k.

    :return:
        A dictionary
        {
            'mean_rank': The mean rank.
            'amr': The adjusted mean rank.
            'mrr': The mean reciprocal rank.
            'hits_at_k': Hits@k for each provided k.
        }
    """
    mr = torch.mean(ranks).item()
    result = dict(
        num_rank=ranks.numel(),
        mean_rank=mr,
        median_rank=torch.median(ranks).item(),
        std_rank=ranks.std(unbiased=True).item(),
        adjusted_mean_rank=mr / emr,
        adjusted_mean_rank_index=1 - (mr - 1) / (emr - 1) if emr > 1.0 else 0.0,
        mean_reciprocal_rank=torch.mean(torch.reciprocal(ranks)).item(),
    )
    result.update({
        f'hits_at_{k}': torch.mean((ranks <= (k + EPSILON)).float()).item()
        for k in ks
    })
    return result
