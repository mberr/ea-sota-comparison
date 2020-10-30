# coding=utf-8
"""Training loops for KG matching models."""
import logging
from abc import abstractmethod
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.utils import data

from .base import BaseTrainer, TrainerCallback
from ..data import KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ..eval import evaluate_matching_model
from ..models import KGMatchingModel
from ..modules import MatchingLoss, Similarity
from ..utils.torch_utils import maximize_memory_utilization
from ..utils.types import IDAlignment, NodeIDs

logger = logging.getLogger(name=__name__)


class AlignmentTrainerCallback(TrainerCallback):
    """Abstract class for trainer callbacks."""

    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: 'AlignmentModelTrainer') -> None:
        """
        Perform actions before the epoch starts.

        :param epoch:
            The epoch.
        :param trainer:
            The trainer.
        """
        raise NotImplementedError


class NodeSampler:
    """Abstract class for node sampler."""

    @abstractmethod
    def sample(
        self,
        positive_batch: IDAlignment,
    ) -> NodeIDs:
        """
        Sample negative node indices for each side.

        positive pair:
            (positive_batch[0, i], positive_batch[1, i])
        negative_pair:
            (positive_batch[0, i], negative_batch[0, i, j])

        :param positive_batch: shape: (2, pos_batch_size)
             The batch of aligned nodes.

        :return: shape: (2, pos_batch_size, num_negatives)
            The negative node IDs. result[0] has to be combined with positive_batch[1] for a valid pair.
        """
        raise NotImplementedError


class RandomNodeSampler(NodeSampler):
    """Randomly select additional nodes."""

    def __init__(
        self,
        num_nodes: Mapping[MatchSideEnum, int],
        num_negatives: int,
    ):
        """
        Initialize the sampler.

        :param num_nodes:
            The number of nodes on each side.
        :param num_negatives: >=0
            The absolute number of negatives samples for each positive one.
        """
        self.num_nodes = num_nodes
        self.num_negatives = num_negatives

    def sample(
        self,
        positive_batch: IDAlignment,
    ) -> NodeIDs:  # noqa: D102
        return torch.stack([
            torch.randint(self.num_nodes[side], size=(positive_batch.shape[1], self.num_negatives))
            for side in SIDES
        ], dim=0)


@torch.no_grad()
def _all_knn(
    node_repr: Mapping[MatchSideEnum, torch.FloatTensor],
    similarity: Similarity,
    batch_size: int,
    k: int,
    num_nodes: Mapping[MatchSideEnum, int],
) -> Mapping[MatchSideEnum, NodeIDs]:
    """
    Get kNN for all nodes.

    :param node_repr: shape: (num_nodes_on_side, dim)
        The node representations.
    :param similarity:
        The similarity measure.
    :param batch_size: >0
        The batch size.
    :param k: >0
        The number of nearest neighbors to return.
    :param num_nodes:
        The number of nodes on each side.

    :return:
        A mapping from side to an array of shape (num_nodes, k) with node IDs of the kNN.
    """
    storage_device = torch.device("cpu")
    compute_device = next(iter(node_repr.values())).device

    n, m = [num_nodes[side] for side in SIDES]
    left, right = [node_repr[side] for side in SIDES]

    # allocate buffers
    hard_right = torch.empty(n, k, dtype=torch.long, device=storage_device)
    top_v = torch.full(size=(k, m), fill_value=float('-inf'), device=compute_device)
    top_i = -torch.ones(k, m, dtype=torch.long, device=compute_device)

    # batched processing
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        # compute batch similarity, shape: (b, m)
        batch_sim = similarity.all_to_all(
            left=left[start:stop],
            right=right,
        )
        hard_right[start:stop] = batch_sim.topk(k=k, dim=1, largest=True).indices.to(storage_device)
        # combine with running top-k, shape: (b + k, m)
        top_v, b_top_i = torch.cat([batch_sim, top_v], dim=0).topk(k=k, dim=0, largest=True)
        real_batch_size = stop - start
        mask = b_top_i < real_batch_size
        top_i[mask] = b_top_i[mask]
    assert (top_i >= 0).all()
    return dict(zip(SIDES, (top_i.t().to(storage_device), hard_right)))


class HardNegativeSampler(NodeSampler, AlignmentTrainerCallback):
    """Select hard negatives."""

    hard_negatives: Mapping[MatchSideEnum, NodeIDs]

    def __init__(self, num_negatives: int, update_frequency: int = 10):
        """Initialize the sampler."""
        self.hard_negatives = None
        self.num_negatives = num_negatives
        self.batch_size = None
        self.update_frequency = update_frequency

    @torch.no_grad()
    def on_epoch_start(
        self,
        epoch: int,
        trainer: 'AlignmentModelTrainer',
    ) -> None:  # noqa: D102
        if epoch % self.update_frequency != 0:
            return

        logger.debug('Updating hard negatives.')
        node_repr = trainer.model.get_node_representations()
        num_nodes = {
            side: x.shape[0]
            for side, x in node_repr.items()
        }
        self.hard_negatives, self.batch_size = maximize_memory_utilization(
            _all_knn,
            parameter_name="batch_size",
            parameter_max_value=self.batch_size or max(v.shape[0] for v in node_repr.values()),
            node_repr=node_repr,
            similarity=trainer.similarity,
            k=self.num_negatives,
            num_nodes=num_nodes,
        )

    def sample(
        self,
        positive_batch: IDAlignment,
    ) -> NodeIDs:  # noqa: D102
        if self.hard_negatives is None:
            raise AssertionError('hard negatives have never been updated.')

        # look-up hard negatives
        return torch.stack(
            tensors=[
                self.hard_negatives[side][pos]
                for side, pos in zip(SIDES, positive_batch.flip(0))
            ],
            dim=0,
        )


#: A 3-tuple:
#   * indices (global)
#   * positives (local)
#   * negatives (local)
AlignmentBatch = Tuple[Optional[Mapping[MatchSideEnum, NodeIDs]], IDAlignment, Optional[NodeIDs]]


class AlignmentBatchCollator:
    """A custom collator for adding negative nodes to a batch of positives."""

    def __init__(
        self,
        node_sampler: Optional[NodeSampler] = None,
    ):
        """
        Initialize the collator.

        :param node_sampler:
            The node sampler.
        """
        self.sampler = node_sampler

    def collate(
        self,
        positives: List[Tuple[IDAlignment]],
    ) -> AlignmentBatch:
        """
        Collate a batch.

        :param positives:
            A tuple of positive pairs.

        :return:
            A tuple of batch node indices per side and the number of positives in the batch.
        """
        global_positives: IDAlignment = torch.stack([p[0] for p in positives], dim=-1)

        # no sampling
        if self.sampler is None:
            return None, global_positives, None

        global_negatives = self.sampler.sample(positive_batch=global_positives)

        # Translate to batch local indices
        indices = dict()
        local_positives = []
        local_negatives = []
        for side, pos_on_side, neg_on_side in zip(SIDES, global_positives, global_negatives):
            # There are positive indices P and negative indices N
            # There may be duplicates
            #   * in P, due to 1-n alignments
            #   * in N, due to random sampling with replacement
            #   * between P and N due to not filtering in N
            # We do not want to re-compute representations; thus we only keep the unique indices.
            indices_on_side = torch.cat([pos_on_side.unsqueeze(dim=-1), neg_on_side], dim=-1)
            indices[side], inverse = indices_on_side.unique(sorted=False, return_inverse=True)
            local_positives.append(inverse[:, 0])
            local_negatives.append(inverse[:, 1:])

        return (
            indices,
            torch.stack(local_positives, dim=0),
            torch.stack(local_negatives, dim=0),
        )


def prepare_alignment_batch_data_loader(
    dataset: KnowledgeGraphAlignmentDataset,
    positive_batch_size: Optional[int] = None,
    negative_sampler: Optional[NodeSampler] = None,
    num_workers: int = 0,
) -> data.DataLoader:
    """
    Prepare a PyTorch data loader for alignment model training.

    :param dataset:
        The knowledge graph alignment dataset.
    :param positive_batch_size:
        The batch size for alignment pairs.
    :param negative_sampler:
        The sampler for additional nodes from the graphs.
    :param num_workers:
        The number of worker processes.

        .. seealso ::
            torch.utils.data.DataLoader

    :return:
        The data loader.
    """
    positives = data.TensorDataset(dataset.alignment.train.t())
    if positive_batch_size is None:
        positive_batch_size = dataset.alignment.num_train
    collator = AlignmentBatchCollator(node_sampler=negative_sampler)
    return data.DataLoader(
        dataset=positives,
        batch_size=positive_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator.collate,
        pin_memory=True,
    )


class AlignmentModelTrainer(BaseTrainer[AlignmentBatch]):
    """A wrapper around a model encapsulating training and evaluation."""

    #: The model instance
    model: KGMatchingModel

    #: The similarity instance
    similarity: Similarity

    #: The loss instance
    loss: MatchingLoss

    def __init__(
        self,
        model: KGMatchingModel,
        similarity: Similarity,
        dataset: KnowledgeGraphAlignmentDataset,
        loss: MatchingLoss,
        batch_size: Optional[int] = None,
        eval_frequency: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        optimizer_cls: Type[Optimizer] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        clip_grad_norm: Optional[float] = None,
        early_stopping_key: Optional[Union[str, Sequence[str]]] = None,
        larger_is_better: bool = False,
        patience: int = 3,
        minimum_relative_difference: float = 0.,
        accumulate_gradients: int = 1,
        device: Optional[torch.device] = None,
        negative_sampler: Optional[NodeSampler] = None,
        num_workers: int = 0,
    ):
        """
        Initialize a new training loop.

        :param model:
            The model.
        :param similarity:
            The similarity.
        :param dataset:
            The dataset.
        :param loss:
            The loss instance.
        :param batch_size:
            The batch size, or None for full-batch training.
        :param eval_frequency:
            The evaluation frequency.
        :param eval_batch_size:
            The maximum batch size used for evaluation. None leads to automatic optimization.
        :param optimizer_cls:
            The optimizer class.
        :param optimizer_kwargs:
            Keyword-based arguments for the optimizer.
        :param clip_grad_norm:
            Whether to apply gradient clipping (norm-based).
        :param early_stopping_key:
            The evaluation key used for early stopping, a sequence of keys to address a value in the nested dictionary
             of evaluation results.
        :param larger_is_better:
            Whether a larger value corresponds to a better result.
        :param patience:
            The patience, i.e. number of steps without improvement to wait until the training is stopped.
        :param minimum_relative_difference:
            The minimum relative difference in the metric value to consider an improvement.
        :param accumulate_gradients:
            Accumulate gradients over batches. This can be used to simulate a larger batch size, while keeping the
            memory footprint small.
        :param device:
            The device on which to train.
        :param num_workers:
            The number of workers to use for preparing batches.
        """
        super().__init__(
            model=model,
            train_batch_size=batch_size,
            eval_frequency=eval_frequency,
            eval_batch_size=eval_batch_size,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            clip_grad_norm=clip_grad_norm,
            early_stopping_key=early_stopping_key,
            larger_is_better=larger_is_better,
            patience=patience,
            minimum_relative_difference=minimum_relative_difference,
            accumulate_gradients=accumulate_gradients,
            device=device,
        )
        self.similarity = similarity
        self.loss = loss
        self.dataset = dataset
        self.alignment = dataset.alignment
        self.num_workers = num_workers
        self.negative_sampler = negative_sampler
        if isinstance(negative_sampler, TrainerCallback):
            self.register_callbacks(negative_sampler)

    def _iter_batches(self) -> Iterable[AlignmentBatch]:  # noqa: D102
        return prepare_alignment_batch_data_loader(
            dataset=self.dataset,
            positive_batch_size=self.train_batch_size,
            negative_sampler=self.negative_sampler,
            num_workers=self.num_workers,
        )

    def _train_one_batch(self, batch: AlignmentBatch) -> Tuple[torch.Tensor, int]:
        # Unpack
        batch_node_indices, batch_alignment, negatives = batch

        # Calculate node representations
        node_repr = self.model(indices=batch_node_indices)

        # return batch loss
        return self.loss(
            alignment=batch_alignment,
            representations=node_repr,
            negatives=negatives,
        ), batch_alignment.shape[1]

    def _eval(self) -> Tuple[Mapping[str, Any], Optional[int]]:
        """Evaluate the model."""
        return evaluate_matching_model(
            model=self.model,
            alignments=self.alignment.to_dict(),
            similarity=self.similarity,
            eval_batch_size=self.eval_batch_size,
        )
