"""Common training loop parts."""
import copy
import logging
import pathlib
from typing import Any, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import torch
from torch import nn
from torch.optim import Optimizer

from kgm.utils.common import NonFiniteLossError, get_value_from_nested_mapping, kwargs_or_empty, last
from kgm.utils.torch_utils import construct_optimizer_from_config, get_device

logger = logging.getLogger(name=__name__)


class EarlyStopping:
    """Early stopping with patience."""

    #: The evaluation key
    key: Sequence[str]

    def __init__(
        self,
        key: Union[str, Sequence[str]],
        larger_is_better: bool,
        patience: int = 3,
        minimum_relative_difference: float = 0.,
    ):
        """
        Initialize early stopping component.

        :param key:
            The evaluation key, a sequence of keys to address a value in the nested dictionary of evaluation results.
        :param larger_is_better:
            Whether a larger value corresponds to a better result.
        :param patience:
            The patience, i.e. number of steps without improvement to wait until the training is stopped.
        :param minimum_relative_difference:
            The minimum relative difference in the metric value to consider an improvement.
        """
        if isinstance(key, str):
            key = key.split('/')
        self.key = key
        self.max_patience = patience
        self.minimum_relative_difference = minimum_relative_difference
        self.larger_is_better = larger_is_better
        self.best_value = None
        self.best_model_state_dict = None
        self.patience = self.max_patience

    def is_better(
        self,
        last_value: float,
    ) -> bool:
        """Whether the last value is better than the current best value."""
        last_value = float(last_value)
        if self.best_value is None:
            return True
        if self.larger_is_better:
            return last_value > self.best_value * (1. + self.minimum_relative_difference)
        else:
            return last_value < self.best_value * (1. - self.minimum_relative_difference)

    def should_stop(
        self,
        evaluation: Mapping[str, Any],
        model: nn.Module,
    ) -> bool:
        """
        Evaluate stopping criterion.

        :param evaluation:
            The evaluation result as dictionary.
        :param model:
            The model. Used to store the state-dict of the best model.

        :return:
            Whether the training should stop.
        """
        last_value = get_value_from_nested_mapping(dictionary=evaluation, keys=self.key, default=None)
        if last_value is None:
            return False
        if self.is_better(last_value=last_value):
            self.patience = self.max_patience
            self.best_value = last_value
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.patience -= 1

        return self.patience < 0


class TrainerCallback:
    """A callback for training."""

    def on_epoch_start(self, epoch: int, trainer: 'BaseTrainer') -> None:
        """
        Execute this code on each epoch's start.

        :param epoch:
            The epoch.
        :param trainer:
            The trainer.
        """
        raise NotImplementedError


BatchType = TypeVar('BatchType')


class BaseTrainer(Generic[BatchType]):
    """A base class for training loops."""

    #: The model
    model: nn.Module

    #: Early stopping component
    early_stopper: Optional[EarlyStopping]

    #: The optimizer instance
    optimizer: Optimizer

    #: Callbacks
    callbacks: List[TrainerCallback]

    def __init__(
        self,
        model: nn.Module,
        train_batch_size: Optional[int] = None,
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
    ):
        """
        Initialize a new training loop.

        :param model:
            The model to train.
        :param train_batch_size:
            The batch size to use for training.
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
        :param accumulate_gradients:
            Accumulate gradients over batches. This can be used to simulate a larger batch size, while keeping the
            memory footprint small.
        :param device:
            The device on which to train.
        """
        device = get_device(device=device)
        # Bind parameters
        self.train_batch_size = train_batch_size
        self.model = model.to(device=device)
        self.epoch = 0
        self.eval_frequency = eval_frequency
        self.eval_batch_size = eval_batch_size
        self.accumulate_gradients = accumulate_gradients
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.callbacks = []

        # early stopping
        if early_stopping_key is not None:
            self.early_stopper = EarlyStopping(
                key=early_stopping_key,
                larger_is_better=larger_is_better,
                patience=patience,
                minimum_relative_difference=minimum_relative_difference,
            )
        else:
            self.early_stopper = None
        self.accumulate_gradients = accumulate_gradients
        self.device = device

        # create optimizer
        if optimizer_cls is None:
            optimizer_cls = 'adam'
        optimizer_config = dict(cls=optimizer_cls)
        optimizer_config.update(kwargs_or_empty(optimizer_kwargs))
        self.optimizer_config = optimizer_config
        self.reset_optimizer()

    def register_callbacks(self, *callback: TrainerCallback) -> None:
        """Register callbacks."""
        self.callbacks.extend(callback)

    def reset_optimizer(self) -> None:
        """Reset the optimizer."""
        self.optimizer = construct_optimizer_from_config(model=self.model, optimizer_config=self.optimizer_config)

    def _train_one_epoch(self) -> Mapping[str, Any]:
        """
        Train the model for one epoch on the given device.

        :return:
            A dictionary of training results. Contains at least `loss` with the epoch loss value.
        """
        epoch_loss, counter = 0., 0

        # Iterate over batches
        i = -1
        for i, batch in enumerate(self._iter_batches()):
            # Compute batch loss
            batch_loss, real_batch_size = self._train_one_batch(batch=batch)

            # Break on non-finite loss values
            if not torch.isfinite(batch_loss).item():
                raise NonFiniteLossError

            # Update epoch loss
            epoch_loss += batch_loss.item() * real_batch_size
            counter += real_batch_size

            # compute gradients
            batch_loss.backward()

            # Apply gradient updates
            if i % self.accumulate_gradients == 0:
                self._parameter_update()

        # For the last batch, we definitely do an update
        if self.accumulate_gradients > 1 and (i % self.accumulate_gradients) != 0:
            self._parameter_update()

        return dict(
            loss=epoch_loss / counter
        )

    def _parameter_update(self):
        """Update the parameters using the optimizer."""
        # Gradient clipping
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                parameters=(p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.clip_grad_norm,
            )

        # update parameters
        self.optimizer.step()

        # clear gradients afterwards
        self.optimizer.zero_grad()

        # call post parameter update hook (e.g. to ensure model constraints)
        self._post_parameter_update_hook()

    def _iter_batches(self) -> Iterable[BatchType]:
        """Iterate over batches."""
        raise NotImplementedError

    def _train_one_batch(self, batch: BatchType) -> Tuple[torch.Tensor, int]:
        """
        Train on a single batch.

        :param batch: shape: (batch_size,)
            The sample IDs.

        :return:
            A tuple (batch_loss, real_batch_size) of the batch loss (a scalar tensor), and the actual batch size.
        """
        raise NotImplementedError

    def _post_parameter_update_hook(self) -> None:
        """Allow applying a hook after each parameter update, e.g. to ensure model constraints."""

    def _eval(self) -> Tuple[Mapping[str, Any], Optional[int]]:
        """
        Evaluate the model.

        :return:
            A tuple (result, eval_batch_size) of a dictionary with the evaluation result, and the maximum possible
            evaluation batch_size.
        """
        return dict(), None

    def train_iter(
        self,
        num_epochs: int = 1,
        final_eval: bool = True,
    ) -> Iterable[Mapping[str, Any]]:
        """
        Train the model, and return intermediate results.

        :param num_epochs:
            The number of epochs.
        :param final_eval:
            Whether to perform an evaluation after the last training epoch.

        :return:
            One result dictionary per epoch.
        """
        epoch_result = dict()
        for _ in range(self.epoch, self.epoch + num_epochs):
            # callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch=self.epoch, trainer=self)

            self.model.train()

            # training step
            self.epoch += 1
            epoch_result = dict(
                epoch=self.epoch,
                train=self._train_one_epoch(),
            )

            # evaluate
            if (final_eval and self.epoch == num_epochs) or (self.eval_frequency is not None and (self.epoch % self.eval_frequency == 0)):
                self.model.eval()
                with torch.no_grad():
                    epoch_result['eval'], self.eval_batch_size = self._eval()

            # Early stopping
            if self.early_stopper is not None:
                if self.early_stopper.should_stop(evaluation=epoch_result, model=self.model):
                    # restore best model's parameters
                    self.model.load_state_dict(self.early_stopper.best_model_state_dict)
                    break

            yield epoch_result

        return epoch_result

    def train(
        self,
        num_epochs: int = 1,
        final_eval: bool = True,
    ) -> Mapping[str, Any]:
        """
        Train the model, and return intermediate results.

        :param num_epochs:
            The number of epochs.
        :param final_eval:
            Whether to perform an evaluation after the last training epoch.

        :return:
            A dictionary containing the result.
        """
        return last(self.train_iter(num_epochs=num_epochs, final_eval=final_eval))

    def save_to_dir(self, checkpoint_dir: Union[pathlib.Path, str]) -> Mapping[str, pathlib.Path]:
        """
        Save state to checkpoint directory.

        :param checkpoint_dir:
            The checkpoint directory.

        :return:
            A mapping from {'model', 'optimizer'} to the files where the corresponding states are saved.
        """
        # Normalize path
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)

        # Ensure directory exists
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        result = dict()
        for name, component in dict(
            model=self.model,
            optimizer=self.optimizer,
        ).items():
            # Save state-dict
            file_path = self._get_checkpoint_path(checkpoint_dir, component=name)
            torch.save(component.state_dict(), file_path)
            result[name] = file_path

        return result

    @staticmethod
    def _get_checkpoint_path(checkpoint_dir, component: str) -> pathlib.Path:
        """
        Return the canonical file path for a component.

        :param checkpoint_dir:
            The checkpoint directory.
        :param component:
            The component, from {'model', 'optimizer'}.

        :return:
            The file path.
        """
        return checkpoint_dir / f'{component}.pt'

    def load_from_dir(
        self,
        checkpoint_dir: Union[pathlib.Path, str],
    ) -> None:
        """
        Load the training state from a checkpoint directory.

        :param checkpoint_dir:
            The checkpoint directory, an existing directory containing the serialized model and optimizer state.
        """
        # Normalize path
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)

        for name, component in dict(
            model=self.model,
            optimizer=self.optimizer,
        ).items():
            # Restore component
            checkpoint_path = self._get_checkpoint_path(checkpoint_dir=checkpoint_dir, component=name)
            component.load_state_dict(state_dict=torch.load(checkpoint_path, map_location=self.device))
