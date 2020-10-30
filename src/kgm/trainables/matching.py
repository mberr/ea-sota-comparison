# coding=utf-8
"""Tune trainables for matching models."""
import copy
import logging
import pathlib
from typing import Any, Iterator, Mapping, MutableMapping

import mlflow
import torch
from ray.tune import tune
from ray.tune.result import DONE
from torch import nn

from kgm.data import KnowledgeGraphAlignmentDataset, get_dataset_by_name
from kgm.models import GraphBasedKGMatchingModel, KGMatchingModel, get_matching_model_by_name
from kgm.modules import MatchingLoss, get_matching_loss, get_pairwise_loss, get_similarity, get_activation
from kgm.training.matching import AlignmentModelTrainer
from kgm.utils.mlflow_utils import connect_mlflow, log_metrics_to_mlflow, log_params_to_mlflow
from kgm.utils.torch_utils import get_device

logger = logging.getLogger(name=__name__)


# pylint: disable=attribute-defined-outside-init
class MatchingTrainable(tune.Trainable):
    """Base class for entity alignment experiments with ray.tune."""

    #: The device
    device: torch.device

    #: The model
    model: nn.Module

    #: The training entity alignment
    dataset: KnowledgeGraphAlignmentDataset

    def _load_data(self, data_config: MutableMapping[str, Any]) -> KnowledgeGraphAlignmentDataset:
        """Load dataset."""
        dataset_name = data_config.pop('dataset')
        subset_name = data_config.pop('subset', None)
        inverse_triples = data_config.pop('inverse_triples', False)
        self_loops = data_config.pop('self_loops', False)
        train_test_split = data_config.pop('train_test_split', None)
        train_validation_split = data_config.pop('train_validation_split', 0.8)
        dataset = get_dataset_by_name(
            dataset_name=dataset_name,
            subset_name=subset_name,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
            train_test_split=train_test_split,
            train_validation_split=train_validation_split
        )
        return dataset

    def _load_loss(self, loss_config: MutableMapping[str, Any]) -> MatchingLoss:
        """Load loss."""
        # Instantiate pairwise loss
        pairwise_loss_config = loss_config.pop('pairwise_loss')
        pairwise_loss_cls_name = pairwise_loss_config.pop('cls')
        if 'activation' in pairwise_loss_config:
            pairwise_loss_config['activation'] = get_activation(activation_name=pairwise_loss_config['activation'])
        pairwise_loss = get_pairwise_loss(pairwise_loss_cls_name, **pairwise_loss_config)

        matching_loss_cls_name = loss_config.pop('cls')
        matching_loss = get_matching_loss(name=matching_loss_cls_name, similarity=self.similarity, base_loss=pairwise_loss, **loss_config)
        return matching_loss

    def _load_model(self, model_config: MutableMapping[str, Any]) -> KGMatchingModel:
        """Load model."""
        cls_name = model_config.pop('cls')
        model_cls = get_matching_model_by_name(name=cls_name)

        if issubclass(model_cls, GraphBasedKGMatchingModel):
            model = model_cls(dataset=self.dataset, **model_config)
        else:
            model = model_cls(num_nodes=self.dataset.num_nodes, **model_config)
        return model

    def _load_training(self, training_config: Mapping[str, Any]) -> None:
        """Load training config."""
        self.max_num_epochs = training_config.get('max_num_epochs', 1)
        self.eval_frequency = training_config.get('eval_frequency', None)
        self.clip_grad_norm = training_config.get('clip_grad_norm', None)
        self.train_batch_size = training_config.get('batch_size', None)
        self.num_neg_per_batch = training_config.get('num_neg_per_batch', None)
        self.accumulate_gradients = training_config.get('accumulate_gradients', 1)
        self.log_only_eval = training_config.pop('log_only_eval', True)

        # early stopping
        self.early_stopping_key = training_config.get('early_stopping_key', None)
        self.larger_is_better = training_config.get('larger_is_better', False)
        self.patience = training_config.get('patience', 3)
        self.minimum_relative_difference = training_config.get('minimum_relative_difference', 0.)

    def _load_mlflow(self, mlflow_config: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        """Connect to MLFlow."""
        self.use_mlflow = True
        if mlflow_config.get('ignore', False):
            self.use_mlflow = False
            return

        experiment_name = mlflow_config.get('name', self.__class__.__name__.replace('Trainable', ''))
        kwargs = {}
        if 'tracking_uri' in mlflow_config.keys():
            kwargs['tracking_uri'] = mlflow_config['tracking_uri']

        connect_mlflow(experiment_name=experiment_name, **kwargs)
        log_params_to_mlflow(config=config)

    def _load_train_iter(self) -> Iterator[Mapping[str, Any]]:
        """Initialize the train iterable."""
        self.trainer = AlignmentModelTrainer(
            model=self.model,
            similarity=self.similarity,
            dataset=self.dataset,
            loss=self.loss,
            batch_size=self.train_batch_size,
            eval_frequency=self.eval_frequency,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            early_stopping_key=self.early_stopping_key,
            larger_is_better=self.larger_is_better,
            patience=self.patience,
            minimum_relative_difference=self.minimum_relative_difference,
            accumulate_gradients=self.accumulate_gradients,
            device=self.device,
        )
        return iter(self.trainer.train_iter(num_epochs=self.max_num_epochs))

    def _setup(self, config):  # noqa: D102
        local_config = config.copy()

        # Logging
        mlflow_config = config.get('mlflow')
        if mlflow_config is None:
            logger.warning('No MLFlow configuration found. Thus, no logging to MLFlow.')
            mlflow_config = dict(ignore=True)
        self._load_mlflow(mlflow_config=mlflow_config, config=config)

        # random seed
        self.seed = local_config.pop('seed')

        # Device
        device_name = local_config.pop('device', None)
        self.device = get_device(device=device_name)
        logger.info('Using device: %s', self.device)

        # Dataset
        data_config = local_config.pop('data')
        self.dataset = self._load_data(data_config)

        # Model
        model_config = local_config.pop('model')
        self.model = self._load_model(model_config).to(self.device)

        # Similarity
        similarity_config = local_config.pop('similarity')
        self.similarity = get_similarity(similarity=similarity_config['cls'], transformation=similarity_config.get('transformation', None))

        # Loss
        loss_config = local_config.pop('loss')
        self.loss = self._load_loss(loss_config=loss_config)

        # Automatically choose eval batch size
        self.eval_batch_size = None

        # Optimizer
        optimizer_config = local_config.pop('optimizer')
        self.optimizer_cls = optimizer_config.pop('cls', None)
        self.optimizer_kwargs = copy.deepcopy(optimizer_config)

        # Training
        train_config = local_config.pop('training')
        self._load_training(training_config=train_config)
        self.train_iter = self._load_train_iter()

    def _train(self) -> Mapping[str, Any]:  # noqa: D102
        result = dict()
        try:
            result = next(self.train_iter)
            if self.log_only_eval:
                while 'eval' not in result:
                    result = next(self.train_iter)
        except StopIteration:
            result[DONE] = True

        # Log to MLFlow
        if self.use_mlflow:
            log_metrics_to_mlflow(metrics=result, step=result['epoch'])

        return result

    def _save(self, tmp_checkpoint_dir):
        """
        Save model at ray.tune checkpoint.

        :param tmp_checkpoint_dir: The directory.

        :return: A dictionary of paths of files containing
            * model's state_dict
            * optimizer's state_dict
        """
        # Construct filename
        mlflow_active_run = mlflow.active_run()
        if mlflow_active_run is not None:
            run_uuid = mlflow_active_run.info.run_id
        else:
            run_uuid = 'UNKNOWN_MLFLOW_UUID'

        # Ensure directory exists
        tmp_checkpoint_dir = pathlib.Path(tmp_checkpoint_dir) / run_uuid
        tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state-dict
        model_path = tmp_checkpoint_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state-dict
        optimizer_path = tmp_checkpoint_dir / 'optimizer.pt'
        if hasattr(self, 'trainer'):
            torch.save(self.trainer.optimizer.state_dict(), optimizer_path)

        return dict(
            model=model_path,
            optimizer=optimizer_path,
        )

    def _restore(self, checkpoint):
        """
        Restore model from ray.tune checkpoint.

        Use tune.run(..., restore=<path-to-checkpoint>) to restore from checkpoint.
        See https://ray.readthedocs.io/en/latest/tune-usage.html#fault-tolerance
        """
        # Restore model
        model_checkpoint = torch.load(checkpoint.pop('model'))
        self.model.load_state_dict(state_dict=model_checkpoint)

        # Restore optimizer
        if hasattr(self, 'trainer'):
            optimizer_checkpoint = torch.load(checkpoint.pop('optimizer'))
            self.trainer.optimizer.load_state_dict(state_dict=optimizer_checkpoint)
