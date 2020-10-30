import copy
import logging

import torch

from os import path as osp

from typing import MutableMapping, Any, Mapping, Iterable

from ray.tune.result import DONE
from torch_geometric.data import Data
from torch_geometric.datasets import DBP15K

from dgmc import DGMC
from dgmc.models import RelCNN
from kgm.data import MatchSideEnum
from kgm.modules.embeddings import get_embedding_pair
from kgm.modules.embeddings.base import NodeEmbeddingInitMethod
from kgm.modules.embeddings.norm import EmbeddingNormalizationMethod
from kgm.trainables.matching import MatchingTrainable
from kgm.utils.common import kwargs_or_empty
from kgm.utils.mlflow_utils import log_metrics_to_mlflow
from kgm.utils.torch_utils import get_device, construct_optimizer_from_config

logger = logging.getLogger(name=__name__)


class SumEmbedding(object):
    def __call__(self, data):
        data.x1, data.x2 = data.x1.sum(dim=1), data.x2.sum(dim=1)
        return data


class DgmcMatching(MatchingTrainable):
    epoch = 0

    def __load_model(self, model_config: MutableMapping[str, Any]) -> DGMC:
        psi_1 = RelCNN(
            in_channels=self.data.x1.size(-1),
            out_channels=model_config.get('dim', 256),
            num_layers=model_config.get('ps1_n_layers', 3),
            batch_norm=model_config.get('psi1_batch_norm', False),
            cat=model_config.get('psi1_cat', True),
            lin=model_config.get('psi1_cat', True),
            dropout=model_config.get('psi1_dropout', 0.5),
        )
        psi_2 = RelCNN(
            in_channels=model_config.get('rnd_dim', 32),
            out_channels=model_config.get('rnd_dim', 32),
            num_layers=model_config.get('ps2_n_layers', 3),
            batch_norm=model_config.get('psi2_batch_norm', False),
            cat=model_config.get('psi2_cat', True),
            lin=model_config.get('psi2_cat', True),
            dropout=model_config.get('psi2_dropout', 0.0),
            )

        model = DGMC(psi_1=psi_1, psi_2=psi_2, num_steps=0, k=10).to(self.device)
        return model

    def _wrap_data(self, model_config: MutableMapping[str, Any]) -> Data:
        embeddings = get_embedding_pair(
            dataset=self.dataset,
            init=model_config.get('node_embedding_init_method', NodeEmbeddingInitMethod.random),
            init_config=model_config.get('node_embedding_init_config', None),
            trainable=model_config.get('trainable_node_embeddings', True),
            embedding_dim=model_config.get('embedding_dim', 300),
            norm=model_config.get('node_embedding_norm', EmbeddingNormalizationMethod.none),
        )

        data = Data(
            x1=embeddings[MatchSideEnum.left].weight.to(self.device),
            edge_index1=torch.unique(self.dataset.left_graph.triples[:, [0, 2]].t(), dim=1).to(self.device),
            x2=embeddings[MatchSideEnum.right].weight.to(self.device),
            edge_index2=torch.unique(self.dataset.right_graph.triples[:, [0, 2]].t(), dim=1).to(self.device),
            train_y=self.dataset.alignment.train.to(self.device),
            test_y=self.dataset.alignment.test.to(self.device),
            val_y=self.dataset.alignment.validation.to(self.device),
        )

        return data

    def __train(self):
        self.model.train()
        self.optimizer.zero_grad()

        _, S_L = self.model(self.data.x1, self.data.edge_index1, None, None, self.data.x2,
                            self.data.edge_index2, None, None, self.data.train_y)

        loss = self.model.loss(S_L, self.data.train_y)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def _train_results(self):
        self.model.eval()

        _, S_L = self.model(self.data.x1, self.data.edge_index1, None, None, self.data.x2,
                            self.data.edge_index2, None, None)

        hits1 = self.model.acc(S_L, self.data.train_y)
        hits10 = self.model.hits_at_k(10, S_L, self.data.train_y)

        return {
            'hits_at_1': float(hits1),
            'hits_at_10': float(hits10),
        }

    @torch.no_grad()
    def _test(self):
        self.model.eval()

        _, S_L = self.model(self.data.x1, self.data.edge_index1, None, None, self.data.x2,
                            self.data.edge_index2, None, None)

        hits1 = self.model.acc(S_L, self.data.test_y)
        hits10 = self.model.hits_at_k(10, S_L, self.data.test_y)

        return {
            'hits_at_1': float(hits1),
            'hits_at_10': float(hits10),
        }

    @torch.no_grad()
    def _eval(self):
        self.model.eval()

        _, S_L = self.model(self.data.x1, self.data.edge_index1, None, None, self.data.x2,
                            self.data.edge_index2, None, None)

        hits1 = self.model.acc(S_L, self.data.val_y)
        hits10 = self.model.hits_at_k(10, S_L, self.data.val_y)

        return {
            'hits_at_1': float(hits1),
            'hits_at_10': float(hits10),
        }

    def eval(self):
        # don't run self._eval if there is no validation split
        if not hasattr(self, 'dataset') or self.dataset.alignment._validation is None:
            results = {
                'train': self._train_results(),
                'test': self._test(),
            }
        else:
            results = {
                'train': self._train_results(),
                'test': self._test(),
                'validation': self._eval()
            }

        return results

    def _train_iter(
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
            self.model.train()

            # training step
            self.epoch += 1
            epoch_result = dict(
                epoch=self.epoch,
                train_loss=float(self.__train()),
            )

            # evaluate
            if (final_eval and self.epoch == num_epochs) or (self.eval_frequency is not None and (self.epoch % self.eval_frequency == 0)) or self.epoch > 100:
                self.model.eval()
                with torch.no_grad():
                    epoch_result['eval'] = self.eval()

            yield epoch_result

        return epoch_result

    def _setup(self, config):
        local_config = config.copy()

        # Logging
        mlflow_config = config.get('mlflow')
        if mlflow_config is None:
            logger.warning('No MLFlow configuration found. Thus, no logging to MLFlow.')
            mlflow_config = dict(ignore=True)
        self._load_mlflow(mlflow_config=mlflow_config, config=config)

        # random seed
        self.seed = local_config.pop('seed')
        torch.manual_seed(self.seed)

        # Device
        device_name = local_config.pop('device', None)
        self.device = get_device(device=device_name)
        logger.info('Using device: %s', self.device)

        # Dataset
        data_config = local_config.pop('data')
        dataset_name = data_config['dataset']
        if dataset_name == 'dbp15kjape_torch_geometric':
            path = osp.join('..', 'data', 'DBP15K')
            self.data = DBP15K(path, data_config['subset'], transform=SumEmbedding())[0].to(self.device)
        else:
            self.dataset = self._load_data(data_config)
            self.train_val_split = data_config.get('train_validation_split') is not None

        # Model
        model_config = local_config.pop('model')
        if dataset_name == 'dbp15kjape_torch_geometric':
            # consistent condition path with data loading
            pass
        else:
            self.data = self._wrap_data(model_config=model_config)
        self.model = self.__load_model(model_config=model_config)
        self.model_config = model_config

        # Optimizer
        optimizer_config = local_config.pop('optimizer')
        optimizer_cls = optimizer_config.pop('cls', 'adam')
        optimizer_kwargs = copy.deepcopy(optimizer_config)
        optimizer_config = dict(cls=optimizer_cls)
        optimizer_config.update(kwargs_or_empty(optimizer_kwargs))
        self.optimizer = construct_optimizer_from_config(model=self.model, optimizer_config=optimizer_config)

        # Training
        train_config = local_config.pop('training')
        self._load_training(training_config=train_config)

        self.train_iter = iter(self._train_iter(num_epochs=self.max_num_epochs))

    def _train(self) -> Mapping[str, Any]:
        if self.epoch == 101:
            self.model.num_steps = self.model_config.get('num_steps', 10)
            self.model.detach = True

        result = dict()
        try:
            result = next(self.train_iter)
            if self.log_only_eval:
                while 'eval' not in result:
                    result = next(self.train_iter)
        except StopIteration:
            result[DONE] = True

        # Log to MLFlow
        if self.use_mlflow and 'epoch' in result:
            log_metrics_to_mlflow(metrics=result, step=result['epoch'])

        if 'eval' in result:
            if 'validation' in result['eval']:
                result['checkpoint_score_attr'] = result['eval']['validation']['hits_at_1']
            else:
                result['checkpoint_score_attr'] = result['eval']['test']['hits_at_1']

        return result
