import copy

from kgm.modules import get_similarity
from kgm.trainables.matching import MatchingTrainable, logger
from kgm.utils.mlflow_utils import log_metrics_to_mlflow
from kgm.utils.torch_utils import get_device


class GCNAlignMatching(MatchingTrainable):
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

        # Device
        device_name = local_config.pop('device', None)
        self.device = get_device(device=device_name)
        logger.info('Using device: %s', self.device)

        # Dataset
        data_config = local_config.pop('data')
        self.dataset = self._load_data(data_config)

        # Model
        model_config = local_config.pop('model')
        # inject device to use - seems needed to make sure the adj. matrices reside on the same device
        # model_config['device'] = self.device
        self.model = self._load_model(model_config).to(self.device)

        # Similarity
        similarity_config = local_config.pop('similarity')
        self.similarity = get_similarity(similarity=similarity_config['cls'], transformation=similarity_config.get('transformation', None)).to(self.device)

        # Loss
        loss_config = local_config.pop('loss')
        self.loss = self._load_loss(loss_config=loss_config).to(self.device)

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

        zero_shot_results = self.trainer._eval()
        # during the first call, this is a tuple containing the evaluation results. We're transforming it to fit the
        # usual style
        zero_shot_results = {
            'eval': zero_shot_results[0]
        }
        log_metrics_to_mlflow(metrics=zero_shot_results, step=0)
