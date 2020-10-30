import tempfile
import unittest
from typing import Any, MutableMapping

from kgm.data import KnowledgeGraphAlignmentDataset, get_synthetic_math_graph, sub_graph_alignment
from kgm.models import KGMatchingModel, PureEmbeddingModel
from kgm.trainables.matching import MatchingTrainable


class DummyMatchingTrainable(MatchingTrainable):
    """A dummy trainable without real dataset or model."""
    #: The number of entities
    num_entities: int = 128

    #: The dimension of the embedding
    dim: int = 8

    def _load_data(self, data_config: MutableMapping[str, Any]) -> KnowledgeGraphAlignmentDataset:
        return sub_graph_alignment(get_synthetic_math_graph(num_entities=33))

    def _load_model(self, model_config: MutableMapping[str, Any]) -> KGMatchingModel:
        return PureEmbeddingModel(dataset=self.dataset, embedding_dim=self.dim)


class MatchingTrainableTests(unittest.TestCase):
    # pylint: disable=protected-access
    def test_trainable(self):
        # Create object without using the constructor (which would require a running MLFlow and Redis instance)
        trainable = DummyMatchingTrainable.__new__(DummyMatchingTrainable)

        # minimal working example
        config = dict(
            seed=42,
            device=None,
            training=dict(),
            data=dict(),
            model=dict(),
            similarity=dict(
                cls='dot',
            ),
            loss=dict(
                cls='sampled',
                pairwise_loss=dict(
                    cls='margin',
                ),
            ),
            optimizer=dict(
                cls='adam',
            ),
            # Disable MLFlow
            mlflow=dict(
                ignore=True,
            )
        )

        # manually call _setup
        trainable._setup(config=config)

        # manually call _train
        result = trainable._train()
        assert isinstance(result, dict)
        assert 'loss' in result['train'].keys()

        # Manually add ray internal attributes
        trainable._experiment_id = 0
        trainable._iteration = 1
        trainable._timesteps_total = 100
        trainable._time_total = 1.0
        trainable._episodes_total = 1

        # test save & load
        with tempfile.TemporaryDirectory() as tmp_dir:
            # save
            trainable.save(tmp_dir)
            # load: missing Tune metadata
            # trainable.restore(tmp_dir)
