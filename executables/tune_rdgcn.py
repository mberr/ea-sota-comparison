"""HPO script for RDGCN."""
import argparse
import logging
import random

import mlflow
import torch
from ray import tune
from ray.tune.schedulers import FIFOScheduler

from kgm.trainables.search_space import SUBSETS, _resolve_node_embedding_init_config, _resolve_subsets, _sample_init
from kgm.modules import SimilarityEnum
from kgm.modules.embeddings.base import EmbeddingNormalizationMode
from kgm.modules.embeddings.norm import EmbeddingNormalizationMethod
from kgm.trainables.matching import MatchingTrainable
from kgm.utils.torch_utils import get_device
from kgm.utils.tune_utils import tune_bool, tune_enum


def _sample_interaction_round_weights(spec):
    # default weights are [0.1, 0.3]
    num_interactions = spec.config.model.num_interactions
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    return [random.choice(weights) for _ in range(num_interactions)]


def _sample_embedding_norm_mode(spec):
    norm = spec.config.model.embedding_norm
    if norm == EmbeddingNormalizationMethod.none:
        return EmbeddingNormalizationMode.none
    return tune.choice([
        EmbeddingNormalizationMode.initial,
        EmbeddingNormalizationMode.every_forward,
    ])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name="kgm").setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    device = get_device(device=None)
    logging.info(f"Running on {device}")

    search_space = dict(
        mlflow=dict(
            tracking_uri=args.tracking_uri,
            name='rdgcn',
        ),
        seed=42,
        device=device,
        training=dict(
            max_num_epochs=1000,
            eval_frequency=10,
            sampler=tune.choice([None, "hard_negative"])
        ),
        data=dict(
            dataset=tune.choice(list(SUBSETS.keys())),
            subset=tune.sample_from(_resolve_subsets),
            train_validation_split=.8,
        ),
        model=dict(
            cls='RDGCN',
            embedding_dim=None,  # automatic
            interaction_round_weights=tune.sample_from(_sample_interaction_round_weights),
            embedding_norm=tune.choice([
                EmbeddingNormalizationMethod.l2,
                EmbeddingNormalizationMethod.none,
            ]),
            embedding_norm_mode=tune.sample_from(_sample_embedding_norm_mode),
            trainable_node_embeddings=tune_bool,
            num_interactions=tune.choice([0, 1, 2, 3]),
            num_gcn_layers=tune.choice([0, 1, 2, 3]),
            node_embedding_init_method=tune.sample_from(_sample_init),
            node_embedding_init_config=tune.sample_from(_resolve_node_embedding_init_config)
        ),
        similarity=dict(
            cls=tune_enum(enum=SimilarityEnum),
            transformation=tune.choice(['bound_inverse', 'negative']),
        ),
        loss=dict(
            cls='sampled',
            pairwise_loss=dict(
                cls="margin",
                margin=1.0,
            ),
            # **default_sampled_matching_loss_hpo,
        ),
        optimizer=dict(
            cls='adam',
            lr=tune.loguniform(1.0e-04, 1.0e-01),
        ),
    )

    scheduler = FIFOScheduler()

    num_gpu = torch.cuda.device_count()
    logging.info(f"Using {num_gpu} GPUs.")
    analysis = tune.run(
        MatchingTrainable,
        name='matching',
        verbose=1,
        num_samples=2000,
        max_failures=0,
        checkpoint_freq=1001,
        checkpoint_at_end=False,
        resources_per_trial=dict(
            cpu=2,
            gpu=1 if torch.cuda.is_available() else 0,
        ),
        config=search_space,
        scheduler=scheduler,
    )
