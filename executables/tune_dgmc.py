import argparse
import logging

import mlflow
import torch
from ray import tune

from kgm.trainables.search_space import SUBSETS, _resolve_embedding_dim, _resolve_node_embedding_init_config, _resolve_subsets, _sample_init
from kgm.modules.embeddings.norm import EmbeddingNormalizationMethod
from kgm.trainables.dgmc_matching import DgmcMatching
from kgm.utils.torch_utils import get_device
from kgm.utils.tune_utils import MedianStoppingRuleNoPreemption, tune_linear_quantized_range, tune_enum

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name="kgm").setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    device = get_device(device=None)
    print(f'Running on {device}')

    search_space = dict(
        mlflow=dict(
            tracking_uri=args.tracking_uri,
            name='dgmc',
        ),
        seed=42,
        device=device,
        training=dict(
            max_num_epochs=200,  # DGMC automatically switches to refinement after epoch 100
            eval_frequency=10,  # note: DGMC automatically switches to eval_frequency of 1 during refinement
        ),
        data=dict(
            dataset=tune.choice(list(SUBSETS.keys())),
            subset=tune.sample_from(_resolve_subsets),
            train_validation_split=.8,
        ),
        model=dict(
            cls='DGMC',
            dim=tune_linear_quantized_range(high=1024, q=32),
            rnd_dim=tune_linear_quantized_range(high=1024, q=32),
            ps1_n_layers=tune.choice(list(range(1, 5))),
            ps2_n_layers=tune.choice(list(range(1, 5))),
            num_steps=10,
            psi1_batch_norm=tune.choice([True, False]),
            psi2_batch_norm=tune.choice([True, False]),
            psi1_cat=tune.choice([True, False]),
            psi2_cat=tune.choice([True, False]),
            psi1_dropout=tune.choice([0.05 * i for i in range(21)]),
            psi2_dropout=0.0,
            node_embedding_init_method=tune.sample_from(_sample_init),
            node_embedding_init_config=tune.sample_from(_resolve_node_embedding_init_config),
            trainable_node_embeddings=False,
            embedding_dim=tune.sample_from(_resolve_embedding_dim),
            node_embedding_norm=tune_enum(EmbeddingNormalizationMethod),
        ),
        optimizer=dict(
            cls='adam',
            lr=tune.choice([1e-4, 1e-3, 1e-2]),
        ),
    )

    scheduler = MedianStoppingRuleNoPreemption(
        time_attr='training_iteration',
        metric='eval/validation/hits_at_1',
        mode='max',
        grace_period=100,
        min_samples_required=3,
    )

    analysis = tune.run(
        DgmcMatching,
        name='dgmc',
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
