"""HPO Script for GCN-Align."""
import argparse
import logging

import mlflow
import torch
from ray import tune

from kgm.data.reduction import DropRelationInformationKnowledgeGraphToGraphReduction
from kgm.modules.embeddings.base import EmbeddingNormalizationMode
from kgm.modules.embeddings.norm import EmbeddingNormalizationMethod
from kgm.trainables.gcn_align_matching import GCNAlignMatching
from kgm.trainables.search_space import SUBSETS, _resolve_embedding_dim, _resolve_node_embedding_init_config, _resolve_subsets, _sample_init, _tune_linear_quantized_range
from kgm.utils.torch_utils import get_device
from kgm.utils.tune_utils import MedianStoppingRuleNoPreemption, default_sampled_matching_loss_hpo, tune_enum

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    device = get_device(device=None)
    print(f'Running on {device}')

    search_space = dict(
        mlflow=dict(
            tracking_uri=args.tracking_uri,
            name='gcn_align',
        ),
        seed=42,
        device=None,
        training=dict(
            max_num_epochs=2000,
            eval_frequency=10,
        ),
        data=dict(
            dataset=tune.choice(list(SUBSETS.keys())),
            subset=tune.sample_from(_resolve_subsets),
            train_validation_split=.8,
        ),
        model=dict(
            cls='dgmcgcnalign',
            embedding_dim=None,  # automatically selected based on initializer
            output_dim=tune.sample_from(_tune_linear_quantized_range()),
            num_layers=tune.choice([1, 2, 3]),
            batch_norm=tune.choice([True, False]),
            cat=tune.choice([True, False]),
            final_linear_projection=tune.choice([True, False]),
            dropout=tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            trainable_node_embeddings=tune.choice([True, False]),
            node_embedding_norm=tune_enum(EmbeddingNormalizationMethod),
            node_embedding_mode=EmbeddingNormalizationMode.every_forward,
            reduction_cls=DropRelationInformationKnowledgeGraphToGraphReduction,
            reduction_kwargs=dict(
                unique=True,
            ),
            vertical_sharing=False,
            horizontal_sharing=tune.choice([True, False]),
            node_embedding_init_method=tune.sample_from(_sample_init),
            node_embedding_init_config=tune.sample_from(_resolve_node_embedding_init_config)
        ),
        similarity=dict(
            cls='cos',
            transformation=tune.choice(['bound_inverse', 'negative']),
        ),
        loss=dict(
            cls='sampled',
            **default_sampled_matching_loss_hpo,
        ),
        optimizer=dict(
            cls='adam',
            lr=tune.loguniform(1.0e-04, 1.0e+00),
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
        GCNAlignMatching,
        name='gcn_align_matching',
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
