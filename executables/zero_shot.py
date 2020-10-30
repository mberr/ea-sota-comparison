"""Zero-Shot evaluation script."""
import argparse
import logging

import mlflow
import torch

from kgm.trainables.search_space import INITIALIZATIONS, SUBSETS
from kgm.data import get_dataset_by_name
from kgm.eval import evaluate_matching_model
from kgm.models import PureEmbeddingModel
from kgm.modules import get_similarity
from kgm.utils.mlflow_utils import log_metrics_to_mlflow, log_params_to_mlflow
from kgm.utils.torch_utils import get_device

SIMILARITIES = (
    ('l1', 'negative'),
    # Do not evaluate different transformations, since we do not learn anything, and the order is the same for all transformations
    # ('l1', 'bound_inverse'),
    ('l2', 'negative'),
    # ('l2', 'bound_inverse'),
    ('cos', None),
    ('dot', None),
)


@torch.no_grad()
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(experiment_name="zero_shot")

    # increase log level
    logger = logging.getLogger("kgm")
    logger.setLevel(level=logging.INFO)
    # resolve device
    device = get_device(device=None)
    logger.info(f"Using device = {device}")

    for dataset_name, subset_names in SUBSETS.items():
        for subset_name in subset_names:
            logger.info(f"Loading dataset: {dataset_name}:{subset_name}")

            # Load dataset
            dataset = get_dataset_by_name(
                dataset_name=dataset_name,
                subset_name=subset_name,
            )
            # Dataset specific evaluation batch size
            eval_batch_size = None

            for init in INITIALIZATIONS[dataset_name]:
                logger.info(f"Instantiating model: {init.value}")

                # Instantiate model
                model = PureEmbeddingModel(
                    dataset=dataset,
                    embedding_dim=None,
                    node_embedding_init_method=init,
                ).to(device=device)

                for similarity_name, transformation_name in SIMILARITIES:
                    logger.info(f"Instantiating similarity: {similarity_name}")

                    # Instantiate similarity
                    similarity = get_similarity(similarity=similarity_name, transformation=transformation_name)

                    # Prepare config
                    config = dict(
                        init=init,
                        dataset=dataset_name,
                        subset=subset_name,
                        similarity=similarity_name,
                        transformation=transformation_name,
                    )

                    # start experiment and log config
                    mlflow.start_run()
                    log_params_to_mlflow(config=config)

                    # Evaluate model
                    result, eval_batch_size = evaluate_matching_model(
                        model=model,
                        alignments=dataset.alignment.to_dict(),
                        similarity=similarity,
                        eval_batch_size=eval_batch_size,
                    )

                    # log results and stop experiment
                    log_metrics_to_mlflow(metrics=result)
                    mlflow.end_run()


if __name__ == '__main__':
    main()
