"""Precompute BERT embeddings."""
import argparse
import functools
import logging
import pathlib
from operator import itemgetter
from typing import Sequence
from unittest.mock import Mock

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP
from tqdm import tqdm

from kgm.data import SIDES, available_datasets, get_dataset_by_name
from kgm.modules.embeddings.init.label import BertEmbeddingProvider, BertEnvelopePreprocessor, EmbeddingProvider, LabelPreprocessor, LowerCasePreprocessor, PrecomputedBertEmbeddingProvider, URILabelExtractor


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name='kgm').setLevel(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    # Dataset options
    all_available_datasets = available_datasets()
    all_datasets = sorted(all_available_datasets.keys())
    parser.add_argument('--dataset', type=str, choices=all_datasets, default=None)
    parser.add_argument('--subset', type=str, choices=sorted(sum(map(list, all_available_datasets.values()), [])), default=None)
    # BERT options
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased', choices=PRETRAINED_VOCAB_ARCHIVE_MAP.keys())
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device {device}')

    # Input normalization
    output_root = args.output
    if output_root is None:
        output_root = pathlib.Path('~', '.kgm', 'bert_prepared')
    output_root = pathlib.Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    logging.info(f'Output to {str(output_root)}')

    tasks = []
    datasets = [args.dataset]
    if datasets[0] is None:
        datasets = all_datasets
    for dataset in datasets:
        if args.subset is None:
            subsets = all_available_datasets[dataset]
        else:
            subsets = [args.subset]
        tasks.extend((dataset, subset) for subset in subsets)
    print(f'Scheduled {len(tasks)} tasks.')

    # BERT Setup
    model_name = args.model
    provider = BertEmbeddingProvider(
        model_name=model_name,
        device=device,
        cache_dir=output_root,
    )
    tokenizer = provider.tokenizer

    # filter existing files
    if not args.force:
        actual_tasks = [
            (dataset, subset)
            for dataset, subset in tasks
            if not all(
                (output_root / PrecomputedBertEmbeddingProvider.get_file_name_from_graph(
                    graph=Mock(dataset_name=dataset, subset_name=subset),
                    cased=not provider.is_lower_case,
                    side=side,
                )).is_file()
                for i_side, side in enumerate(SIDES)
            )
        ]
        print(f'Removed {len(tasks) - len(actual_tasks)} tasks due to existing files.')
        tasks = actual_tasks

    label_preprocessors = [
        URILabelExtractor()
    ]
    if provider.is_lower_case:
        label_preprocessors.append(LowerCasePreprocessor())
    label_preprocessors.append(BertEnvelopePreprocessor())

    task_progress = tqdm(tasks, unit='task', unit_scale=True)
    for dataset, subset in task_progress:
        task_progress.set_postfix(dict(dataset=dataset, subset=subset))
        # load dataset
        dataset = get_dataset_by_name(dataset_name=dataset, subset_name=subset)
        side_progress = tqdm(dataset.graphs.items(), total=2, unit='side')
        for side, graph in side_progress:
            side_progress.set_postfix(dict(side=side.value))
            output_path = output_root / PrecomputedBertEmbeddingProvider.get_file_name_from_graph(
                graph=graph,
                cased=not provider.is_lower_case,
                side=side,
            )
            if not output_path.is_file() or args.force:
                torch.save(
                    obj={
                        label:
                            get_raw_embedding(
                                label=label,
                                tokenizer=tokenizer,
                                provider=provider,
                                label_preprocessors=label_preprocessors,
                                lang=graph.lang_code,
                            )
                        for label in tqdm(
                            map(
                                itemgetter(0),
                                sorted(graph.entity_label_to_id.items(), key=itemgetter(1))
                            ),
                            unit='label',
                            unit_scale=True,
                            total=graph.num_entities,
                        )
                    },
                    f=output_path,
                )
    logging.info('Finished successfully.')


def get_raw_embedding(
    label: str,
    lang: str,
    tokenizer: BertTokenizer,
    provider: EmbeddingProvider,
    label_preprocessors: Sequence[LabelPreprocessor],
) -> torch.FloatTensor:
    label = functools.reduce(lambda acc, prep: prep.preprocess(*acc), label_preprocessors, (label, None))[0]
    tokens = tokenizer.tokenize(text=label)
    return provider.get_token_embeddings(tokens=tokens, lang=lang)


if __name__ == '__main__':
    main()
