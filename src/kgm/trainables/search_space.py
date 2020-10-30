"""Common HPO settings."""
import random

from ray import tune

from kgm.modules.embeddings.base import NodeEmbeddingInitMethod
from kgm.utils.tune_utils import tune_linear_quantized_range

SUBSETS = {
    "dbp15k_jape": ["zh_en", "ja_en", "fr_en"],
    "wk3l15k": ["en_de", "en_fr"],
    "openea": ["D_Y_15K_V2", "D_W_15K_V2", "EN_DE_15K_V2", "EN_FR_15K_V2"],
}
INITIALIZATIONS = {
    "dbp15k_jape": [
        NodeEmbeddingInitMethod.xu_precomputed,
        NodeEmbeddingInitMethod.rdgcn_precomputed,
    ],
    "wk3l15k": [
        NodeEmbeddingInitMethod.bert_precomputed,
    ],
    "openea": [
        NodeEmbeddingInitMethod.openea_rdgcn_precomputed,
    ]
}
DIMENSIONS = {
    NodeEmbeddingInitMethod.xu_precomputed: 300,
    NodeEmbeddingInitMethod.rdgcn_precomputed: 300,
    NodeEmbeddingInitMethod.openea_rdgcn_precomputed: 300,
    NodeEmbeddingInitMethod.bert_precomputed: 768,
}
NODE_EMBEDDING_INIT_CONFIG = {
    NodeEmbeddingInitMethod.xu_precomputed: None,
    NodeEmbeddingInitMethod.rdgcn_precomputed: None,
    NodeEmbeddingInitMethod.openea_rdgcn_precomputed: None,
    NodeEmbeddingInitMethod.bert_precomputed: dict(
        pooling=tune.choice(['mean', 'sum', 'max'])
    ),
}


def _sample_init(spec):
    return random.choice(INITIALIZATIONS[spec.config.data.dataset])


def _resolve_embedding_dim(spec):
    return DIMENSIONS[spec.config.model.node_embedding_init_method]


def _resolve_node_embedding_init_config(spec):
    return NODE_EMBEDDING_INIT_CONFIG[spec.config.model.node_embedding_init_method]


def _resolve_subsets(spec):
    return random.choice(SUBSETS[spec.config.data.dataset])


def _tune_linear_quantized_range(low=None, q=32):
    def _tune(spec):
        high = DIMENSIONS[spec.config.model.node_embedding_init_method]
        return tune_linear_quantized_range(high=high, low=low, q=q)

    return _tune
