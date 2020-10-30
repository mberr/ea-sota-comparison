"""
Label-based embedding initialization.

Comprises the following steps:

1. Label preprocessing

2. Tokenization

3. Embedding lookup

4. Fallback out-of-vocabulary initialization

5. Token Embedding Pooling

"""
import json
import logging
import pathlib
import urllib.parse
import zipfile
from operator import itemgetter
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas
import requests
import torch

from .base import PretrainedNodeEmbeddingInitializer
from ....data import KnowledgeGraph, MatchSideEnum, SIDES
from ....data.loaders import dataset_name_normalization
from ....utils.common import invert_mapping
from ....utils.data_utils import check_hashsums, resolve_cache_root, resolve_google_drive_file_url, save_response_content
from ....utils.torch_utils import get_device

logger = logging.getLogger(__name__)


class LabelPreprocessor:
    """Transforms labels."""

    def preprocess(
        self,
        label: str,
        lang: str,
    ) -> Tuple[str, str]:
        """
        Transform a label.

        .. note ::
            `lang` should refer to the current language of the label, i.e. it should be passed through if the language
            didn't change, but should be updated if the language changes (e.g. for label translations).

        :param label:
            The label.
        :param lang:
            The language.

        :return:
            A tuple (new_label, new_lang).
        """
        raise NotImplementedError


class LowerCasePreprocessor(LabelPreprocessor):
    """Return lowercased version of a label."""

    def preprocess(self, label: str, lang: str) -> Tuple[str, str]:  # noqa: D102
        return label.lower(), lang


class ReplaceCharacterPreprocessor(LabelPreprocessor):
    """Replace characters in the label based on translation map."""

    replacement_map: Mapping[str, str]

    def __init__(
        self,
        replacement_map: Optional[Mapping[str, str]] = None,
    ):
        """
        Initialize the preprocessor.

        :param replacement_map:
            The replacement map. Defaults to {'_':  ' '}.
        """
        if replacement_map is None:
            replacement_map = {
                '_': ' '
            }
        self.replacement_map = replacement_map

    def preprocess(self, label: str, lang: str) -> Tuple[str, str]:  # noqa: D102
        for search, replace in self.replacement_map.items():
            label = label.replace(search, replace)
        return label, lang


class URILabelExtractor(LabelPreprocessor):
    """Extract labels from URI."""

    def preprocess(
        self,
        label: str,
        lang: str,
    ) -> Tuple[str, str]:  # noqa: D102
        if label.startswith('http://'):
            label = label.rsplit(sep='/', maxsplit=1)[-1]
            label = urllib.parse.unquote(string=label)
        return label, lang


class BertEnvelopePreprocessor(LabelPreprocessor):
    """Add BERT-specific start and end token."""

    def preprocess(
        self,
        label: str,
        lang: str,
    ) -> Tuple[str, str]:  # noqa: D102
        return f'[CLS] {label} [SEP]', lang


class Tokenizer:
    """Split labels into tokens."""

    def tokenize(
        self,
        label: str,
        lang: str,
    ) -> Sequence[str]:
        """
        Tokenize a label.

        :param label:
            The label.
        :param lang:
            The language.

        :return:
            The list of tokens.
        """
        raise NotImplementedError


class NoTokenizer(Tokenizer):
    """Do not apply tokenization."""

    def tokenize(
        self,
        label: str,
        lang: str,
    ) -> Sequence[str]:  # noqa: D102
        return [label]


class WhiteSpaceTokenizer(Tokenizer):
    """Split by whitespace."""

    def tokenize(self, label: str, lang: str) -> Sequence[str]:  # noqa: D102
        return label.split()


class EmbeddingProvider:
    """Provides embeddings, e.g. from pre-trained language models."""

    def get_token_embeddings(
        self,
        tokens: Sequence[str],
        lang: str,
    ) -> Sequence[Optional[torch.FloatTensor]]:
        """
        Get embeddings for a sequence of tokens.

        :param tokens:
            Tokens to lookup an embedding for.
        :param lang:
            Language of the passed token.

        :return:
            A sequence of vectors for each token. May contain None if the the token is unknown. This should trigger external out-of-vocabulary initialization.
        """
        raise NotImplementedError

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the dimension of each embedding."""
        return None


class BertEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for BERT features."""

    def __init__(
        self,
        model_name: str = 'bert-base-multilingual-cased',
        device: Optional[torch.device] = None,
        cache_dir: Optional[pathlib.Path] = None,
        layers: Sequence[int] = (-4, -3, -2, -1),
    ):
        r"""
        Initialize the provider.

        :param model_name:
            The model name, cf. :meth:`BertTokenizer.from_pretrained`
        :param device:
            The device to use for computing representations.
        :param cache_dir:
            The cache directory where to store the downloaded models.
        :param layers:
            Which layers to select for the representation.

        .. seealso ::
            http://jalammar.github.io/illustrated-bert/
        """
        # http://jalammar.github.io/illustrated-bert/
        # Gist: summing up the last four layers yields good result while keeping the dimension
        # lower (768 vs. 3072 for concat'ing the last four hidden layers).
        from pytorch_pretrained_bert.modeling import BertModel
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        if cache_dir is None:
            cache_dir = pathlib.Path('~', '.kgm', 'bert_prepared').expanduser()
        self.is_lower_case = model_name.endswith('uncased')
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=self.is_lower_case, cache_dir=cache_dir)
        self.device = get_device(device=device)
        self.model = BertModel.from_pretrained(model_name).to(device=device)
        self.model.eval()
        self.layers = layers

    def get_token_embeddings(
        self,
        tokens: Sequence[str],
        lang: str,
    ) -> Sequence[Optional[torch.FloatTensor]]:  # noqa: D102
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.as_tensor(data=[indexed_tokens], device=self.device)
        try:
            # encoded_layers has the format [n_layers, batch_size, n_tokens, hidden_dim]
            encoded_layers = self.model(tokens_tensor)[0]
        except RuntimeError as e:
            raise ValueError(f'Offending label: {tokens}') from e
        return list(torch.stack([encoded_layers[layer] for layer in self.layers], dim=0).sum(dim=0).squeeze(dim=0))

    @property
    def embedding_dim(self) -> Optional[int]:  # noqa: D102
        return 768


class PrecomputedBertEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for precomputed BERT features."""

    #: The token embeddings for each label.
    token_embeddings: Mapping[str, torch.FloatTensor]

    CASED_MODEL_NAME = 'bert-base-multilingual-cased'
    UNCASED_MODEL_NAME = 'bert-base-multilingual-uncased'

    def __init__(
        self,
        graph: KnowledgeGraph,
        side: MatchSideEnum,
        cache_root: pathlib.Path = None,
        cased: bool = True,
    ):
        """
        Initialize the embedding source containing embeddings from multi-lingual BERT.

        :param graph:
            The graph.
        :param cache_root:
            The cache root, where the precomputed embeddings are stored.
        :param cased:
            Whether to use the cased, or uncased version.
        """
        # Note: while there is a pre-trained model for Chinese available, Chinese is also included in the
        # multi-lingual model
        # BERT cased matches differently for Chinese (e.g. '阿' is only matched in uncased, even though
        # '阿'.lower() == '阿')
        self.cache_root = resolve_cache_root(cache_root, "bert_prepared")
        embedding_file_path = self.cache_root / self.get_file_name_from_graph(
            graph=graph,
            cased=cased,
            side=side,
        )
        if not embedding_file_path.is_file():
            raise FileNotFoundError(
                f'Expected raw embeddings file to exist at {embedding_file_path}. '
                f'Generate them for the dataset using `prepare_bert.py`.'
            )
        self.token_embeddings = torch.load(embedding_file_path, map_location='cpu')
        logger.info('Loaded raw embeddings from file: %s', embedding_file_path)

    @staticmethod
    def get_file_name_from_graph(
        graph: KnowledgeGraph,
        side: MatchSideEnum,
        cased: bool,
    ) -> str:
        """Get canonical filename."""
        model_name = PrecomputedBertEmbeddingProvider.CASED_MODEL_NAME if cased else PrecomputedBertEmbeddingProvider.UNCASED_MODEL_NAME
        return f'{dataset_name_normalization(graph.dataset_name)}_{graph.subset_name}_{model_name}_{side.value}.pt'

    def get_token_embeddings(self, tokens: Sequence[str], lang: str) -> Sequence[Optional[torch.FloatTensor]]:  # noqa: D102
        # precomputed: tokens = [label]
        assert len(tokens) == 1
        label = tokens[0]
        return [v for v in self.token_embeddings[label]]

    @property
    def embedding_dim(self) -> Optional[int]:  # noqa: D102
        return next(iter(self.token_embeddings.values()))[0].shape[-1]


class OutOfVocabularyEmbeddingProvider:
    """Provide embeddings for out-of-vocabulary tokens."""

    def __init__(
        self,
        embedding_dim: int = 300,
        stateful: bool = True,
    ):
        """
        Initialize the provider.

        :param embedding_dim:
            The embedding dimension.
        :param stateful:
            Whether the generated OOV vectors are cached. If True, the OOV words / signs receive the same vector when
            occurring another time. The cache is used across languages (e.g. '9' for chinese and english would receive
            the same random vector).
        """
        self.embedding_dim = embedding_dim
        self.oov = dict()
        self.stateful = stateful

    def get_embedding(
        self,
        label: str,
    ) -> torch.FloatTensor:
        """
        Get embedding for token.

        :param label:
            The token.

        :return:
            The vector.
        """
        oov_vector = self.oov.get(label, None)
        if oov_vector is None:
            oov_vector = self.generate()
            if self.stateful:
                self.oov[label] = oov_vector
        return oov_vector

    def update(self, x: torch.FloatTensor) -> None:
        """Update aggregate statistics about the embedding vectors."""

    def generate(self) -> torch.FloatTensor:
        """Generate a new vector."""
        raise NotImplementedError


class ZeroOOV(OutOfVocabularyEmbeddingProvider):
    """Provide zero vector."""

    def generate(self) -> torch.FloatTensor:  # noqa: D102
        return torch.zeros(self.embedding_dim, dtype=torch.float32)


class RandomOOV(OutOfVocabularyEmbeddingProvider):
    """Provide random vectors."""

    def __init__(
        self,
        embedding_dim: int = 300,
        stateful: bool = True,
    ):
        """
        Initialize the module.

        :param embedding_dim: >0
            The embedding dimension.
        :param stateful:
            Whether to be stateful, i.e. re-use the same OOV representation when encountering a OOV token twice.
        """
        super().__init__(embedding_dim=embedding_dim, stateful=stateful)
        self.x = None
        self.x2 = None
        self.count = 0

    def update(self, x: torch.FloatTensor) -> None:  # noqa: D102
        if self.count == 0:
            x_cpy = x.clone().detach()
            self.x = x_cpy
            self.x2 = x_cpy ** 2
        else:
            self.x += x
            self.x2 += x ** 2
        self.count += 1

    @property
    def mean(self) -> Union[float, torch.FloatTensor]:
        """Return the mean."""
        if self.count == 0:
            return 0
        return self.x / self.count

    @property
    def std(self) -> Union[float, torch.FloatTensor]:
        """Return the standard deviation."""
        if self.count == 0:
            return 1.
        return self.x2 / self.count - (self.x / self.count) ** 2

    def generate(self) -> torch.FloatTensor:  # noqa: D102
        m = torch.distributions.Normal(loc=self.mean, scale=self.std)
        vector = m.sample(sample_shape=(self.embedding_dim,))
        return vector


class TokenEmbeddingPooling:
    """Method for pooling a sequence of token representation to a single representation."""

    def __init__(
        self,
        reduction: Callable[[torch.FloatTensor, int], torch.FloatTensor],
    ):
        """
        Initialize the module.

        :param reduction:
            The reduction method.
        """
        self.reduction = reduction

    def pool(
        self,
        token_embeddings: Sequence[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """
        Pool token embeddings into a single vector.

        :param token_embeddings:
            The individual token embeddings, each of shape: (dim,)

        :return: shape: (dim,)
            A single pooled token sequence representations.
        """
        result = self.reduction(torch.stack(tensors=token_embeddings, dim=0), 0)
        if isinstance(result, tuple):
            result = result[0]
        return result


MeanPooling = TokenEmbeddingPooling(reduction=torch.mean)
SumPooling = TokenEmbeddingPooling(reduction=torch.sum)
MaxPooling = TokenEmbeddingPooling(reduction=torch.max)
FirstPooling = TokenEmbeddingPooling(reduction=lambda x, d: x[0])


def prepare_label_based_embeddings(
    source: EmbeddingProvider,
    pooling: TokenEmbeddingPooling,
    label_preprocessors: Union[None, LabelPreprocessor, List[LabelPreprocessor]] = None,
    tokenizer: Optional[Tokenizer] = None,
    oov: Optional[OutOfVocabularyEmbeddingProvider] = None,
    graph: Optional[KnowledgeGraph] = None,
    labels: Sequence[str] = None,
    lang_code: Optional[str] = None,
) -> PretrainedNodeEmbeddingInitializer:
    """
    Prepare label-based embeddings.

    :param source:
        The source embedding provider yielding a vector for each token.
    :param pooling:
        The token pooling mechanism to reduce a sequence of token embeddings into a single fixed-size vector.
    :param label_preprocessors:
        Label preprocessors. Transform label, potentially also changing the language.
    :param tokenizer:
        Split a label into tokens.
    :param oov:
        The out-of-vocabulary embedding provider, when the token is not in the vocabulary.
    :param graph:
        The graph. If None, labels and lang_code must be provided.
    :param labels:
        The labels.
    :param lang_code:
        The language code.

    :return:
        A pretrained embedding initializer.
    """
    # Input normalization
    if label_preprocessors is None:
        label_preprocessors = []
    if isinstance(label_preprocessors, LabelPreprocessor):
        label_preprocessors = [label_preprocessors]

    if graph is not None:
        lang_code = graph.lang_code
        labels = list(map(itemgetter(0), sorted(graph.entity_label_to_id.items(), key=itemgetter(1))))

    embedding_dim = source.embedding_dim
    if embedding_dim is None:
        logger.warning(f'{source} does not provide embedding_dim; falling back to OOV method\'s information.')
        embedding_dim = oov.embedding_dim
    x = torch.empty(len(labels), embedding_dim, dtype=torch.float32, device='cpu')
    for i, label in enumerate(labels):
        label_, lang_ = (label, lang_code)
        for preprocessor in label_preprocessors:
            label_, lang_ = preprocessor.preprocess(label=label_, lang=lang_)
        if tokenizer is None:
            tokens = [label_]
        else:
            tokens = tokenizer.tokenize(label=label_, lang=lang_)
        vs = list(source.get_token_embeddings(tokens=tokens, lang=lang_))
        for j in range(len(vs)):
            if vs[j] is None:
                if oov is None:
                    raise ValueError(f'{vs[j]} is not in vocabulary, but no OOV method was provided.')
                vs[j] = oov.get_embedding(label=tokens[j])
        x[i, :] = pooling.pool(token_embeddings=vs)
    return PretrainedNodeEmbeddingInitializer(embeddings=x)


def get_wu_word_embeddings(
    graph: KnowledgeGraph,
    cache_root: Optional[pathlib.Path] = None,
) -> PretrainedNodeEmbeddingInitializer:
    """
    Get RDGCN node initialization with word embeddings by Wu et al.

    :param graph:
        The graph.
    :param cache_root:
        The cache root. Defaults to ~/.kgm.

    :return:
        A pretrained node embedding initializer.
    """
    cache_root = resolve_cache_root(cache_root, "wu")

    # download raw data if necessary
    raw_path = cache_root / "raw.zip"
    file_id = "1qBKkTPlzPMgvLsC5QWTe0IpEHpSNyRD0"
    sha512 = "e5e1999cfd919d3370d75b413cda3fddfca64e971dc207077f7d93dab4d83a04a90e42e9b81cf52f814f82859f8ec498147cd37ab2cce35a7d07009a89c18de3"
    if not (
        raw_path.is_file() and check_hashsums(destination=raw_path, sha512=sha512)
    ):
        logger.info("Downloading data.")
        with requests.Session() as session:
            response = resolve_google_drive_file_url(id_=file_id, session=session)
            save_response_content(response=response, destination=raw_path)

    subset_name = graph.subset_name
    i = subset_name.split('_').index(graph.lang_code)
    side = SIDES[i]
    subset_root = cache_root / subset_name
    subset_root.mkdir(parents=True, exist_ok=True)
    result_path = PretrainedNodeEmbeddingInitializer.output_file_path(directory=subset_root, side=side)
    if result_path.is_file():
        return PretrainedNodeEmbeddingInitializer.from_path(directory=subset_root, side=side)

    # load word embeddings
    foreign_language = subset_name.split('_')[0]
    with zipfile.ZipFile(raw_path, mode="r") as zf:
        with zf.open(str(pathlib.PurePath("data", subset_name, f'{foreign_language}_vectorList.json')), "r") as json_file:
            emb = torch.as_tensor(data=json.load(json_file), dtype=torch.float32)
        logger.info(f'Loaded embeddings of shape {emb.shape}')

        with zf.open(str(pathlib.PurePath("data", subset_name, f"ent_ids_{i + 1}"))) as id_file:
            their_id_df = pandas.read_csv(
                id_file,
                sep="\t",
                header=None,
                names=["id", "label"],
            )

    # load entity-id to label
    their_label_to_id = dict(zip(their_id_df['label'].tolist(), their_id_df['id'].tolist()))
    assert set(their_label_to_id.keys()) == set(graph.entity_label_to_id.keys())

    # match via labels with our IDs
    our_id_to_label = invert_mapping(mapping=graph.entity_label_to_id)
    id_translation = {
        our_id: their_label_to_id[label]
        for our_id, label in our_id_to_label.items()
    }

    # set embeddings
    init_data = torch.empty(graph.num_entities, 300, dtype=torch.float32)
    for our_id, their_id in id_translation.items():
        init_data[our_id] = emb[their_id]

    initializer = PretrainedNodeEmbeddingInitializer(embeddings=init_data)
    initializer.save_to_path(directory=subset_root, side=side)
    return initializer


def get_xu_word_embeddings(
    graph: KnowledgeGraph,
    cache_root: Optional[pathlib.Path] = None,
) -> PretrainedNodeEmbeddingInitializer:
    """
    Get word embedding initialization for DBP15k (JAPE) used by Xu, as done in Pytorch geometric.

    :param graph:
        The graph.
    :param cache_root:
        The cache root. Defaults to ~/.kgm

    :return:
        A node embedding initializer.
    """
    if dataset_name_normalization(name=graph.dataset_name) != "dbp15kjape":
        raise ValueError(f"Xu embeddings are only available for dataset_name='dbp15k_jape', but dataset_name={graph.dataset_name}")

    cache_root = resolve_cache_root(cache_root, "xu")

    # download raw data if necessary
    raw_path = cache_root / "raw.zip"
    file_id = "1dYJtj1_J4nYJdrDY95ucGLCuZXDXI7PL"
    sha512 = "4d080b69db96395833ff684a66f917f4bafa78549f6152b3bd1f72caf6cffb56b6f4ec02523915ef70410d8052105cdbed74fd8b97feef943efdc2b6a871dbac"
    if not (
        raw_path.is_file() and check_hashsums(destination=raw_path, sha512=sha512)
    ):
        logger.info("Downloading data.")
        with requests.Session() as session:
            response = resolve_google_drive_file_url(id_=file_id, session=session)
            save_response_content(response=response, destination=raw_path)

    word_embedding_path = cache_root / "word_embeddings.pt"
    if not word_embedding_path.is_file():
        logger.info("Extracting word embeddings.")
        word_embeddings = dict()
        with zipfile.ZipFile(file=raw_path, mode="r") as zf:
            with zf.open(name=str(pathlib.PurePath("DBP15K", "sub.glove.300d")), mode="r") as word_emb_file:
                for line in word_emb_file.readlines():
                    info = line.strip().decode(encoding="utf8").split(" ")
                    if len(info) > 300:
                        key = info[0]
                        values = info[1:]
                    else:
                        key = "**UNK**"
                        values = info
                    word_embeddings[key] = torch.as_tensor(
                        data=list(map(float, values)),
                        dtype=torch.float32,
                    )
        torch.save(word_embeddings, word_embedding_path)
    else:
        word_embeddings = torch.load(word_embedding_path)

    subset_root = cache_root / graph.subset_name
    subset_root.mkdir(parents=True, exist_ok=True)

    # load labels
    i = graph.subset_name.split('_').index(graph.lang_code)
    label_path = cache_root / graph.subset_name / f"{graph.lang_code}_alignment.tsv"
    if label_path.is_file():
        df = pandas.read_csv(
            label_path,
            sep="\t",
            encoding="utf8",
        )
    else:
        with zipfile.ZipFile(file=raw_path, mode="r") as zf:
            internal_id_path = pathlib.PurePath("DBP15K", graph.subset_name, f"ent_ids_{i + 1}")
            with zf.open(name=str(internal_id_path), mode="r") as id_file:
                id_df = pandas.read_csv(
                    id_file,
                    sep="\t",
                    header=None,
                    names=["their_id", "uri"],
                    dtype=dict(
                        their_id=int,
                        uri=str,
                    ),
                    encoding="utf8",
                )
            internal_label_path = pathlib.PurePath("DBP15K", graph.subset_name, f"id_features_{i + 1}")
            with zf.open(name=str(internal_label_path), mode="r") as label_file:
                data = []
                for line in label_file:
                    info = line.decode("utf8").strip().split('\t')
                    info = info if len(info) == 2 else info + ['**UNK**']
                    their_id, label = info
                    data.append((their_id, label))
                label_df = pandas.DataFrame(
                    data=data,
                    columns=["their_id", "label"],
                )
                label_df["their_id"] = pandas.to_numeric(label_df["their_id"])
            # align IDs
            our_df = pandas.DataFrame(
                data=list(graph.entity_label_to_id.items()),
                columns=["uri", "our_id"],
            )
            df = our_df.merge(
                right=id_df,
                how="outer",
                on="uri"
            ).merge(
                right=label_df,
                how="outer",
                on="their_id",
            )

            df = df.dropna(subset=["our_id"])
            df.to_csv(
                label_path,
                sep="\t",
                index=False,
            )

    assert not df["their_id"].isna().any()
    selection = df[df["label"] == "**UNK**"]
    if len(selection) > 0:
        logger.warning(
            f'{graph.dataset_name}:{graph.subset_name}:{graph.lang_code}: Could not get label for \n'
            f'{selection}.'
        )

    # allocate array
    side = SIDES[i]
    result_path = PretrainedNodeEmbeddingInitializer.output_file_path(directory=subset_root, side=side)
    if result_path.is_file():
        return PretrainedNodeEmbeddingInitializer.from_path(directory=subset_root, side=side)

    result = torch.empty(graph.num_entities, 300, dtype=torch.float32)

    # aggregate word embeddings
    unknown = word_embeddings["**UNK**"]
    for _, row in df.iterrows():
        entity_id = row["our_id"]
        entity_label = row["label"]
        result[entity_id, :] = torch.stack([
            word_embeddings.get(word, unknown)
            for word in entity_label.strip().lower().split()
        ], dim=0).sum(dim=0)

    initializer = PretrainedNodeEmbeddingInitializer(embeddings=result)
    initializer.save_to_path(directory=subset_root, side=side)
    return initializer
