"""Data loading for Entity Alignment datasets."""
import abc
import io
import json
import logging
import lzma
import pathlib
import tarfile
import zipfile
from typing import Collection, Generic, Mapping, Optional, Tuple, Type, TypeVar, Union

import numpy
import pandas
import requests
import torch

from .knowledge_graph import EntityAlignment, KnowledgeGraph, KnowledgeGraphAlignmentDataset, MatchSideEnum, SIDES
from ..utils.common import get_all_subclasses, get_subclass_by_name, multi_hash
from ..utils.data_utils import check_hashsums, resolve_cache_root, resolve_google_drive_file_url, save_response_content
from ..utils.torch_utils import split_tensor
from ..utils.types import IDAlignment, Triples

A = TypeVar('A', zipfile.ZipFile, tarfile.TarFile)

logger = logging.getLogger(name=__name__)


class Archive(Generic[A]):
    """A generic class for reading from archives."""

    #: The archive file
    archive_file: A

    #: The default file extension:
    default_file_extension: str

    def __init__(self, archive_path: pathlib.Path):
        """
        Initialize the archive.

        :param archive_path:
            The archive path.
        """
        self.path = archive_path

    def __enter__(self):  # noqa: D105
        self.archive_file = self._open_archive(path=self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D105
        self.archive_file.close()

    # pylint: disable=unused-argument
    def open_file(
        self,
        relative_path: Union[pathlib.Path, str],
        encoding: Optional[str] = None,
    ) -> io.TextIOBase:
        """Open a file from the archive in read mode."""
        return self.archive_file.open(name=str(relative_path))

    def _open_archive(self, path: pathlib.Path) -> A:
        """Open the archive in read mode."""
        raise NotImplementedError


class ZipArchive(Archive[zipfile.ZipFile]):
    """A zipfile archive."""

    default_file_extension = 'zip'

    def _open_archive(
        self,
        path: pathlib.Path,
    ) -> zipfile.ZipFile:  # noqa: D102
        return zipfile.ZipFile(file=path)


class TarArchive(Archive[tarfile.TarFile]):
    """A tarfile archive."""

    default_file_extension = 'tar.gz'

    def _open_archive(
        self,
        path: pathlib.Path,
    ) -> tarfile.TarFile:  # noqa: D102
        return tarfile.open(name=path)

    def open_file(
        self,
        relative_path: Union[pathlib.Path, str],
        encoding: Optional[str] = None,
    ) -> io.TextIOBase:  # noqa: D102
        return io.TextIOWrapper(self.archive_file.extractfile(member=str(relative_path)), encoding=encoding)


def apply_compaction(
    triples: Triples,
    compaction: Mapping[int, int],
    columns: Union[int, Collection[int]],
    dim: int = 0,
) -> Triples:
    """
    Apply ID compaction to triples.

    :param triples:
        The triples
    :param compaction:
        The ID compaction, i.e. mapping old ID to new ID.
    :param columns:
        The columns on which to apply the compaction.
    :param dim:
        The dimension along which to apply the compaction.

    :return:
        The updated triples.
    """
    if compaction is None:
        return triples
    if isinstance(columns, int):
        columns = [columns]
    if dim not in {0, 1}:
        raise KeyError(dim)
    triple_shape = triples.shape
    if dim == 1:
        triples = triples.t()
    new_cols = []
    for c in range(triples.shape[1]):
        this_column = triples[:, c]
        if c in columns:
            new_cols.append(torch.tensor([compaction[int(e)] for e in this_column]))  # pylint: disable=not-callable
        else:
            new_cols.append(this_column)
    new_triples = torch.stack(new_cols, dim=1 - dim)
    assert new_triples.shape == triple_shape
    return new_triples


def compact_columns(
    triples: Triples,
    label_to_id_mapping: Mapping[str, int],
    columns: Union[int, Collection[int]],
) -> Tuple[Triples, Optional[Mapping[str, int]], Optional[Mapping[int, int]]]:
    """
    Calculate compaction of the columns of triples.

    :param triples: shape: (num_triples, 3)
        The original triples.
    :param label_to_id_mapping:
        The old label-to-ID mapping.
    :param columns:
        The columns on which to calculate the compaction.

    :return:
        A 3-tuple (new_triples, new_mapping, compaction) where
        * new_triples: shape: (num_triples, 3)
            The compacted triples.
        * new_mapping:
            The updated label to ID mapping.
        * compaction:
            A mapping old ID to new ID.

        Note: new_mapping and compaction may be None, if the old triples where already compact.
    """
    ids = label_to_id_mapping.values()
    num_ids = len(ids)
    assert len(set(ids)) == len(ids)
    max_id = max(ids)
    if num_ids < max_id + 1:
        compaction = dict((old, new) for new, old in enumerate(sorted(ids)))
        assert set(compaction.keys()) == set(label_to_id_mapping.values())
        assert set(compaction.values()) == set(range(num_ids))
        new_triples = apply_compaction(triples, compaction, columns, dim=0)
        new_mapping = {label: compaction[_id] for label, _id in label_to_id_mapping.items()}
        logger.info('Compacted: %d -> %d', max_id, num_ids - 1)
    else:
        compaction = None
        new_triples = triples
        new_mapping = label_to_id_mapping
        logger.debug('No compaction necessary.')
    return new_triples, new_mapping, compaction


def compact_graph(
    graph: KnowledgeGraph,
    no_duplicates: bool = True,
) -> Tuple[KnowledgeGraph, Optional[Mapping[int, int]], Optional[Mapping[int, int]]]:
    """
    Compact a KG.

    :param graph:
        The KG.
    :param no_duplicates:
        Whether to drop duplicates.

    :return:
        The updated KG, and mappings from old ID to compact ID, or None if the KG is already compliant.
    """
    if graph.inverse_triples:
        raise NotImplementedError

    triples0 = graph.triples

    # Compact entities
    triples1, compact_entity_label_to_id, entity_compaction = compact_columns(triples=triples0, label_to_id_mapping=graph.entity_label_to_id, columns=(0, 2))

    # Compact relations
    triples2, compact_relation_label_to_id, relation_compaction = compact_columns(triples=triples1, label_to_id_mapping=graph.relation_label_to_id, columns=(1,))

    # Filter duplicates
    if no_duplicates:
        old_size = triples2.shape[0]
        triples2 = torch.unique(triples2, dim=0)
        new_size = triples2.shape[0]
        if new_size < old_size:
            logger.info('Aggregated edges: %d -> %d.', old_size, new_size)

    # Compile to new knowledge graph
    compact_graph_ = KnowledgeGraph(
        triples=triples2,
        entity_label_to_id=compact_entity_label_to_id,
        relation_label_to_id=compact_relation_label_to_id,
        lang_code=graph.lang_code,
        dataset_name=graph.dataset_name,
        subset_name=graph.subset_name
    )

    return compact_graph_, entity_compaction, relation_compaction


def compact_single_alignment(
    single_alignment: IDAlignment,
    left_compaction: Mapping[int, int],
    right_compaction: Mapping[int, int],
) -> IDAlignment:
    """
    Apply ID compaction to a single alignment.

    :param single_alignment: shape: (2, num_alignments), dtype: long
        The alignment.
    :param left_compaction:
        The compaction for the left side, i.e. a mapping old ID -> new ID for the left graph.
    :param right_compaction:
        The compaction for the right side, i.e. a mapping old ID -> new ID for the right graph.

    :return: shape: (2, num_alignments)
        The updated alignment.
    """
    compact_single_alignment_ = single_alignment
    for col, compaction in enumerate([left_compaction, right_compaction]):
        compact_single_alignment_ = apply_compaction(triples=compact_single_alignment_, compaction=compaction, columns=col, dim=1)
    return compact_single_alignment_


def compact_knowledge_graph_alignment(
    alignment: EntityAlignment,
    left_entity_compaction: Mapping[int, int],
    right_entity_compaction: Mapping[int, int],
) -> EntityAlignment:
    """
    Apply ID compaction to entity alignment.

    :param alignment:
        The entity alignment.
    :param left_entity_compaction:
        The compaction for the left side, i.e. a mapping old ID -> new ID for the left graph.
    :param right_entity_compaction:
        The compaction for the right side, i.e. a mapping old ID -> new ID for the right graph.

    :return:
        The updated entity alignment.
    """
    # Entity compaction
    compact_entity_alignment_train = compact_single_alignment(single_alignment=alignment.train, left_compaction=left_entity_compaction, right_compaction=right_entity_compaction)
    compact_entity_alignment_test = compact_single_alignment(single_alignment=alignment.test, left_compaction=left_entity_compaction, right_compaction=right_entity_compaction)

    if alignment.num_validation > 0:
        compact_entity_alignment_val = compact_single_alignment(single_alignment=alignment.validation, left_compaction=left_entity_compaction, right_compaction=right_entity_compaction)
    else:
        compact_entity_alignment_val = None

    return EntityAlignment(
        train=compact_entity_alignment_train,
        test=compact_entity_alignment_test,
        _validation=compact_entity_alignment_val,
    )


def compact_knowledge_graph_alignment_dataset(
    left_graph: KnowledgeGraph,
    right_graph: KnowledgeGraph,
    alignment: EntityAlignment,
    no_duplicates: bool = True,
) -> Tuple[KnowledgeGraph, KnowledgeGraph, EntityAlignment]:
    """
    Compact a knowledge graph alignment dataset.

    When loading a KG with pre-defined label-to-ID mappings, it might happen that the ID range is not consecutive, or starts from 0.
    Thus, a compaction is applied by mapping the IDs monotonously to {0, ..., num_labels - 1}.

    :param left_graph:
        The left KG.
    :param right_graph:
        The right KG.
    :param alignment:
        The entity alignment.
    :param no_duplicates:
        Whether to discard duplicate triples.

    :return:
        The updated left/right graph and alignment.
    """
    left_compact_graph, left_entity_compaction = compact_graph(graph=left_graph, no_duplicates=no_duplicates)[:2]
    right_compact_graph, right_entity_compaction = compact_graph(graph=right_graph, no_duplicates=no_duplicates)[:2]
    compact_alignment = compact_knowledge_graph_alignment(
        alignment=alignment,
        left_entity_compaction=left_entity_compaction,
        right_entity_compaction=right_entity_compaction,
    )
    return left_compact_graph, right_compact_graph, compact_alignment


def load_triples(
    triples_file: io.TextIOBase,
    delimiter: str = '\t',
    encoding: str = 'utf8',
    engine: str = 'c',
) -> Tuple[Triples, Mapping[str, int], Mapping[str, int]]:
    """
    Load triples from a file-like object.

    :param triples_file:
        The opened file-like object.
    :param delimiter:
        The delimiter.
    :param encoding:
        The encoding,
    :param engine:
        The pandas engine.
    :return:
        A tuple (triples, entity_label_to_id, relation_label_to_id) where
        * triples: shape: (num_triples, 3), dtype: long
        * entity_label_to_id / relation_label_to_id: mapping from labels to IDs.
    """
    # Load triples from tsv file
    df = pandas.read_csv(
        filepath_or_buffer=triples_file,
        sep=delimiter,
        encoding=encoding,
        header=None,
        names=['h', 'r', 't'],
        engine=engine,
        dtype=str,
    )
    df = df.applymap(str)

    # Sorting ensures consistent results when the triples are permuted
    entity_label_to_id = {
        e: i for i, e in enumerate(sorted(set(df['h'].unique()).union(set(df['t'].unique()))))
    }
    relation_label_to_id = {
        r: i for i, r in enumerate(sorted(df['r'].unique()))
    }

    # Label triples to ID
    for col, mapping in zip('hrt', [entity_label_to_id, relation_label_to_id, entity_label_to_id]):
        df[col] = df[col].apply(mapping.__getitem__)

    triples = torch.as_tensor(data=df.values, dtype=torch.long).unique(dim=0)

    # Log some info
    logger.info(
        'Loaded %d unique triples, with %d unique entities and %d unique relations.',
        triples.shape[0],
        len(entity_label_to_id),
        len(relation_label_to_id)
    )
    return triples, entity_label_to_id, relation_label_to_id


def _load_label_to_id(
    archive: Archive,
    relative_path: pathlib.Path,
) -> Mapping[str, int]:
    """
    Load entity label to ID file.

    :param archive:
        The opened archive file.
    :param relative_path:
        The relative path within the archive.

    :return:
        A mapping from entity labels to IDs.
    """
    with archive.open_file(relative_path=relative_path) as text_file:
        df = pandas.read_csv(filepath_or_buffer=text_file, names=['id', 'label'], header=None, sep='\t', encoding='utf8', engine='c')
        return dict(zip(df['label'].values.tolist(), df['id'].values.tolist()))


def _load_entity_alignment(
    archive: Archive,
    relative_path: pathlib.Path,
    left_graph: KnowledgeGraph,
    right_graph: KnowledgeGraph,
    sep: str = '\t',
) -> IDAlignment:
    """
    Load entity alignment from an open archive.

    :param archive:
        The opened archive.
    :param relative_path:
        The relative path within the archive.
    :param left_graph:
        The left KG.
    :param right_graph:
        The right KG.

    :return: shape: (2, num_alignments)
        The entity alignment.
    """
    # Load label alignment
    with archive.open_file(relative_path=relative_path) as text_file:
        entity_alignment = pandas.read_csv(
            filepath_or_buffer=text_file,
            names=['L', 'R'],
            header=None,
            sep=sep,
            encoding='utf8',
            engine='c' if len(sep) == 1 else 'python',
            dtype=str,
        )

    return translate_alignment(labelled_entity_alignment=entity_alignment, left_graph=left_graph, right_graph=right_graph)


def translate_alignment(
    labelled_entity_alignment: pandas.DataFrame,
    left_graph: KnowledgeGraph,
    right_graph: KnowledgeGraph,
) -> IDAlignment:
    """
    Convert an alignment of labels to an alignment of IDs.

    :param labelled_entity_alignment: columns: ['L', 'R']
        The entity alignment, label-based.
    :param left_graph:
        The left KG.
    :param right_graph:
        The right KG.

    :return: shape: (2, num_alignments)
        The ID-based alignment.
    """
    # Translate to ID alignment
    alignment = torch.stack(
        [
            torch.as_tensor(
                data=labelled_entity_alignment[col].apply(graph.entity_label_to_id.get, args=(-1,)),
                dtype=torch.long,
            )
            for col, graph in zip('LR', [left_graph, right_graph])
        ],
        dim=0,
    )

    # Drop invalid
    invalid_mask = (alignment < 0).any(dim=0)
    num_invalid = invalid_mask.sum()
    if num_invalid > 0:
        logger.warning('Dropping %d invalid rows.', num_invalid)
    alignment = alignment[:, ~invalid_mask]

    alignment = alignment.unique(dim=1)
    logger.info('Loaded alignment of size %d.', alignment.shape[1])
    return alignment


def _load_tensor_from_csv(
    archive: Archive,
    relative_path: pathlib.Path,
) -> torch.LongTensor:
    """
    Load an integer tensor from a TSV file in an opened archive.

    :param archive:
        The opened archive.
    :param relative_path:
        The relative path within the archive.

    :return: dtype: long
        The tensor.
    """
    with archive.open_file(relative_path=relative_path) as text_file:
        return torch.tensor(  # pylint: disable=not-callable
            data=pandas.read_csv(filepath_or_buffer=text_file, header=None, sep='\t', encoding='utf8', engine='c').values,
            dtype=torch.long,
        )


class OnlineKnowledgeGraphAlignmentDatasetLoader:
    """Contains a lazy reference to a knowledge graph alignment data set."""

    #: The URL where the data can be downloaded from
    url: str

    #: The subsets
    subsets: Collection[str] = frozenset()

    #: The pre-defined train-test splits
    predefined_splits: Collection[float] = frozenset()

    #: The archive file type
    archive_type: Type[Archive] = TarArchive

    #: The file name for the archive
    archive_file_name: str

    #: The directory where the datasets will be extracted to
    cache_root: pathlib.Path

    def __init__(
        self,
        subset: Optional[str] = None,
        train_test_split: Optional[float] = None,
        cache_root: Optional[Union[pathlib.Path, str]] = None,
        compact: bool = True,
        train_validation_split: Optional[float] = 0.8,
        with_inverse_triples: bool = False,
        with_self_loops: bool = False,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize the data loader.

        :param subset:
            The name of the subset to use. Check subsets() for available subsets. If None, use the alphabetically
            first one. This should *not* happen within a production environment.
        :param train_test_split:
            The train-test split ratio.
        :param cache_root:
            The cache root to use for caching downloaded files.
        :param compact:
            Whether to compact the label-to-ID mappings, i.e. ensure that the IDs are consecutive from
            {0, ..., num_labels-1}
        :param train_validation_split:
            The train-validation split ratio.
        :param with_inverse_triples:
            Whether to add inverse triples.
        :param with_self_loops:
            Whether to add self-loops.
        :param random_seed:
            The random seed to use for splitting.
        """
        cache_root = resolve_cache_root(cache_root, self.cache_sub_directory_name)
        logger.info('Using cache_root=%s', cache_root)
        self.cache_root = cache_root

        if subset is None:
            subset = sorted(self.subsets)[0]
            logger.warning('No subset specified. This should not happen in production. Using "%s".', subset)
        if subset not in self.subsets:
            raise ValueError(f'Invalid subset={subset}. Allowed subsets: {self.subsets} (check '
                             f'{self.__class__.__name__}.subsets() for this list).')
        self.subset = subset

        if train_test_split is None:
            train_test_split = 0.3
            logger.warning('No train_test_split was given. Defaulting to 0.3.')
        if train_test_split <= 0.0 or train_test_split >= 1.0:
            raise ValueError(f'Split must be a float with 0 < train_test_split < 1, but train_test_split={train_test_split},')
        if train_test_split not in self.predefined_splits:
            logger.warning('Using a custom train_test_split=%f, and none of the pre-defined: %s.', train_test_split, self.predefined_splits)
        self.train_test_split = train_test_split

        self.compact = compact
        self.train_validation_split = train_validation_split
        self.with_inverse_triples = with_inverse_triples
        self.with_self_loops = with_self_loops
        self.random_seed = random_seed

    @property
    def cache_sub_directory_name(self) -> str:
        """Return the name of the sub-directory within the cache root."""
        return self.__class__.__name__.lower()

    def _get_split_name(self) -> str:
        """Get a unique split name."""
        return str(hash((self.train_validation_split, self.train_test_split, self.random_seed)))

    def load(
        self,
        force_download: bool = False,
    ) -> KnowledgeGraphAlignmentDataset:
        """
        Load the dataset.

        :param force_download:
            Whether to force downloading the file, even if is already exists.

        :return:
            The dataset.
        """
        # Ensure directory exists
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Check if files already exist
        archive_path = self.cache_root / f'{self.archive_file_name}.{self.archive_type.default_file_extension}'  # pylint: disable=no-member
        if archive_path.is_file() and not force_download:
            logger.info('Checking hash sums for existing file %s.', str(archive_path))
            check_sums_match = check_hashsums(destination=archive_path, **self.hash_digests())
            if not check_sums_match:
                logger.warning('Checksums do not match. Forcing download.')
            force_download = not check_sums_match
        else:
            force_download = True

        if force_download:
            # create session
            session = requests.Session()
            if 'drive.google.com' in self.url:
                _id = self.url.split('?id=')[1]
                response = resolve_google_drive_file_url(id_=_id, session=session)
            else:
                logger.info('Requesting dataset from %s', self.url)
                response = session.get(url=self.url, stream=True)

            # Real download
            save_response_content(response=response, destination=archive_path)
            check_sums_match = check_hashsums(destination=archive_path, **self.hash_digests())
            if not check_sums_match:
                raise ValueError('Checksums do not match!')
        else:
            logger.info('Skipping to download from %s due to existing files in %s.', self.url, self.cache_root)

        # Try to load from artifact
        artifact_root = self.cache_root / 'preprocessed' / self.__class__.__name__.lower() / self.subset

        # graphs
        graphs = dict()
        compactions = dict()
        for side in SIDES:
            graph = compaction = "load-from-archive"

            # try to load from artifact
            graph_path = artifact_root / f"{side.value}_graph"
            compaction_path = graph_path / "compaction.json.xz"
            if graph_path.is_dir():
                try:
                    graph = KnowledgeGraph.load(directory=graph_path)
                    logger.info(f"Loaded preprocessed graph from {graph_path}")
                except FileNotFoundError as error:
                    logger.error(f"Error occurred by loading graph from {graph_path}: {error}")
            if compaction_path.is_file():
                with lzma.open(compaction_path, "rt") as json_file:
                    compaction = json.load(json_file)

            # load from archive only if necessary
            if graph == "load-from-archive" or compaction == "load-from-archive":
                with self.archive_type(archive_path=archive_path) as archive:
                    graph = self._load_graph(archive=archive, side=side)
                # compact
                graph, compaction = compact_graph(graph=graph, no_duplicates=True)[:2]
                # save
                graph.save(directory=graph_path)
                with lzma.open(compaction_path, "wt") as json_file:
                    json.dump(
                        compaction,
                        fp=json_file,
                        sort_keys=True,
                        indent=2,
                    )
                logger.info(f"Saved preprocessed graph to {graph_path}")
            assert graph is not None
            graphs[side], compactions[side] = graph, compaction

        left_graph, right_graph = [graphs[side] for side in SIDES]

        # alignment
        # key0 = .
        all_alignment_path = artifact_root / "alignment.pt"
        # key1 = (train_test_split, random_seed)
        train_test_key = multi_hash(self.train_test_split, self.random_seed, hash_function="md5")
        test_indices_path = artifact_root / "splits" / f"test_{train_test_key}.pt"
        test_indices_path.parent.mkdir(parents=True, exist_ok=True)
        # key2 = (train_test_split, train_validation_split, random_seed)
        train_test_validation_key = multi_hash(self.train_test_split, self.random_seed, self.train_validation_split, hash_function="md5")
        train_indices_path = artifact_root / "splits" / f"train_{train_test_validation_key}.pt"
        validation_indices_path = artifact_root / "splits" / f"validation_{train_test_validation_key}.pt"
        if all_alignment_path.is_file():
            all_alignment = torch.load(all_alignment_path)
            num_alignments = all_alignment.shape[1]
            logger.info(f"Loaded {num_alignments} preprocessed alignments from {all_alignment_path}")

            train_validation_indices = None
            if test_indices_path.is_file():
                test_indices = torch.load(test_indices_path)
                logger.info(f"Loaded {test_indices.numel()} preprocessed test indices from {test_indices_path}")
            else:
                # train-test split
                train_validation_indices, test_indices = split_tensor(tensor=torch.randperm(num_alignments), ratios=self.train_test_split, seed=self.random_seed)
                torch.save(test_indices, test_indices_path)
                logger.info(f"Saved {test_indices.numel()} preprocessed test indices to {test_indices_path}")

            validation_indices = None
            if train_indices_path.is_file():
                train_indices = torch.load(train_indices_path)
                logger.info(f"Loaded {train_indices.numel()} preprocessed train indices from {train_indices_path}")
                if self.train_validation_split is not None:
                    validation_indices = torch.load(validation_indices_path)
                    logger.info(f"Loaded {validation_indices.numel()} preprocessed validation indices from {validation_indices_path}")
            else:
                if train_validation_indices is None:
                    train_validation_indices = torch.as_tensor(data=sorted(set(range(num_alignments)).difference(test_indices.tolist())))
                if self.train_validation_split is not None:
                    train_indices, validation_indices = split_tensor(tensor=train_validation_indices, ratios=self.train_validation_split, )
                    torch.save(validation_indices, validation_indices_path)
                    logger.info(f"Saved {validation_indices.numel()} preprocessed validation indices to {validation_indices_path}")
                else:
                    train_indices = train_validation_indices
                torch.save(train_indices, train_indices_path)
                logger.info(f"Saved {train_indices.numel()} preprocessed train indices to {train_indices_path}")

            # Compose alignment
            alignment = EntityAlignment(**{
                part: all_alignment[:, indices]
                for part, indices in dict(
                    train=train_indices,
                    test=test_indices,
                    _validation=validation_indices,
                ).items()
            })
        else:
            # load from archive only if necessary
            with self.archive_type(archive_path=archive_path) as archive:
                alignment = self._load_alignment(archive=archive, left_graph=left_graph, right_graph=right_graph)

            # compact
            alignment = compact_knowledge_graph_alignment(
                alignment=alignment,
                left_entity_compaction=compactions[MatchSideEnum.left],
                right_entity_compaction=compactions[MatchSideEnum.right],
            )

            # (re-)split if necessary
            if self.train_validation_split is not None:
                if round(self.train_validation_split * (alignment.num_train + alignment.num_validation)) == alignment.num_train:
                    logger.debug('Data was already split')
                else:
                    if alignment.num_validation > 0:
                        logger.warning('Re-splitting data.')
                    alignment = alignment.validation_split(train_ratio=self.train_validation_split, seed=self.random_seed)
                    logger.info('Train-Validation-Split')

            # better format for saving
            a = torch.cat([alignment.train, alignment.test, alignment.validation], dim=1)

            # lexicographic sort
            i1 = a[1].argsort()
            i2 = a[0, i1].argsort()
            i = i1[i2]
            i: torch.Tensor
            a = a[:, i]
            torch.save(a, all_alignment_path)
            logger.info(f"Store preprocessed alignments to {all_alignment_path}")

            # inverse
            i = i.argsort()
            i_train, i_test, i_validation = i.split(
                split_size=[
                    alignment.num_train,
                    alignment.num_test,
                    alignment.num_validation,
                ])
            for path, indices in (
                (test_indices_path, i_test),
                (train_indices_path, i_train),
                (validation_indices_path, i_validation),
            ):
                torch.save(indices, path)
                logger.info(f"Store preprocessed split to {path}")

        dataset = KnowledgeGraphAlignmentDataset(
            left_graph=left_graph,
            right_graph=right_graph,
            alignment=alignment,
        )

        if self.with_inverse_triples:
            dataset = dataset.with_inverse_triples()
            logger.info('Created inverse triples')

        if self.with_self_loops:
            dataset = dataset.with_self_loops()
            logger.info('Created self-loops')

        return dataset

    def hash_digests(self) -> Mapping[str, str]:
        """Return the hash digests for file integrity check."""
        return dict()

    def _load_graph(
        self,
        archive: Archive,
        side: MatchSideEnum,
    ) -> KnowledgeGraph:
        """
        Load one graph from an archive.

        :param archive:
            The opened archive.

        :param side:
            The side.

        :return:
            The knowledge graph for this side.
        """
        raise NotImplementedError

    def _load_alignment(
        self,
        archive: Archive,
        left_graph: KnowledgeGraph,
        right_graph: KnowledgeGraph,
    ) -> EntityAlignment:
        """
        Load the entity alignment from an opened archive.

        :param archive:
            The opened archive.
        :param left_graph:
            The left graph.
        :param right_graph:
            The right graph.

        :return:
            The alignment.
        """
        raise NotImplementedError


class _DBP15k(OnlineKnowledgeGraphAlignmentDatasetLoader, abc.ABC):
    """
    Superclass for DBP15k variants.

    The datasets were first described in https://iswc2017.semanticweb.org/wp-content/uploads/papers/MainProceedings/188.pdf

    > We selected DBpedia (2016-04) to build three cross-lingual datasets. DBpedia isa large-scale multi-lingual KB
    > including inter-language links (ILLs) from entities of English version to those in other languages. In our
    > experiments, we extracted 15 thousand ILLs with popular entities from English to Chinese, Japanese and French
    > respectively, and considered them as our reference alignment (i.e., gold standards).  Our  strategy  to  extract
    > datasets  is  that  we  randomly  selected  an ILL pair s.t. the involved entities have at least 4
    > relationship triples and then extracted  relationship  and  attribute  infobox  triples  for  selected  entities.

    This implementation only considers the relationship triples, and NOT the attributes triples.
    """

    subsets = frozenset({'zh_en', 'ja_en', 'fr_en', })


class DBP15kJAPE(_DBP15k):
    """Smaller variant of DBP15k from JAPE repository."""

    url = 'https://github.com/nju-websoft/JAPE/raw/master/data/dbp15k.tar.gz'
    predefined_splits = frozenset({0.1, 0.2, 0.3, 0.4, 0.5})
    archive_file_name = 'dbp15k_jape'

    @property
    def root(self) -> pathlib.Path:
        """Return the relative path within the archive."""
        return pathlib.Path('dbp15k', self.subset, f'0_{str(int(100 * self.train_test_split))[0]}')

    def hash_digests(self) -> Mapping[str, str]:  # noqa: D102
        return dict(
            sha512='a3bcee42dd0ecfd7188be36c57b9ec6d57b2995d0cf6a17e8fd6f302b4e70d2fc354282f7f7130040bcdcc6c7a55eab7a3af4c361fb1fd98c376bda1490e3f9d',
        )

    def _load_graph(
        self,
        archive: Archive,
        side: MatchSideEnum,
    ) -> KnowledgeGraph:  # noqa: D102
        lang_codes = self.subset.split('_')
        lang_code = lang_codes[0] if side == MatchSideEnum.left else lang_codes[1]
        num = 1 if side == MatchSideEnum.left else 2
        triples = _load_tensor_from_csv(archive=archive, relative_path=self.root / f'triples_{num}')
        entity_label_to_id = _load_label_to_id(archive=archive, relative_path=self.root / f'ent_ids_{num}')
        relation_label_to_id = _load_label_to_id(archive=archive, relative_path=self.root / f'rel_ids_{num}')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
            lang_code=lang_code,
            dataset_name='dbp15kjape',
            subset_name=self.subset
        )

    def _load_alignment(
        self,
        archive: Archive,
        left_graph: KnowledgeGraph,
        right_graph: KnowledgeGraph,
    ) -> EntityAlignment:  # noqa: D102
        return EntityAlignment(
            train=_load_tensor_from_csv(archive=archive, relative_path=self.root / 'sup_ent_ids').t(),
            test=_load_tensor_from_csv(archive=archive, relative_path=self.root / 'ref_ent_ids').t(),
        )


class _WK3l(OnlineKnowledgeGraphAlignmentDatasetLoader, abc.ABC):
    """
    Base class for WK3l variants.

    The datasets were first described in https://www.ijcai.org/Proceedings/2017/0209.pdf.

    > WK3l contains English(En),  French (Fr),  and German (De) knowledge graphs under DBpedia's dbo:Person domain,
    > where a part of triples are  aligned  by  verifying  the  ILLs  on  entities,  and  multi-lingual  labels  of  the
    > DBpedia  ontology  on  some  relations. The number of entities in each language is adjusted to obtain two data sets.
    > [...]
    > For both data sets, German graphs are sparser than English and French graphs.
    """

    url = 'https://drive.google.com/open?id=1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z'
    subsets = frozenset({'en_de', 'en_fr'})
    archive_type = ZipArchive
    archive_file_name = 'wk3l'

    @property
    def size(self) -> str:
        """Return the size of the dataset."""
        raise NotImplementedError

    @property
    def cache_sub_directory_name(self) -> str:
        """Return the name of the directory within the cache root."""
        return 'wk3l'

    @property
    def root(self) -> pathlib.Path:
        """Return the relative path within the archive."""
        return pathlib.Path('data', f'WK3l-{self.size}', self.subset)

    def hash_digests(self) -> Mapping[str, str]:  # noqa: D102
        return dict(
            sha512='b5b64db8acec2ef83a418008e8ff6ddcd3ea1db95a0a158825ea9cffa5a3c34a9aba6945674304f8623ab21c7248fed900028e71ad602883a307364b6e3681dc',
        )

    def _load_graph(
        self,
        archive: Archive,
        side: MatchSideEnum,
    ) -> KnowledgeGraph:  # noqa: D102
        lang_codes = self.subset.split('_')
        lang_code = lang_codes[0] if side == MatchSideEnum.left else lang_codes[1]
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        triples_path = self.root / f'P_{lang_code}_v{suffix}.csv'
        with archive.open_file(relative_path=triples_path) as triples_file:
            triples, entity_label_to_id, relation_label_to_id = load_triples(triples_file=triples_file, delimiter='@@@', encoding='utf8', engine='python')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
            lang_code=lang_code,
            dataset_name=f'wk3l{self.size}',
            subset_name=self.subset,
        )

    def _load_alignment(
        self,
        archive: Archive,
        left_graph: KnowledgeGraph,
        right_graph: KnowledgeGraph,
    ) -> EntityAlignment:  # noqa: D102
        # From first ILLs
        lang_left, lang_right = self.subset.split('_')
        suffix = '' if self.size == '15k' else '_' + str(self.size)

        first_ill_id_alignment = _load_entity_alignment(
            archive=archive,
            relative_path=self.root / f'{lang_left}2{lang_right}_fk{suffix}.csv',
            left_graph=left_graph,
            right_graph=right_graph,
            sep='@@@',
        )
        second_ill_id_alignment = _load_entity_alignment(
            archive=archive,
            relative_path=self.root / f'{lang_right}2{lang_left}_fk{suffix}.csv',
            left_graph=right_graph,
            right_graph=left_graph,
            sep='@@@',
        ).flip(0)

        # Load label alignment
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        with archive.open_file(relative_path=self.root / f'P_{self.subset}_v{suffix}.csv') as triples_alignment_file:
            df = pandas.read_csv(
                filepath_or_buffer=triples_alignment_file,
                header=None,
                delimiter='@@@',
                names=['lh', 'lr', 'lt', 'rh', 'rr', 'rt'],
                engine='python',
                dtype=str,
                encoding='utf8',
            )
        df = pandas.DataFrame(data=numpy.concatenate([
            df[['lh', 'rh']].values,
            df[['lt', 'rt']].values,
        ]), columns=['L', 'R'])
        triple_id_alignment = translate_alignment(labelled_entity_alignment=df, left_graph=left_graph, right_graph=right_graph)
        logger.info('Loaded alignment of size %d from triple alignment.', triple_id_alignment.shape[1])

        # Merge alignments
        id_alignment = torch.cat(
            tensors=[
                first_ill_id_alignment,
                second_ill_id_alignment,
                triple_id_alignment,
            ], dim=1
        ).unique(dim=1)
        logger.info('Merged alignments to alignment of size %d.', id_alignment.shape[1])

        # As the split used by MTransE (ILL for testing, triples alignments for training) contains more than 95% test leakage, we use our own split
        return EntityAlignment.from_full_alignment(
            alignment=id_alignment,
            train_test_split=self.train_test_split,
            train_validation_split=self.train_validation_split,
            seed=self.random_seed,
        )


class WK3l15k(_WK3l):
    """The smaller variant of WK3l."""

    @property
    def size(self) -> str:  # noqa: D102
        return '15k'


class OpenEA(OnlineKnowledgeGraphAlignmentDatasetLoader):
    """OpenEA datasets."""

    url = 'https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=1'
    subsets = frozenset({
        f'{left}_{right}_{size}K_V{version}'
        for size in (15, 100)
        for left, right in (('EN', 'DE'), ('EN', 'FR'), ('D', 'Y'), ('D', 'W'))
        for version in (1, 2)
    })
    predefined_splits = frozenset({0.3, })
    archive_type = ZipArchive
    archive_file_name = 'openea'

    def hash_digests(self) -> Mapping[str, str]:  # noqa: D102
        return dict(
            sha512='b7d1465c60130a02e9eec85e43841bb2d777fd2d13133f06c29d5507bc12cb29ebec1065319a5d746fe261ee459dba7c16d61095d7cc7038274dc63932e93029',
        )

    @property
    def root(self) -> pathlib.Path:
        """Return the relative path within the archive."""
        return pathlib.Path('OpenEA_dataset_v1.1', self.subset)

    def _load_graph(self, archive: Archive, side: MatchSideEnum) -> KnowledgeGraph:
        lang_code = 'en'
        num = 1 if side == MatchSideEnum.left else 2
        triples_path = self.root / f'rel_triples_{num}'
        with archive.open_file(relative_path=triples_path) as triples_file:
            triples, entity_label_to_id, relation_label_to_id = load_triples(triples_file=triples_file, delimiter='\t')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
            lang_code=lang_code,
            dataset_name='OpenEA',
            subset_name=self.subset,
        )

    def _load_alignment(self, archive: Archive, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        if self.train_test_split == 0.3:
            return self._load_predefined_alignment(archive, left_graph, right_graph)
        return self._load_custom_alignment(archive, left_graph, right_graph)

    def _load_custom_alignment(self, archive: Archive, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        logger.warning('Using a different split than provided by the dataset.')
        return EntityAlignment.from_full_alignment(
            alignment=_load_entity_alignment(
                archive=archive,
                relative_path=self.root / 'ent_links',
                left_graph=left_graph,
                right_graph=right_graph,
            ),
            train_test_split=self.train_test_split,
            train_validation_split=self.train_validation_split,
            seed=self.random_seed,
        )

    def _load_predefined_alignment(self, archive: Archive, left_graph: KnowledgeGraph, right_graph: KnowledgeGraph) -> EntityAlignment:
        logger.info('Using pre-defined split, fold 1.')
        split_root = self.root / '721_5fold' / '1'
        train, validation, test = [
            _load_entity_alignment(
                archive=archive,
                relative_path=split_root / fn,
                left_graph=left_graph,
                right_graph=right_graph,
            )
            for fn in ('train_links', 'valid_links', 'test_links')
        ]
        if self.train_validation_split is None:
            logger.warning('Merging pre-defined train and validation part.')
            train, validation = torch.cat([train, validation], dim=1), None
        elif round(self.train_validation_split * (train.shape[1] + validation.shape[1])) != train.shape[1]:
            logger.warning('Using a custom train-validation split, since the dataset provides only a 70-20-10 split.')
            train_validation = torch.cat([train, validation], dim=1)
            # pylint: disable=unbalanced-tuple-unpacking
            train, validation = split_tensor(tensor=train_validation, ratios=self.train_validation_split, dim=1, seed=self.random_seed)
        return EntityAlignment(
            train=train,
            test=test,
            _validation=validation,
        )


def dataset_name_normalization(name: str) -> str:
    """Normalize a dataset name."""
    return name.lower().replace('_', '')


def available_datasets() -> Mapping[str, Collection[str]]:
    """List available datasets with their subsets."""
    return {
        dataset_name_normalization(cls.__name__): cls.subsets
        for cls in get_all_subclasses(base_class=OnlineKnowledgeGraphAlignmentDatasetLoader)
        if not cls.__name__.startswith('_')
    }


def get_dataset_by_name(
    dataset_name: str,
    subset_name: Optional[str] = None,
    train_test_split: Optional[float] = None,
    cache_root: Optional[Union[pathlib.Path, str]] = None,
    compact: bool = True,
    train_validation_split: Optional[float] = 0.8,
    inverse_triples: bool = False,
    self_loops: bool = False,
    random_seed: int = 42,
    force_download: bool = False,
) -> KnowledgeGraphAlignmentDataset:
    """Load a dataset specified by name and subset name.

    :param dataset_name:
        The case-insensitive dataset name. One of ("DBP15k", )
    :param subset_name:
        An optional subset name
    :param train_test_split: 0 < x < 1
        A specification of the train-test split to use.
    :param cache_root:
        An optional cache directory for extracted downloads. If None is given, use /tmp/{dataset_name}
    :param compact:
        Whether to apply compaction, i.e. ensure consecutive relation and entity IDs.
    :param train_validation_split: 0 < x < 1
        An optional train-validation split ratio.
    :param inverse_triples:
        Whether to generate inverse triples (o, p_inv, s) for every triple (s, p, o).
    :param self_loops:
        Whether to generate self-loops (e, self_loop, e) for each entity e.
    :param random_seed:
        The seed to use for random splitting.
    :param force_download:
        Force downloading the files even if they already exist.

    :return:
        A dataset, a collection of two KG, and an entity alignment.
    """
    dataset_loader = get_dataset_loader_by_name(
        dataset_name=dataset_name,
        subset_name=subset_name,
        train_test_split=train_test_split,
        cache_root=cache_root,
        compact=compact,
        train_validation_split=train_validation_split,
        inverse_triples=inverse_triples,
        self_loops=self_loops,
        random_seed=random_seed,
    )

    # load dataset
    dataset = dataset_loader.load(force_download=force_download)
    logger.info('Created dataset: %s', dataset)

    return dataset


def get_dataset_loader_by_name(
    dataset_name: str,
    subset_name: Optional[str] = None,
    train_test_split: Optional[float] = None,
    cache_root: Optional[Union[pathlib.Path, str]] = None,
    compact: bool = True,
    train_validation_split: Optional[float] = 0.8,
    inverse_triples: bool = False,
    self_loops: bool = False,
    random_seed: int = 42,
):
    """Create a dataset loader for a dataset specified by name and subset name.

    :param dataset_name:
        The case-insensitive dataset name. One of ("DBP15k", )
    :param subset_name:
        An optional subset name
    :param train_test_split: 0 < x < 1
        A specification of the train-test split to use.
    :param cache_root:
        An optional cache directory for extracted downloads. If None is given, use /tmp/{dataset_name}
    :param compact:
        Whether to apply compaction, i.e. ensure consecutive relation and entity IDs.
    :param train_validation_split: 0 < x < 1
        An optional train-validation split ratio.
    :param inverse_triples:
        Whether to generate inverse triples (o, p_inv, s) for every triple (s, p, o).
    :param self_loops:
        Whether to generate self-loops (e, self_loop, e) for each entity e.
    :param random_seed:
        The seed to use for random splitting.

    :return:
        A dataset loader.
    """
    # Normalize train-test-split
    if train_test_split is None:
        train_test_split = 0.3
    if isinstance(train_test_split, str):
        train_test_split = int(train_test_split) / 100.
    assert isinstance(train_test_split, float)

    # Resolve data set loader class
    dataset_loader_cls = get_subclass_by_name(
        base_class=OnlineKnowledgeGraphAlignmentDatasetLoader,
        name=dataset_name,
        normalizer=dataset_name_normalization,
        exclude=_WK3l,
    )

    # Instantiate dataset loader
    return dataset_loader_cls(
        subset=subset_name,
        train_test_split=train_test_split,
        cache_root=cache_root,
        compact=compact,
        train_validation_split=train_validation_split,
        with_inverse_triples=inverse_triples,
        with_self_loops=self_loops,
        random_seed=random_seed,
    )
