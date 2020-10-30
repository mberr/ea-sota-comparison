"""Utilities for retrieving and checking data files."""
import hashlib
import logging
import pathlib
from typing import Optional

import humanize
import requests
import tqdm

logger = logging.getLogger(name=__name__)


def resolve_google_drive_file_url(
    id_: str,
    session: requests.Session,
) -> requests.Response:
    """
    Resolve the download path for a Google Drive file.

    This method clicks through download confirmation pages.

    :param id_:
        The file ID.
    :param session:
        The session.

    :return:
        The response.
    """
    # cf. https://stackoverflow.com/a/39225272
    GOOGLE_DRIVE_BASE_URL = "https://docs.google.com/uc?export=download"

    # request file
    response = session.get(GOOGLE_DRIVE_BASE_URL, params={'id': id_}, stream=True)

    # Download warning page
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': id_, 'confirm': value}
            return session.get(GOOGLE_DRIVE_BASE_URL, params=params, stream=True)

    return response


def save_response_content(
    response: requests.Response,
    destination: pathlib.Path,
    chunk_size: int = 2 ** 16,
    show_progress: bool = True,
) -> None:
    """
    Save content from a response to a file.

    :param response:
        The response object.

    :param destination:
        The destination where the content should be stored. Its parent directories will we created if they do not exist already.

    :param chunk_size:
        The chunk size in which to write to the file.

    :param show_progress:
        Whether to show a progress bar during download.

    """
    if response.status_code != requests.codes.ok:  # pylint: disable=no-member
        raise ValueError(f'Status Code of response is not OK ({requests.codes.ok}), but {response.status_code}')  # pylint: disable=no-member

    # Ensure that the parent directory exists.
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Try to infer download size
    try:
        total_size = int(response.headers.get('content-length', None))
    except TypeError:
        total_size = None
        logger.warning('Could not infer download size.')

    logger.info('Downloading from %s to %s', response.url, str(destination.absolute()))
    with destination.open(mode='wb') as f:
        iterator = response.iter_content(chunk_size=chunk_size)
        if show_progress:
            progress_bar = tqdm.tqdm(desc='Download', total=total_size, unit='iB', unit_scale=True, unit_divisor=2 ** 10)
        for chunk in iterator:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            # Write to file
            f.write(chunk)

            # Update counter
            chunk_size = len(chunk)

            # Update progress bar, if such exist
            if show_progress:
                progress_bar.update(n=chunk_size)
    if show_progress:
        progress_bar.close()

    # Check total file size against header information
    actual_size = destination.stat().st_size
    if total_size is not None:
        if actual_size != total_size:
            raise RuntimeError(f'Download of {response.url} failed. Expected size {total_size} vs. actual size {actual_size}')
    logger.info('Finished download of %s.', humanize.naturalsize(value=actual_size, binary=True))


def check_hashsums(
    destination: pathlib.Path,
    chunk_size: int = 64 * 2 ** 10,
    **hashes: str,
) -> bool:
    """
    Check a file for hash sums.

    :param destination:
        The file path.
    :param chunk_size:
        The chunk size for reading the file.
    :param hashes:
        The expected hashsums as (algorithm_name, hash_sum) pairs where hash_sum is the hexdigest

    :return:
        Whether all hash sums match.
    """
    if len(hashes) == 0:
        logger.warning('There are no hash sums to check for.')
        return True

    # instantiate algorithms
    hash_algorithms = {}
    for alg in hashes.keys():
        hash_algorithms[alg] = hashlib.new(alg)

    # calculate hash sums of file incrementally
    buffer = memoryview(bytearray(chunk_size))
    with destination.open('rb', buffering=0) as f:
        for this_chunk_size in iter(lambda: f.readinto(buffer), 0):
            for alg in hash_algorithms.values():
                alg.update(buffer[:this_chunk_size])

    # Compare digests
    integer_file = True
    for alg, digest in hashes.items():
        digest_ = hash_algorithms[alg].hexdigest()
        if digest_ != digest:
            logger.fatal('Hashsum does not match! expected %s=%s, but got %s', alg, digest, digest_)
            integer_file = False
        else:
            logger.info('Successfully checked with %s', alg)
    return integer_file


def resolve_cache_root(
    cache_root: Optional[pathlib.Path],
    *directories: str
) -> pathlib.Path:
    """
    Resolve cache root.

    :param cache_root:
        The cache root. If None, use ~/.kgm
    :param directories:
        Additional directories inside the cache root which are created if necessary.

    :return:
        An absolute path to an existing directory.
    """
    # default cache root
    if cache_root is None:
        cache_root = pathlib.Path('~', '.kgm')
    # Ensure it is an absolute path
    cache_root = cache_root.expanduser().absolute()
    # Create sub-directories
    for directory in directories:
        cache_root = cache_root / directory
    # Ensure that cache_root is an existing directory
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root
