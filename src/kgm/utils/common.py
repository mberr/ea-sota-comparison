"""General utility methods."""
import hashlib
import inspect
import logging
import pickle
import random
import string
from collections import deque
from enum import Enum
from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Type, TypeVar, Union

T = TypeVar('T')

logger = logging.getLogger(name=__name__)


def enum_values(enum_cls: Type[Enum]) -> List:
    """List enum values."""
    return [v.value for v in enum_cls]


def value_to_enum(enum_cls: Type[Enum], value: T) -> Enum:
    """Lookup enum for a given value."""
    pos = [v for v in enum_cls if v.value == value]
    if len(pos) != 1:
        raise AssertionError(f'Could not resolve {value} for enum {enum_cls}. Available are {list(v for v in enum_cls)}.')
    return pos[0]


def identity(x: T) -> T:
    """Return the value itself."""
    return x


def get_all_subclasses(base_class: Type[T]) -> Set[Type[T]]:
    """Get a collection of all (recursive) subclasses of a given base class."""
    return set(base_class.__subclasses__()).union(s for c in base_class.__subclasses__() for s in get_all_subclasses(c))


def get_subclass_by_name(
    base_class: Type[T],
    name: str,
    normalizer: Optional[Callable[[str], str]] = None,
    exclude: Optional[Union[Collection[Type[T]], Type[T]]] = None,
) -> Type[T]:
    """Get a subclass of a base-class by name.

    :param base_class:
        The base class.
    :param name:
        The name.
    :param normalizer:
        An optional name normalizer, e.g. str.lower
    :param exclude:
        An optional collection of subclasses to exclude.

    :return:
        The subclass with matching name.
    :raises ValueError:
        If no such subclass can be determined.
    """
    if normalizer is None:
        normalizer = identity
    if exclude is None:
        exclude = set()
    if isinstance(exclude, type):
        exclude = {exclude}
    norm_name = normalizer(name)
    for subclass in get_all_subclasses(base_class=base_class).difference(exclude):
        if normalizer(subclass.__name__) == norm_name:
            return subclass
    subclass_dict = {normalizer(c.__name__): c for c in get_all_subclasses(base_class=base_class)}
    raise ValueError(f'{base_class} does not have a subclass named {norm_name}. Subclasses: {subclass_dict}.')


def argparse_bool(x):
    """Convert a command line arguments for a boolean value."""
    return str(x).lower() in {'true', '1', 'yes'}


def kwargs_or_empty(kwargs: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the dictionary, or an empty dictionary."""
    if kwargs is None:
        kwargs = {}
    return kwargs


def reduce_kwargs_for_method(
    method,
    kwargs: Optional[Mapping[str, Any]] = None,
    raise_on_missing: bool = True,
) -> Mapping[str, Any]:
    """Prepare keyword arguments for a method.

    Drops excess parameters with warning, and checks whether arguments are provided for all mandantory parameters.
    """
    # Ensure kwargs is a dictionary
    kwargs = kwargs_or_empty(kwargs=kwargs)

    # compare keys with argument names
    signature = inspect.signature(method)
    parameters = set(signature.parameters.keys())

    # Drop arguments which are unexpected
    to_drop = set(kwargs.keys()).difference(parameters)
    if len(to_drop) > 0:
        dropped = {k: kwargs[k] for k in to_drop}
        logger.warning('Dropping parameters: %s', dropped)
    kwargs = {k: v for k, v in kwargs.items() if k not in to_drop}

    # Check whether all necessary parameters are provided
    missing = set()
    for parameter in signature.parameters.values():
        if (parameter.default is parameter.empty) and parameter.name not in kwargs.keys() and parameter.name != 'self' and parameter.kind != parameter.VAR_POSITIONAL and parameter.kind != parameter.VAR_KEYWORD:
            missing.add(parameter.name)

    # check whether missing parameters are provided via kwargs
    missing = missing.difference(kwargs.get('kwargs', dict()).keys())

    if len(missing) > 0 and raise_on_missing:
        raise ValueError(f'Method {method.__name__} missing required parameters: {missing}')

    return kwargs


def to_dot(
    config: Dict[str, Any],
    prefix: Optional[str] = None,
    separator: str = '.',
    function_to_name: bool = True,
) -> Dict[str, Any]:
    """Convert nested dictionary to flat dictionary.

    :param config:
        The potentially nested dictionary.
    :param prefix:
        An optional prefix.
    :param separator:
        The separator used to flatten the dictionary.
    :param function_to_name:
        Whether to convert functions to a string representation.

    :return:
        A flat dictionary where nested keys are joined by a separator.
    """
    result = dict()
    for k, v in config.items():
        if prefix is not None:
            k = f'{prefix}{separator}{k}'
        if isinstance(v, dict):
            v = to_dot(config=v, prefix=k, separator=separator)
        elif hasattr(v, '__call__') and function_to_name:
            v = {k: v.__name__ if hasattr(v, '__name__') else str(v)}
        else:
            v = {k: v}
        result.update(v)
    return result


def from_dot(
    dictionary: Mapping[str, Any],
    separator: str = '.',
) -> Dict[str, Any]:
    """Convert flat dictionary to a nested dictionary.

    :param dictionary:
        The flat dictionary.
    :param separator:
        The separator used to flatten the dictionary.

    :return:
        A nested dictionary where flat keys are split by a separator.
    """
    result = {}
    for k, v in dictionary.items():
        key_sequence = k.split(sep=separator)
        sub_result = result
        for key in key_sequence[:-1]:
            if key not in sub_result:
                sub_result[key] = dict()
            sub_result = sub_result[key]
        sub_result[key_sequence[-1]] = v
    return result


class NonFiniteLossError(RuntimeError):
    """A non-finite loss value."""


K = TypeVar('K')
V = TypeVar('V')


def invert_mapping(
    mapping: Mapping[K, V]
) -> Mapping[V, K]:
    """
    Invert a mapping. Has to be a bijection, i.e. one-to-one mapping.

    :param mapping:
        The mapping key -> value

    :return:
        The mapping value -> key
    """
    if len(set(mapping.values())) < len(mapping):
        raise ValueError('Mapping is not a bijection, since there are duplicate values!')
    return {
        v: k
        for k, v in mapping.items()
    }


def random_split_range(max_val: int, num: int) -> List[int]:
    """Randomly split a range into num parts."""""
    if max_val <= 0:
        raise ValueError(f'max_val must be strictly positive, but max_val{max_val}')
    if num <= 0:
        raise ValueError(f'num must be strictly positive, but num={num}')
    if num > max_val:
        raise ValueError(f'Cannot split {max_val} into {num} positive parts.')
    breaks = [0] + sorted(random.sample(range(1, max_val), k=num - 1)) + [max_val]
    return [(stop - start) for start, stop in zip(breaks, breaks[1:])]


def get_value_from_nested_mapping(
    dictionary: Mapping[str, Any],
    keys: Sequence[str],
    default: Optional = 'raise',
) -> Any:
    """
    Get a value from a nested dictionary addressed by a sequence of keys.

    :param dictionary:
        The (nested) dictionary.
    :param keys:
        A sequence of keys.

    :return:
        The value.
    """
    for key in keys:
        if key not in dictionary:
            if default == 'raise':
                raise KeyError
            else:
                return default
        dictionary = dictionary[key]
    return dictionary


def integer_portion(
    number: int,
    ratio: float = 1.,
    multiple_of: int = 1,
) -> int:
    """
    Multiply a number by a ratio and round the result.

    Constraints:
    1. The output is at least multiple_of
    2. Besides, the output is the closest multiple.

    :param number:
        The original number.
    :param ratio:
        The relative factor.
    :param multiple_of:
        Use the closest multiple of this number.
    :return:
    """
    for name, value in dict(
        number=number,
        ratio=ratio,
        multiple_of=multiple_of,
    ).items():
        if value <= 0:
            raise ValueError(f'{name} needs to be strictly positive, but is {value}.')
    return max(int(round(number * ratio / multiple_of)), 1) * multiple_of


def last(iterable: Iterable[T]) -> T:
    """Return the last item of an iterable."""
    return deque(iterable, maxlen=1).pop()


def random_sentence_list(
    num_sentences: int = 1,
    word_sep: str = ' ',
    min_num_words: int = 1,
    max_num_words: int = 1,
    max_word_length: int = 10,
    min_word_length: int = 2,
    word_prefix: str = '',
    sentence_prefix: str = '',
    alphabet: Sequence[str] = string.ascii_letters,
) -> Sequence[str]:
    """Generate a list of random words."""
    return [
        sentence_prefix + word_sep.join(
            word_prefix + ''.join(
                random.sample(
                    alphabet,
                    random.randrange(min_word_length, max_word_length + 1)
                )
            )
            for _ in range(random.randrange(min_num_words, max_num_words) if max_num_words > min_num_words else min_num_words)
        )
        for _ in range(num_sentences)
    ]


def multi_hash(*keys: Any, hash_function: str = "sha512") -> str:
    """Return a hash sum for a sequence of objects."""
    return hashlib.new(name=hash_function, data=pickle.dumps(tuple(keys))).hexdigest()
