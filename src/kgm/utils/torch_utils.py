"""Utility methods using pytorch."""
import enum
import itertools
import logging
from abc import ABC
from collections import defaultdict
from operator import itemgetter
from typing import Any, Callable, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy
import pandas
import torch
from torch import nn, optim

from .common import get_subclass_by_name, integer_portion, reduce_kwargs_for_method
from .types import IDAlignment, NodeIDs

logger = logging.getLogger(name=__name__)

_ACTIVATION_NAME_TO_CLASS = {
    cls.__name__.lower(): cls for cls in (
        nn.ELU,
        nn.Hardshrink,
        nn.Hardtanh,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.PReLU,
        nn.ReLU,
        nn.RReLU,
        nn.SELU,
        nn.CELU,
        nn.Sigmoid,
        nn.Softplus,
        nn.Softshrink,
        nn.Softsign,
        nn.Tanh,
        nn.Tanhshrink,
    )
}


def get_activation_class_by_name(activation_cls_name: str) -> Type[nn.Module]:
    """Translate an activation name (a string) to the corresponding class."""
    key = activation_cls_name.lower()
    if key not in _ACTIVATION_NAME_TO_CLASS.keys():
        raise KeyError(f'Unknown activation class name: {key} not in {_ACTIVATION_NAME_TO_CLASS.keys()}.')
    activation_cls_name = _ACTIVATION_NAME_TO_CLASS[key]
    return activation_cls_name


def get_device(
    device: Union[None, str, torch.device],
) -> torch.device:
    """Resolve the device, either specified as name, or device."""
    if device is None:
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device=device)
    assert isinstance(device, torch.device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        logger.warning('Requested device %s, but CUDA is unavailable. Falling back to cpu.', device)
        device = torch.device('cpu')
    return device


def split_tensor(
    tensor: torch.Tensor,
    ratios: Union[float, Sequence[float]],
    shuffle: Optional[bool] = True,
    dim: Optional[int] = 0,
    seed: Optional[int] = 42,
) -> Sequence[torch.Tensor]:
    """
    Split tensor into multiple partitions along a dimension.

    The splits are performed consecutive, where each individual split is according to the given ratios.

    :param tensor:
        The tensor to split.
    :param ratios:
        A sequence of floats between [0, 1] specifying the ratio of the first partition of each split.
    :param shuffle:
        Whether to randomize order of data.
    :param dim:
        The dimension to split along.
    :param seed:
        The random seed to use for shuffling.

    :return:
        A sequence of disjoint subsets of the input tensor.
    """
    if isinstance(ratios, float):
        ratios = [ratios]

    num_elements = tensor.shape[dim]

    # shuffle
    if shuffle:
        # random seeding
        if seed is not None:
            generator = torch.manual_seed(seed=seed)
        else:
            generator = torch.random.default_generator
        indices = torch.randperm(n=num_elements, generator=generator, device=tensor.device)
    else:
        indices = torch.arange(0, num_elements, device=tensor.device)

    output = []
    remainder = indices
    for ratio in ratios:
        size_first = integer_portion(number=remainder.shape[0], ratio=ratio)
        this, remainder = remainder[:size_first], remainder[size_first:]
        output.append(tensor.index_select(dim=dim, index=this))
    output.append(tensor.index_select(dim=dim, index=remainder))
    return output


def _guess_num_nodes(
    num_nodes: Optional[int],
    source: Optional[NodeIDs] = None,
    target: Optional[NodeIDs] = None,
) -> int:
    """Try to guess the number of nodes."""
    if num_nodes is not None:
        return num_nodes
    if source is None and target is None:
        raise ValueError('If no num_nodes are given, either source, or target must be given!')
    return max(x.max().item() for x in (source, target) if x is not None)


def simple_sparse_softmax(
    edge_weights: torch.FloatTensor,
    index: NodeIDs,
    num_nodes: Optional[int] = None,
    dim: int = 0,
    eps: float = 1.0e-06,
    temperature: Optional[Union[float, torch.FloatTensor]] = None,
    method: str = 'global_max',
) -> torch.FloatTensor:
    """Compute sparse softmax.

    :param edge_weights: shape: (..., num_edges, ...)
    :param index: shape: (num_edges,)
    :param num_nodes:
        The total number of nodes. If not given, it will be estimated as max(source.max(), target.max()) + 1.
    :param dim:
        The dimension along which to perform the softmax.
    :param eps:
        Clamp normalization values for numerical stability.
    :param temperature:
        Optional softmax temperature to apply. Higher temperature leads to more uniform distribution.

    :return: shape: (num_edges, d)
    """
    # Guess number of nodes from source/target, if not provided explicitly
    num_nodes = _guess_num_nodes(num_nodes=num_nodes, source=index, target=None)

    # apply softmax temperature
    if temperature is not None:
        edge_weights = edge_weights / temperature

    # modification for numerical stability; note that this subtracts the **global** maximum value. If the values heavily differ between the normalization groups, this distort the result.
    if method == 'global_max':
        adjustment = edge_weights.max()
    elif method == 'logsumexp':
        adjustment = edge_weights.new_zeros(num_nodes).index_add(dim=dim, index=index, source=edge_weights.exp()).log().index_select(dim=dim, index=index)
    elif method == 'none':
        adjustment = 0
    elif method == 'cpu_max':
        res = pandas.DataFrame(data=dict(index=index.detach().cpu().numpy(), weight=edge_weights.detach().cpu().numpy())).groupby(by='index').agg({'weight': 'max'})
        df_index = torch.as_tensor(data=res.index.values, dtype=torch.long, device=index.device)
        df_max = torch.as_tensor(data=res.values[:, 0], device=edge_weights.device, dtype=edge_weights.dtype)
        adjustment = edge_weights.new_zeros(num_nodes).scatter_(dim=0, index=df_index, src=df_max).index_select(dim=0, index=index)
    else:
        raise NotImplementedError
    edge_weights = edge_weights - adjustment
    edge_weights = torch.exp(edge_weights)
    edge_weights = edge_weights / torch.zeros(num_nodes, device=index.device).index_add_(dim=dim, index=index, source=edge_weights).clamp_min(eps).index_select(dim=dim, index=index)
    return edge_weights


def get_optimizer_class_by_name(name: str) -> Type[optim.Optimizer]:
    """Return an optimizer class given its name."""
    return get_subclass_by_name(base_class=optim.Optimizer, name=name, normalizer=str.lower)


def _is_oom_error(error: RuntimeError) -> bool:
    """Check whether a runtime error was caused by insufficient memory."""
    message = error.args[0]

    # CUDA out of memory
    if 'CUDA out of memory.' in message:
        return True

    # CPU out of memory
    if "[enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory:" in message:
        return True
    return False


R = TypeVar('R')


def maximize_memory_utilization(
    func: Callable[..., R],
    parameter_name: str,
    parameter_max_value: int,
    *args,
    **kwargs
) -> Tuple[R, int]:  # noqa: D401
    """
    Iteratively reduce parameter value until no RuntimeError is generated by CUDA.

    :param func:
        The callable.
    :param parameter_name:
        The name of the parameter to maximise.
    :param parameter_max_value:
        The maximum value to start with.
    :param args:
        Additional positional arguments for func. Does _not_ include parameter_name!
    :param kwargs:
        Additional keyword-based arguments for func. Does _not_ include parameter_name!

    :return:
        The result, as well as the maximum value which led to successful execution.
    """
    result = None
    direct_success = True
    if not all((not torch.is_tensor(obj) or obj.device.type == 'cuda') for obj in itertools.chain(args, kwargs.values())):
        logger.warning('Using maximize_memory_utilization on non-CUDA tensors. This may lead to undocumented crashes due to CPU OOM killer.')
    while parameter_max_value > 0:
        p_kwargs = {parameter_name: parameter_max_value}
        try:
            result = func(*args, **p_kwargs, **kwargs)
            if not direct_success:
                logger.info('Execution succeeded with %s=%d', parameter_name, parameter_max_value)
            break
        except RuntimeError as runtime_error:
            # Failed at least once
            direct_success = False

            # clear cache
            torch.cuda.empty_cache()

            # check whether the error is an out-of-memory error
            if not _is_oom_error(error=runtime_error):
                raise runtime_error

            logger.info('Execution failed with %s=%d', parameter_name, parameter_max_value)
            parameter_max_value //= 2
    if parameter_max_value == 0:
        raise MemoryError(f'Execution did not even succeed with {parameter_name}=1.')
    return result, parameter_max_value


def construct_optimizer_from_config(model: nn.Module, optimizer_config: MutableMapping[str, Any]) -> optim.Optimizer:
    """
    Create a pytorch optimizer for a model, given a config.

    :param model:
        The model.
    :param optimizer_config:
        The config: dict(
            cls=<OPTIMIZER_CLASS_NAME>,
            **kwargs,
        )
        where kwargs are passed down to the optimizer's constructor, and stripped before from unused arguments.

    :return:
        The optimizer instance.
    """
    optim_name = optimizer_config.pop('cls')
    opt_cls = get_optimizer_class_by_name(name=optim_name)

    # reduce to parameter needed
    optimizer_config = reduce_kwargs_for_method(opt_cls.__init__, kwargs=optimizer_config, raise_on_missing=False)

    # instantiate optimizer
    optimizer = opt_cls(params=(p for p in model.parameters() if p.requires_grad), **optimizer_config)

    return optimizer


class AlignmentTypeEnum(str, enum.Enum):
    """Enum for possible alignment types."""

    one_to_one = '1-1'
    one_to_many = '1-n'
    many_to_one = 'n-1'
    many_to_many = 'n-m'


def get_alignment_type(
    alignment: IDAlignment,
) -> AlignmentTypeEnum:
    """
    Determine type of alignment.

    :param alignment: shape: (2, num_alignments)
        The alignment.

    :return:
        The type.
    """
    num_alignments = alignment.shape[1]
    left_uniq, right_uniq = [len(torch.unique(alignment[i])) == num_alignments for i in range(2)]
    if left_uniq and right_uniq:
        return AlignmentTypeEnum.one_to_one
    if left_uniq and not right_uniq:
        return AlignmentTypeEnum.one_to_many
    if not left_uniq and right_uniq:
        return AlignmentTypeEnum.many_to_one
    return AlignmentTypeEnum.many_to_many


# pylint: disable=abstract-method
class ExtendedModule(nn.Module):
    """Extends nn.Module by a few utility methods."""

    @property
    def device(self) -> torch.device:
        """Return the model's device."""
        devices = {
            tensor.data.device
            for tensor in itertools.chain(self.parameters(), self.buffers())
        }
        if len(devices) == 0:
            raise ValueError('Could not infer device, since there are neither parameters nor buffers.')
        elif len(devices) > 1:
            device_info = dict(
                parameters=dict(self.named_parameters()),
                buffers=dict(self.named_buffers()),
            )
            raise ValueError(f'Ambiguous device! Found: {devices}\n\n{device_info}')
        return next(iter(devices))

    def reset_parameters(self):
        """Reset the model's parameters."""
        # Make sure that all modules with parameters do have a reset_parameters method.
        uninitialized_parameters = set(map(id, self.parameters()))
        parents = defaultdict(list)

        # Recursively visit all sub-modules
        task_list = []
        for name, module in self.named_modules():

            # skip self
            if module is self:
                continue

            # Track parents for blaming
            for p in module.parameters():
                parents[id(p)].append(module)

            # call reset_parameters if possible
            if hasattr(module, 'reset_parameters'):
                task_list.append((name.count('.'), module))

        # initialize from bottom to top
        # This ensures that specialized initializations will take priority over the default ones of its components.
        for module in map(itemgetter(1), sorted(task_list, reverse=True, key=itemgetter(0))):
            module.reset_parameters()
            uninitialized_parameters.difference_update(map(id, module.parameters()))

        # emit warning if there where parameters which were not initialised by reset_parameters.
        if len(uninitialized_parameters) > 0:
            logger.warning('reset_parameters() not found for all modules containing parameters. %d parameters where likely not initialised.', len(uninitialized_parameters))

            # Additional debug information
            for i, p_id in enumerate(uninitialized_parameters, start=1):
                logger.debug('[%3d] Parents to blame: %s', i, parents.get(p_id))


class SparseMatrix(ExtendedModule, ABC):
    """A matrix."""

    #: The shape (n_rows, n_cols)
    shape: Tuple[int, int]

    def __init__(self, shape: Tuple[int, int]):
        """
        Initialize matrix.

        :param shape:
            The shape, (n_rows, n_cols).
        """
        super().__init__()
        self.shape = shape

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        """
        Matrix-matrix multiplication.

        :param other: shape: (n_cols, d)
            The vector.

        :return: shape: (n_rows, d)
            out[i, :] = self[:, i] * other[i, :]
        """
        if other.shape[0] != self.shape[1]:
            raise ValueError(f'Shape mismatch: self.shape={self.shape}, other.shape={other.shape}. {self.shape[1]} != {other.shape[0]}.')
        return self._real_matmul(other=other)

    def _real_matmul(self, other: torch.Tensor) -> torch.Tensor:
        """Perform the matrix-matrix multiplication."""
        raise NotImplementedError

    def t(self) -> 'SparseMatrix':
        """Matrix transposition."""
        raise NotImplementedError

    def detach(self) -> 'SparseMatrix':
        """Detaches the values, i.e. breaks the gradient flow."""
        raise NotImplementedError

    def dense(self) -> torch.Tensor:
        """Return a dense version of the matrix."""
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return self @ x."""
        return self @ x


class SparseCOOMatrix(SparseMatrix):
    """A sparse matrix in COO format."""

    #: The indices of the non-zero elements.
    sparse_matrix: torch.sparse.Tensor

    def __init__(
        self,
        matrix: torch.sparse.Tensor
    ):
        """
        Initialize the matrix.

        :param matrix:
            The matrix.
        """
        super().__init__(shape=matrix.shape)
        assert len(matrix.shape) == 2
        self.register_buffer(name='sparse_matrix', tensor=matrix.coalesce())

    @staticmethod
    def from_indices_values_pair(
        indices: torch.LongTensor,
        values: Optional[torch.Tensor] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> 'SparseCOOMatrix':
        """
        Instantiate the matrix using a pair of indices and optional values.

        :param indices: shape: (2, nnz)
            The indices.
        :param values: shape: (nnz,)
            The values.
        :param size:
            The size. If None, infer from indices.

        :return:
            The matrix.
        """
        if size is None:
            size = tuple((indices.max(dim=1).values + 1).tolist())
        if isinstance(size, int):
            size = (size, size)
        for dim, (index_dim, size_dim) in enumerate(zip(indices, size)):
            max_id_on_dim = index_dim.max().item()
            if max_id_on_dim >= size_dim:
                raise ValueError(f'Index out of range for dim={dim}: {max_id_on_dim} vs. {size_dim}')
        if values is None:
            values = indices.new_ones(indices.shape[1], dtype=torch.float32)
        return SparseCOOMatrix(matrix=torch.sparse_coo_tensor(indices=indices, values=values, size=size))

    @staticmethod
    def from_edge_tensor(
        edge_tensor: torch.LongTensor,
        edge_weights: Optional[torch.Tensor] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> 'SparseCOOMatrix':
        """
        Construct a sparse adjacency matrix for a given edge_tensor.

        :param edge_tensor: shape: (2, num_edges)
            The edge tensor, elements: (source, target)
        :param edge_weights: shape: (num_edges,)
            Edge weights.
        :param size: >0
            The size, format num_nodes or (num_targets, num_sources).

        :return:
            The adjacency matrix.
        """
        return SparseCOOMatrix.from_indices_values_pair(
            indices=edge_tensor.flip(0),
            values=edge_weights,
            size=size,
        )

    @staticmethod
    def from_dense(
        dense: torch.Tensor,
    ) -> 'SparseCOOMatrix':
        """
        Construct a sparse matrix from a given dense version.

        :param dense: shape: (m, n)
            The dense matrix. Should have some/many zero elements.

        :return:
            The sparse matrix containing only the non-zero elements.
        """
        # convert to sparse matrix
        indices = dense.nonzero(as_tuple=True)
        values = dense[indices]
        return SparseCOOMatrix.from_indices_values_pair(
            indices=torch.stack(indices, dim=0),
            values=values,
            size=dense.shape,
        )

    @staticmethod
    def eye(n: int, device: Union[torch.device, str, None] = None) -> 'SparseCOOMatrix':
        """
        Construct a sparse identity matrix.

        :param n:
            The dimension.
        :param device:
            The device.

        :return:
            The identity matrix.
        """
        return SparseCOOMatrix.from_indices_values_pair(
            indices=torch.arange(n, device=device).unsqueeze(dim=0).repeat(2, 1),
            size=n,
        )

    @property
    def indices(self) -> torch.LongTensor:
        """Return the indices."""
        return self.sparse_matrix.indices()

    @property
    def values(self) -> torch.FloatTensor:
        """Return the values."""
        return self.sparse_matrix.values()

    def sum(self, dim: int) -> torch.Tensor:
        """
        Compute the sum along a dimension.

        :param dim:
            The dimension. From {0, 1}.

        :return: shape: (shape_at_dim,)
            The sum, a tensor of shape[dim].
        """
        return torch.sparse.sum(input=self.sparse_matrix, dim=dim).to_dense()

    def normalize(
        self,
        dim: int = 1,
        target_sum: Optional[float] = None,
    ) -> 'SparseCOOMatrix':
        """
        Normalize the matrix row-wise / column-wise.

        :param dim:
            The dimension.
        :param target_sum:
            An optional target value for the row/column sum. Defaults to 1.

        :return:
            The normalized matrix.
        """
        weights = self.sum(dim=dim).reciprocal()
        if target_sum is not None:
            weights = weights * target_sum
        weights = self.scatter(x=weights, dim=1 - dim) * self.values
        return self.with_weights(weights=weights)

    @property
    def source(self) -> torch.LongTensor:
        """Return the source indices for message passing."""
        return self.indices[1]

    @property
    def target(self) -> torch.LongTensor:
        """Return the target indices for message passing."""
        return self.indices[0]

    def with_weights(self, weights: torch.Tensor) -> 'SparseCOOMatrix':
        """Return a matrix of the same structure, with adjusted weights."""
        return SparseCOOMatrix.from_indices_values_pair(
            indices=self.indices,
            values=weights,
            size=self.shape,
        )

    def without_weights(self) -> 'SparseCOOMatrix':
        """Return the matrix without weights."""
        self.coalesce_()
        return SparseCOOMatrix(
            matrix=torch.sparse_coo_tensor(
                indices=self.sparse_matrix.indices(),
                values=torch.ones_like(self.sparse_matrix.values()),
                size=self.shape,
            ),
        )

    def scatter(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Scatter elements of x to the edges.

        :param x: shape: (self.shape[dim], d1, ..., dk)
            The values for each node.
        :param dim: The dimension, from {0, 1}.
            dim=0 -> from target
            dim=1 -> from source
        :return: shape: (nnz, d1, ..., dk)
            The values broadcasted to each edge.
        """
        if x.shape[0] != self.shape[dim]:
            raise ValueError(x.shape, self.shape[dim])
        return x.index_select(dim=0, index=self.indices[dim])

    def gather(self, m: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather elements of m from edges to nodes.

        :param m: shape: (num_edges, d1, ..., dk)
            The values for each edge.
        :param dim: The dimension, from {0, 1}.
            dim=0 -> to source
            dim=1 -> to target
        :return: shape: (num_nodes, d1, ..., dk)
            The values broadcasted to each node.
        """
        if m.shape[0] != self.indices.shape[1]:
            raise ValueError(m.shape, self.indices.shape[1])
        return m.new_zeros(self.shape[dim], *m.shape[1:]).index_add(dim=0, index=self.indices[dim], source=m)

    def t(self) -> 'SparseCOOMatrix':
        """Transposed matrix."""
        return SparseCOOMatrix(matrix=self.sparse_matrix.t())

    def _real_matmul(self, other: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # torch.sparse.mm requires float values
        if self.values.is_floating_point() and other.is_floating_point():
            return torch.sparse.mm(mat1=self.sparse_matrix, mat2=other)

        msg = self.scatter(x=other)
        if self.values is not None:
            msg = msg * self.values.view(msg.shape[0], 1)
        return self.gather(m=msg)

    def coalesce_(self) -> 'SparseCOOMatrix':
        """In-place index de-duplication."""
        self.sparse_matrix = self.sparse_matrix.coalesce()
        return self

    def coalesce(self) -> 'SparseCOOMatrix':
        """
        Collapses duplicate entries for (row, col) in indices.

        Since COO format permits duplicates (row, col), and some operations require unique indices, this operation
        collapses them, by adding the elements. This operation is quite costly.
        """
        return SparseCOOMatrix(matrix=self.sparse_matrix.coalesce())

    def __add__(self, other: 'SparseCOOMatrix') -> 'SparseCOOMatrix':  # noqa: D105
        if not isinstance(other, SparseCOOMatrix):
            raise NotImplementedError
        return SparseCOOMatrix(matrix=self.sparse_matrix + other.sparse_matrix)

    def detach(self) -> 'SparseCOOMatrix':  # noqa: D102
        return SparseCOOMatrix(matrix=self.sparse_matrix.detach())

    def dense(self) -> torch.Tensor:  # noqa: D102
        assert len(self.shape) == 2
        self.coalesce_()
        result = self.values.new_zeros(size=self.shape)
        result[self.indices[0], self.indices[1]] = self.values
        return result

    @property
    def edge_tensor(self) -> torch.LongTensor:
        """Return the edge_tensor view of the adjacency matrix."""
        return torch.stack([
            self.source,
            self.target,
        ], dim=0)

    @property
    def edge_weights(self) -> torch.FloatTensor:
        """Return the edge_weights view of the adjacency matrix."""
        return self.values

    @property
    def nnz(self) -> int:
        """Return the number of occupied indices."""
        return self.indices.shape[1]

    def extra_repr(self) -> str:
        """Return a string with some basic information."""
        return f'size={self.shape}, nnz={self.nnz}, sparsity={1. - (self.nnz / numpy.prod(self.shape)):.2%}'
