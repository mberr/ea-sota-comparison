"""Type annotation aliases."""
import torch

#: A (n, 3) tensor of IDs.
Triples = torch.LongTensor

#: A (n,) tensor of IDs.
EntityIDs = torch.LongTensor

#: A (n,) tensor of IDs.
RelationIDs = torch.LongTensor

#: A (n,) tensor of IDs.
NodeIDs = torch.LongTensor

#: A (2, n) tensor of IDs.
IDAlignment = torch.LongTensor

#: A (2, n) tensor of IDs.
EdgeTensor = torch.LongTensor
