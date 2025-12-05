# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Iterable, List, Optional, Tuple, Union

# Third Party
import torch
import numpy as np

# First Party
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate

def TokenDatabase__hash_tokens(
        self, tokens: Union[torch.Tensor, List[int]], prefix_hash: Optional[int] = None
    ) -> int:
        if isinstance(tokens, np.ndarray):
            tokens_tuple = tuple(tokens.tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        if prefix_hash is not None:
            return self.hash_func((prefix_hash, tokens_tuple))
        return self.hash_func(tokens_tuple)

@_lmcache_nvtx_annotate
def ChunkedTokenDatabase_process_tokens(
    self,
    tokens: Optional[Union[torch.Tensor, List[int]]] = None,
    hashes: Optional[List[int]] = None,
    offsets: Optional[List[int]] = None,
    mask: Optional[torch.Tensor] = None,
    make_key: bool = True,
) -> Iterable[Tuple[int, int, Union[CacheEngineKey, int]]]:
    """Process the tokens/hashes and return the corresponding cache engine keys.

    :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

    :param Optional[List[int]] hashes: The hashes to process. If provided,
        it will be used instead of tokens to generate cache engine keys.

    :param Optional[List[int]] offsets: The number of tokens in each chunk.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched,
        and the Falses will ALWAYS be at the PREFIX of the tensor.

    :param bool make_key: Whether to make the cache engine key or not.
        If False, the hash value will be returned instead.

    :returns: A iterable of tuples with three elements. The first element
        is the start index of the tokens for the key. The second element
        is the end index of the tokens for the key. The third element is
        the cache engine key (or hash) for the tokens.

    :raises: ValueError if the number of Falses in the mask is not a
        multiple of the chunk size.
    """
    if mask is not None:
        num_falses = mask.size - int(np.sum(mask))
    else:
        num_falses = 0

    if num_falses % self.chunk_size != 0:
        raise ValueError(
            "The number of Falses in the mask is not a multiple of the chunk size."
        )

    if tokens is not None:
        total_len = len(tokens)
        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                if make_key:
                    yield start_idx, end_idx, self._make_key_by_hash(hash_val)
                else:
                    yield start_idx, end_idx, hash_val
    elif hashes is not None:
        assert offsets is not None, (
            "If hashes are provided, offsets must also be provided."
        )
        start_idx = 0
        for hash_val, offset in zip(hashes, offsets, strict=False):
            end_idx = start_idx + offset
            if make_key:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)
            else:
                yield start_idx, end_idx, hash_val
            start_idx = end_idx
    else:
        raise ValueError("Either tokens or hashes must be provided.")
