# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Iterable, List, Optional, Tuple, Union

# Third Party
from lmcache.utils import CacheEngineKey
import torch

# Type alias for process_tokens return value
# (start_index, end_index, cache_engine_key｜hash)
ProcessTokensResult = Tuple[int, int, Union[CacheEngineKey, int]]


def process_tokens(
    self,
    tokens: Optional[Union[torch.Tensor, List[int]]] = None,
    hashes: Optional[List[int]] = None,
    offsets: Optional[List[int]] = None,
    mask: Optional[torch.Tensor] = None,
    make_key: bool = True,
    request_configs: Optional[dict] = None,
) -> Iterable[ProcessTokensResult]:
    """Process the tokens and return the corresponding cache engine keys.

    :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

    :param Optional[List[int]] hashes: The hashes to process. If provided,
        it will be used instead of tokens to generate cache engine keys.

    :param Optional[List[int]] offsets: The number of tokens in each chunk.

    :param Optional[torch.Tensor] mask: The mask for the tokens. Should
        have the same length as tokens. And the mask should ALWAYS be like
        FFFFFTTTTTTT, where True means the tokens needs to be matched,
        and the Falses will ALWAYS be at the PREFIX of the tensor.

    :param bool make_key: Whether to make the cache engine key or not.
        If False, the hash value will be returned instead.

    :param Optional[dict] request_configs: The configs of the request.

    :returns: A iterable of tuples with three elements. The first element
        is the start index of the tokens for the key. The second element
        is the end index of the tokens for the key. The third element is
        the cache engine key for the tokens.

    """

    if tokens is not None:
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
        else:
            tokens = tokens.to(device="cpu", dtype=torch.long)

        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses >= len(tokens):
            # All tokens are masked out, so nothing to process.
            return

        token_chunks = self._fast_split_by_subtensor(tokens)
        start_idx = 0
        for idx, token_chunk in enumerate(token_chunks):
            token_chunk_len = len(token_chunk)
            end_idx = start_idx + token_chunk_len
            if idx > 0:
                start_idx += self.sep_len
                end_idx += self.sep_len
            if start_idx >= num_falses:
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(
                            self._hash_tokens(token_chunk), request_configs
                        ),
                    )
                else:
                    yield start_idx, end_idx, self._hash_tokens(token_chunk)
            start_idx = end_idx
    elif hashes is not None:
        assert offsets is not None, (
            "If hashes are provided, offsets must also be provided."
        )
        start_idx = 0
        for hash_val, offset in zip(hashes, offsets, strict=False):
            end_idx = start_idx + offset
            if make_key:
                yield (
                    start_idx,
                    end_idx,
                    self._make_key_by_hash(hash_val, request_configs),
                )
            else:
                yield start_idx, end_idx, hash_val
            start_idx = end_idx
    else:
        raise ValueError("Either tokens or hashes must be provided.")
