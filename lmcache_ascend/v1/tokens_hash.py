from typing import List, Union, Optional, Any
from lmcache.logging import init_logger
import torch

def _hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[List[Any]] = None,
    ) -> int:
        if isinstance(tokens, torch.Tensor):
            tokens_tuple = tuple(tokens.cpu().tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        # Ignore extra keys for now
        # Extra keys are for multi-modal inputs and
        # request specific metadata (e.g., LoRA ID).
        # NOTE: Pre python3.12 and for operating system that has ASLR turned on,
        #       hashing none will give inconsistent values across workers
        #       https://github.com/python/cpython/issues/99540
        if extra_keys is None:
            return self.hash_func((prefix_hash, tokens_tuple))
        return self.hash_func((prefix_hash, tokens_tuple, extra_keys))