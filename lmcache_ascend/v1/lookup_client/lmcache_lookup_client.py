# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import wraps

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


def normalize_token_ids(func):
    """Decorator that normalizes token_ids to a list for transport serialization.

    vLLM may pass custom types like ConstantList that are not directly
    serializable by the transport layer. This decorator ensures token_ids
    is always converted to a list before being passed to the underlying function.
    """

    @wraps(func)
    def wrapper(self, token_ids, lookup_id, request_configs=None):
        if not isinstance(token_ids, list):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            elif hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            else:
                token_ids = list(token_ids)
        return func(self, token_ids, lookup_id, request_configs)

    return wrapper
