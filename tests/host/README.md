# GDN host regressions

From the repository root, with Python 3.10 or later:

```bash
python -m unittest discover -s tests/host -v
```

No torch, vLLM, LMCache installation or NPU is needed. The tests load actual
production source modules, or compile selected production class methods from
their AST with explicit dependency doubles. They do not duplicate the tested
implementations. This isolates ownership and protocol regressions from device
setup and the root pytest dependency bootstrap.

`test_gdn_mm_bypass.py` covers sticky image bypass, new/resumed/cached request
metadata, finish-only steps and repeated completion polling on both TP ranks,
cleanup, and the unchanged text lookup path.

On the async branch, `test_gdn_async_host.py` additionally covers configuration
precedence, the thread-affine lookup owner, pending versus miss, cancellation,
manager/factory delegation, private snapshot ownership and budget, shared
Attention/state store accounting, transfer/cleanup ordering, receive ACKs and
failure quarantine. Device operations in these tests are doubles: passing does
not prove DMA, NPU numerical correctness or full plugin bootstrap compatibility.

The ordinary `tests/v1/test_state_integration.py` suite also contains the
multimodal bypass expectation. Run it separately in the repository's normal
dependency environment; the standalone host run is not a claim that the entire
integration suite passed.
