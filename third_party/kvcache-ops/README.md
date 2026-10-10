# KVCache Ops

KVCache Ops is a simple library containing LLM KVCache related operators for Ascend NPU.

We currently have a few operators that support KVCache offload/reload (D2H & H2D).

## Compilation

To compile the kernels, we leverage the ascendc_library function from the ascend_toolkit.

This means that we can leverage compilation macros like `__CCE_AICORE__` to switch between implementation at compile time for the device side.

However, we have embedded the host side execution part also in the kernels files, and therefore we introduce the host side macro `ASCEND_AICORE_ARCH` for host side compilation.

To use the kernels and compile in your application, you can try the following:

```
# CMakeLists.txt
# assume kvcache-ops is a submodule in your main application
# ...
add_subdirectory(third_party/kvcache-ops)
# ...
```

## Future work

Separate the arguments into a op host tiling data structure and modify the build step for the kernels.