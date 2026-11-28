<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache-Ascend/main/docs/logos/lmcache-ascend-logo.png" width="720" alt="lmcache-ascend logo">
  </p>
  <h3 align="center">
  LMCache-Ascend Plugin
  </h3>

  [![Code Quality](https://github.com/LMCache/LMCache-Ascend/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/LMCache/LMCache-Ascend/actions/workflows/code-quality.yml)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache-Ascend)
  
  <br />

  <p align="center">
  | <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a>
  | <a href="https://blog.lmcache.ai/"><b> LMCache Blog</b></a> 
  | <a href="https://docs.lmcache.ai/"><b>Documentation</b></a> 
  | <a href="https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q"><b> Slack</b></a>
  | <a href="https://deepwiki.com/LMCache/LMCache-Ascend"><b>LMCache-Ascend Wiki</b></a>
  </p>
</div>

--------------------------------------------------------------------------------

## Overview

LMCache-Ascend is a community maintained plugin for running LMCache on the Ascend NPU.

> **Maintenance notice**: The stable in-process version (`LMCacheAscendConnectorV1Dynamic`) is maintained on the [`dsv4_support_045`](https://github.com/LMCache/LMCache-Ascend/tree/dsv4_support_045) branch and is **tightly coupled with LMCache 0.4.5** — pair it with the LMCache 0.4.5 release line, not newer versions. The `main` branch moves to MP mode (`LMCacheMPConnector`) and will no longer maintain the in-process mode.

## Getting Started

The minimal deployment is three steps: install LMCache and LMCache-Ascend from source in the same environment (`main` tracks the LMCache `dev` branch), start the `lmcache server`, then start vLLM with `LMCacheMPConnector`. Prerequisites, version pairing, commands, and verification are documented upstream:

- [Ascend NPU](https://docs.lmcache.ai/platform/devices/npu/index.html) — prerequisites, source installation, running vLLM-Ascend with LMCache, supported KV cache formats.
- [LMCache-driven mode](https://docs.lmcache.ai/platform/devices/npu/lmcache_driven.html) — full deployment guide (environment preparation, container, installation, server, vLLM, verification).

## Documentation

MP mode documentation is maintained in the upstream LMCache docs:

- [MP mode on Ascend NPU](https://docs.lmcache.ai/platform/devices/npu/mp_mode.html) — transfer-mode matrix, supported models, and limitations.
- [LMCache-driven mode](https://docs.lmcache.ai/platform/devices/npu/lmcache_driven.html) — full deployment guide (environment preparation, container, installation, server, vLLM, verification).
- [Engine-driven mode](https://docs.lmcache.ai/platform/devices/npu/engine_driven.html) — deployment guide (Qwen3 example).
- [Hybrid Mamba models](https://docs.lmcache.ai/platform/devices/npu/hybrid_mamba.html) — Qwen3.5 / Qwen3.6 / Qwen3.8 flags and Ascend differences.
