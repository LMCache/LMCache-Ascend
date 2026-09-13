# DeepSeek-V4 Cross-Node 16-NPU P/D Disaggregation Deployment Guide

| Role | Node | DP × TP | NPU Assignment | API Ports | DP RPC Port | KV Components |
| --- | --- | --- | --- | --- | --- | --- |
| Prefill (P) | P node | `8 × 2` | `0,1`, `2,3` … `14,15` | `7100-7107` | `12321` | Mooncake + LMCache |
| Decode (D) | D node | `16 × 1` | `0`, `1` … `15` | `7100-7115` | `12322` | Mooncake |
| Request proxy | P node | - | - | `8900` | - | 8 P backends and 16 D backends |

Both the P and D nodes use the following fixed `MooncakeHybridConnector`
topology:

```json
{
  "prefill": {"dp_size": 8, "tp_size": 2},
  "decode": {"dp_size": 16, "tp_size": 1}
}
```

Each DP rank runs as an independent `vllm serve` process, and the scripts pass
`--data-parallel-rank` explicitly. The `engine_id` identifies a physical node:
all eight ranks on the P node share `engine_id=0`, while all sixteen ranks on
the D node share `engine_id=1`. The Mooncake base port is `36000` on P and
`36100` on D. The connector derives the actual port from the DP and TP ranks,
so do not adjust the base port per rank in the launcher.

P uses `MultiConnector`: Mooncake handles online P-to-D KV transfers, while
LMCache provides pooling only on P. D uses only `MooncakeHybridConnector` and
does not load `decode/lmcache-decode-config.yaml`.

## 1. Copy the Files to Both Nodes

Keep a complete copy of this directory on both servers, then run only the
components required for each role:

- On P, run `prefill/start_p_sever.sh` and `prefill/start_proxy.sh`.
- On D, run `decode/start_d_sever.sh`.

## 2. Required Manual Changes Before Startup

### P Node

Update `prefill/run_dp_template.sh`:

- Change `HCCL_IF_IP=192.168.0.223` to the P-node IP used for HCCL and
  Mooncake communication.
- Change the three `*_SOCKET_IFNAME=enp23s0f3` values to the interface name
  associated with that IP.
- Change `/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp` to the actual model path
  on P.

Update `prefill/start_p_sever.sh`:

- Change `--dp-address 7.246.92.163` to a reachable DP address on P.

Update `prefill/start_proxy.sh`:

- Change `--host 7.246.92.163` to the proxy listen address. Use `0.0.0.0` to
  listen on all interfaces.
- Change all eight `--prefiller-hosts` entries to the P-node address.
- Change all sixteen `--decoder-hosts` entries to the D-node address.

### D Node

Update `decode/run_dp_template.sh`:

- Change `local_ip=192.168.0.50` to the D-node IP used for HCCL and Mooncake
  communication.
- Change `nic_name="enp23s0f3"` to the interface name associated with that IP.
- Change `/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp` to the actual model path
  on D.

Update `decode/start_d_sever.sh`:

- Change `--dp-address 7.246.92.165` to a reachable DP address on D.

Inspect the available addresses and interfaces with:

```bash
ip -br addr
ip route
```

The required ports must be reachable between the two nodes. At minimum,
check the following ranges:

- P API: `7100-7107`
- D API: `7100-7115`
- P/D DP RPC: `12321` and `12322`
- Mooncake base and rank-derived ports: allow `36000-36015` and `36100-36115`
- Proxy: `8900`
- P-side LMCache internal API: the port range starting at `3999`

## 3. Preflight Checks

Verify that each server has 16 available NPUs:

```bash
npu-smi info
```

LMCache is not configured on D.

Run a dry run on each node. These commands print the generated launch commands
without loading the model:

```bash
# P: prints 8 commands, each bound to 2 NPUs, using ports 7100-7107
bash dsv4_mooncake_lmcache_1p1d_script/prefill/start_p_sever.sh --dry-run

# D: prints 16 commands, each bound to 1 NPU, using ports 7100-7115
bash dsv4_mooncake_lmcache_1p1d_script/decode/start_d_sever.sh --dry-run
```

Run the local script tests with:

```bash
python3 -m unittest \
  dsv4_mooncake_lmcache_1p1d_script/tests/test_cross_node_16card_scripts.py -v
```

## 4. Startup Sequence

Start Decode on D:

```bash
bash dsv4_mooncake_lmcache_1p1d_script/decode/start_d_sever.sh
```

Start Prefill on P:

```bash
bash dsv4_mooncake_lmcache_1p1d_script/prefill/start_p_sever.sh
```

After the P APIs and their corresponding D APIs are ready, start the proxy on
P:

```bash
bash dsv4_mooncake_lmcache_1p1d_script/prefill/start_proxy.sh
```

## 5. Health Checks and Test Request

Check all backends from P. Replace the example node names with actual IP
addresses:

```bash
for port in {7100..7107}; do curl -fsS "http://P_NODE_IP:${port}/health"; done
for port in {7100..7115}; do curl -fsS "http://D_NODE_IP:${port}/health"; done
curl -fsS http://127.0.0.1:8900/healthcheck
```

Send application requests through the proxy:

```bash
curl http://127.0.0.1:8900/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4",
    "messages": [{"role": "user", "content": "Hello, please introduce yourself."}],
    "max_tokens": 64,
    "stream": false
  }'
```

P logs are written by DP rank to `model_lmcache_noprefix_dp0.log` through
`model_lmcache_noprefix_dp7.log`. D logs are written to
`model_decode_dp0.log` through `model_decode_dp15.log`.
