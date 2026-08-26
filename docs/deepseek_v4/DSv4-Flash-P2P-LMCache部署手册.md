# DeepSeek-V4-Flash + LMCache P2P 双机部署手册（dsv4_support_045 版）

> **文档定位**：基于 `LMCache/LMCache-Ascend` 的 `dsv4_support_045` 分支，在两台 Ascend 服务器上部署 DeepSeek-V4-Flash 的 P2P 双机 KV 共享服务。全程使用绝对路径，不建软链。
>
> **硬件**：2 台 Ubuntu 服务器，每台 8 × Ascend 910 NPU，内网互联
> **模型**：DeepSeek-V4-Flash-w8a8-mtp（绝对路径 `/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp`）
> **镜像**：`quay.nju.edu.cn/ascend/vllm-ascend:v0.22.1rc1-a3`
> **仓库**：`LMCache/LMCache-Ascend` @ `dsv4_support_045` 分支（含 DSv4 官方支持）

## 目录约定（本手册统一使用，无软链）

| 路径 | 用途 |
|---|---|
| `/mnt/sdb/zj/p2p/` | **部署根目录**（仓库、脚本、yaml、日志全在这） |
| `/mnt/sdb/zj/p2p/LMCache-Ascend/` | 本仓库（dsv4_support_045 分支） |
| `/mnt/sdb/zj/p2p/LMCache/` | LMCache 上游源码（v0.4.5） |
| `/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp/` | 模型权重 |

---

## 第 0 步：查询两机内网 IP（先做，后面所有配置依赖它）

> ⚠️ 部署前必须先拿到两台机的**内网 IP** 和**网卡名**，然后替换本手册所有 `<A机内网IP>` / `<B机内网IP>` / `<网卡名>` 占位符。

**在 A 机和 B 机宿主机上各执行**：

```bash
# 方法1：看所有网卡和 IP（找 192.168.x.x 或 10.x.x.x 的内网地址）
ip addr | grep -E "inet |^[0-9]+: "

# 方法2：看默认路由走哪张网卡（这张就是跨机通信网卡）
ip route | grep default

# 方法3：验证两机互通（在 A 机 ping B 机的内网 IP）
ping -c 3 <B机内网IP>
```

**示例输出解读**：

```
2: enp23s0f3: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
    inet 192.168.0.223/24 brd 192.168.0.255 scope global ... enp23s0f3
    ↑ 网卡名=enp23s0f3   ↑ 本机内网 IP=192.168.0.223

default via 192.168.0.1 dev enp23s0f3 ...
                         ↑ 默认路由网卡 = enp23s0f3（跨机通信用这张）
```

**记录三样东西**：
- A 机内网 IP（如 `192.168.0.223`）
- B 机内网 IP（如 `192.168.0.50`）
- 网卡名（两机通常同名，如 `enp23s0f3`；若不同则各记各的）

> ⚠️ **必须用内网 IP，不能用公网 IP**。公网 IP 是云厂商 NAT 上去的，网卡上没绑，LMCache 按网卡名匹配 IP 会失败。

---

## 第一部分：端口放行（两机各做一次）

**两机都要放行以下 TCP 端口**：

```
8010, 8011          # A/B 实例 vLLM API
9000, 9800, 9900    # LMCache controller（A 机监听，B 要连）
8200-8217           # P2P init / lookup
8500-8517           # LMCache worker
```

**云控制台安全组**：入方向加规则，源 IP 填两机所在的内网网段（如 `192.168.0.0/24`）。

**本机防火墙**（两机宿主机各跑）：

```bash
firewall-cmd --add-port=8010-8011/tcp --permanent 2>/dev/null
firewall-cmd --add-port=9000/tcp --permanent 2>/dev/null
firewall-cmd --add-port=8200-8517/tcp --permanent 2>/dev/null
firewall-cmd --add-port=9800-9900/tcp --permanent 2>/dev/null
firewall-cmd --reload 2>/dev/null
```

> ⚠️ 云机器默认安全组只开 22。不放行的话，B 连 A 的 controller:9000 会超时，跨机 P2P 建不起来（表现：KV 命中率 0）。

---

## 第二部分：启动容器（两机各做一次）

### 2.1 拉镜像

```bash
docker pull quay.nju.edu.cn/ascend/vllm-ascend:v0.22.1rc1-a3
```

### 2.2 启动容器

两机各执行（容器名可统一，如 `vllm-022-zj`）：

```bash
docker run -itd \
  --shm-size=200g --privileged --net=host \
  --device=/dev/davinci0  --device=/dev/davinci1  --device=/dev/davinci2  --device=/dev/davinci3 \
  --device=/dev/davinci4  --device=/dev/davinci5  --device=/dev/davinci6  --device=/dev/davinci7 \
  --device=/dev/davinci8  --device=/dev/davinci9  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /var/log/npu:/var/log/npu \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /usr/src/kernels:/usr/src/kernels:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /mnt/sdb/models:/mnt/sdb/models \
  -v /mnt/sdb/zj:/mnt/sdb/zj \
  --name vllm-022-zj \
  --entrypoint /bin/bash \
  quay.nju.edu.cn/ascend/vllm-ascend:v0.22.1rc1-a3
```

**关键参数**：`--net=host`（跨机 P2P 依赖）、`--privileged`（NPU 访问）、`--shm-size=200g`、挂载 `/mnt/sdb/models`（模型）和 `/mnt/sdb/zj`（工作目录）。

### 2.3 进入容器

```bash
docker exec -u root -it vllm-022-zj bash
npu-smi info   # 确认能看到 NPU 且空闲
```

---

## 第三部分：克隆仓库 + 拉子模块（两机各做一次）

本部署用两个仓库，都放在 `/mnt/sdb/zj/p2p/` 下并列：

```bash
mkdir -p /mnt/sdb/zj/p2p
cd /mnt/sdb/zj/p2p

# 1. LMCache-Ascend（dsv4_support_045 分支，带子模块）
git clone -b dsv4_support_045 --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git

# 2. LMCache 上游源码（v0.4.5，与分支名 045 对应）
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
```

**校验**（每条都必须有输出）：

```bash
ls /mnt/sdb/zj/p2p/LMCache-Ascend/pyproject.toml          # 主仓库
ls /mnt/sdb/zj/p2p/LMCache-Ascend/third_party/kvcache-ops/CMakeLists.txt   # 子模块1
ls /mnt/sdb/zj/p2p/LMCache-Ascend/third_party/hcomm/       # 子模块2
ls /mnt/sdb/zj/p2p/LMCache/lmcache/__init__.py             # 上游 LMCache
```

> ⚠️ 两个子模块在 `atomgit.com` / `gitcode.com` 上，国内一般能拉。若 `--recurse-submodules` 卡住，进仓库后单独补：
> ```bash
> cd /mnt/sdb/zj/p2p/LMCache-Ascend
> git submodule update --init third_party/kvcache-ops
> git submodule update --init third_party/hcomm
> ```

---

## 第四部分：安装 LMCache + LMCache-Ascend（两机各做一次）

参考 `reinstall_lmcache.sh` 的三步安装法。把下面整段在**容器内**执行（或存成 `/mnt/sdb/zj/p2p/install.sh` 后 `bash` 运行）：

```bash
#!/usr/bin/bash
set -e

PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
P2P_DIR=/mnt/sdb/zj/p2p

echo "==== [1/3] 安装 LMCache (跳过CUDA扩展, --no-deps 防止连锁升级容器内科学计算库) ===="
cd ${P2P_DIR}/LMCache
NO_CUDA_EXT=1 pip install -e . --no-build-isolation --no-deps -i ${PIP_INDEX}
# 显式补缺的纯 Python 依赖(缺什么补什么, 不要放开依赖解析)
pip install sortedcontainers -i ${PIP_INDEX}

echo "==== [2/3] 安装 LMCache-Ascend (自动注入CANN头文件路径) ===="
cd ${P2P_DIR}/LMCache-Ascend
rm -rf build
RT=$(find /usr/local/Ascend -name "rt_external_device.h" 2>/dev/null | head -1)
if [ -z "$RT" ]; then
    echo "[ERROR] CANN 包里找不到 rt_external_device.h, 无法自动定位 include 路径"
    exit 1
fi
echo "找到头文件: ${RT}"
export CXXFLAGS="-I$(dirname "$(dirname "$RT")")"
pip install -e . --no-build-isolation -i ${PIP_INDEX} --timeout 120

echo "==== [3/3] 验证 ===="
pip show lmcache lmcache-ascend | grep -E "Name|Version|Location"
echo "[DONE] 安装完成"
```

**成功标志**：

```
Name: lmcache
Version: 0.4.5
Location: /mnt/sdb/zj/p2p/LMCache
Name: lmcache-ascend
...
[DONE] 安装完成
```

> - 步骤 2 编译 kvcache-ops C++ 内核，**耗时 5~15 分钟正常**。
> - 若报其它 `ModuleNotFoundError`，`pip install xxx -i ${PIP_INDEX}` 补装即可，**不要放开 `--no-deps`**（防止 pip 连锁升级容器内的 torch/numpy）。
> - 容器重建后 pip 安装会丢，但源码在挂载盘不丢——**重跑本脚本即可恢复**。

装网络工具（容器最小化镜像无 iproute2）：

```bash
apt-get update && apt-get install -y iproute2
ip addr show <网卡名> | grep "inet "   # 容器是 --net=host，应能看到本机内网 IP
```

---

## 第五部分：部署脚本（核心，全部绝对路径）

两个自包含脚本：生成 yaml + 设环境变量 + 起 controller（仅 A）+ 起 vLLM。**先按第 0 步查到的 IP 替换占位符**。

### 5.1 A 机脚本 `start_p2p_a.sh`

在 **A 机容器内**执行以下整段（自动创建脚本）：

```bash
cat > /mnt/sdb/zj/p2p/start_p2p_a.sh <<'SCRIPT_EOF'
#!/bin/bash
# DSv4 + LMCache P2P 双机 · A机(controller + 实例A)
set -euo pipefail

# ===== 按第0步查询结果修改这里 =====
MY_IP="<A机内网IP>"              # 例如 192.168.0.223
NIC="<网卡名>"                    # 例如 enp23s0f3
NPUS="0,1,2,3,4,5,6,7"
# ==================================

CONTROLLER_HOST="${MY_IP}"        # controller 在 A 机本机
INSTANCE_ID="lmcache_colocated_a"
P2P_DIR="/mnt/sdb/zj/p2p"
MODEL_PATH="/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp"
LOG_DIR="${P2P_DIR}/logs"
YAML_FILE="${P2P_DIR}/lmcache-p2p-a.yaml"

mkdir -p "${LOG_DIR}"

# ---- 1. 通用环境变量 ----
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_TRACK_USAGE=false
export GLOO_SOCKET_IFNAME="${NIC}"
export TP_SOCKET_IFNAME="${NIC}"
export HCCL_SOCKET_IFNAME="${NIC}"
export HCCL_IF_IP="${MY_IP}"
export LMCACHE_LOG_LEVEL=INFO
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"

# ---- 2. 生成 LMCache yaml ----
cat > "${YAML_FILE}" <<EOF
chunk_size: 1024
local_cpu: True
max_local_cpu_size: 16
enable_async_loading: True
use_layerwise: False
numa_mode: "auto"
save_unfull_chunk: False

# P2P KV 共享（colocated kv_both，无 enable_pd）
enable_p2p: True
p2p_host: "${MY_IP}"
p2p_init_ports: [8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207]
p2p_lookup_ports: [8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217]
transfer_channel: "hccl"
p2p_use_npu: True
p2p_pull_mode: True
p2p_delay_pull: False
p2p_npu_buffer_size: 134217728

enable_controller: True
lmcache_instance_id: "${INSTANCE_ID}"
controller_pull_url: "${CONTROLLER_HOST}:9800"
controller_reply_url: "${CONTROLLER_HOST}:9900"
lmcache_worker_ports: [8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507]

extra_config:
  save_only_first_rank: true
  lookup_backoff_time: 0.001
EOF
export LMCACHE_CONFIG_FILE="${YAML_FILE}"
echo "[A] Wrote ${YAML_FILE} (id=${INSTANCE_ID})"

# ---- 3. 起 lmcache_controller（仅 A 机）----
echo "[A] starting lmcache_controller ..."
nohup lmcache_controller \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": 9800, "reply": 9900}' \
    > "${LOG_DIR}/controller.log" 2>&1 &
echo "[A] controller pid=$!"
sleep 2

# ---- 4. 起 vLLM 实例 A ----
echo "[A] starting vllm (port 8010) ..."
vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8010 \
    --served-model-name dsv4 \
    --no-enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --api-server-count 1 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --gpu-memory-utilization 0.8 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true}}' \
    2>&1 | tee "${LOG_DIR}/instance_a.log"
SCRIPT_EOF
chmod +x /mnt/sdb/zj/p2p/start_p2p_a.sh
```

### 5.2 B 机脚本 `start_p2p_b.sh`

在 **B 机容器内**执行以下整段：

```bash
cat > /mnt/sdb/zj/p2p/start_p2p_b.sh <<'SCRIPT_EOF'
#!/bin/bash
# DSv4 + LMCache P2P 双机 · B机(实例B, 不起controller)
set -euo pipefail

# ===== 按第0步查询结果修改这里 =====
MY_IP="<B机内网IP>"              # 例如 192.168.0.50
CONTROLLER_HOST="<A机内网IP>"     # controller 在 A 机
NIC="<网卡名>"                    # 例如 enp23s0f3
NPUS="0,1,2,3,4,5,6,7"
# ==================================

INSTANCE_ID="lmcache_colocated_b"
P2P_DIR="/mnt/sdb/zj/p2p"
MODEL_PATH="/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp"
LOG_DIR="${P2P_DIR}/logs"
YAML_FILE="${P2P_DIR}/lmcache-p2p-b.yaml"

mkdir -p "${LOG_DIR}"

# ---- 1. 通用环境变量 ----
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_TRACK_USAGE=false
export GLOO_SOCKET_IFNAME="${NIC}"
export TP_SOCKET_IFNAME="${NIC}"
export HCCL_SOCKET_IFNAME="${NIC}"
export HCCL_IF_IP="${MY_IP}"
export LMCACHE_LOG_LEVEL=INFO
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"

# ---- 2. 生成 LMCache yaml ----
cat > "${YAML_FILE}" <<EOF
chunk_size: 1024
local_cpu: True
max_local_cpu_size: 16
enable_async_loading: True
use_layerwise: False
numa_mode: "auto"
save_unfull_chunk: False

# P2P KV 共享（colocated kv_both，无 enable_pd）
enable_p2p: True
p2p_host: "${MY_IP}"
p2p_init_ports: [8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207]
p2p_lookup_ports: [8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217]
transfer_channel: "hccl"
p2p_use_npu: True
p2p_pull_mode: True
p2p_delay_pull: False
p2p_npu_buffer_size: 134217728

enable_controller: True
lmcache_instance_id: "${INSTANCE_ID}"
controller_pull_url: "${CONTROLLER_HOST}:9800"
controller_reply_url: "${CONTROLLER_HOST}:9900"
lmcache_worker_ports: [8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507]

extra_config:
  save_only_first_rank: true
  lookup_backoff_time: 0.001
EOF
export LMCACHE_CONFIG_FILE="${YAML_FILE}"
echo "[B] Wrote ${YAML_FILE} (id=${INSTANCE_ID})"

# ---- 3. 起 vLLM 实例 B（不起 controller）----
echo "[B] starting vllm (port 8011) ..."
vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8011 \
    --served-model-name dsv4 \
    --no-enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --api-server-count 1 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --gpu-memory-utilization 0.8 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true}}' \
    2>&1 | tee "${LOG_DIR}/instance_b.log"
SCRIPT_EOF
chmod +x /mnt/sdb/zj/p2p/start_p2p_b.sh
```

### 5.3 A、B 脚本差异速查

| 项 | A 机 | B 机 |
|---|---|---|
| `MY_IP` | A 机内网 IP | B 机内网 IP |
| `CONTROLLER_HOST` | A 机自己 | **A 机 IP**（B 连 A 的 controller） |
| `lmcache_instance_id` | `lmcache_colocated_a` | `lmcache_colocated_b` |
| yaml 文件 | `lmcache-p2p-a.yaml` | `lmcache-p2p-b.yaml` |
| `--port` | 8010 | 8011 |
| controller | A 起 | B 不起 |
| 其余（vllm 参数/环境变量/yaml 内容） | **完全一致** | **完全一致** |

---

## 第六部分：启动流程

### 6.1 清理残留（两机各做，防止端口被占）

```bash
pkill -9 -f vllm
pkill -9 -f lmcache_controller
sleep 3
npu-smi info   # 确认卡全空闲
```

> ⚠️ 必做。controller 的 9800/9900 被残留进程占用会导致新 controller 起不来 → 跨机 P2P 失效 → KV 命中率 0。

### 6.2 A 机启动

```bash
bash /mnt/sdb/zj/p2p/start_p2p_a.sh
```

**期望输出顺序**：

```
[A] Wrote /mnt/sdb/zj/p2p/lmcache-p2p-a.yaml (id=lmcache_colocated_a)
[A] starting lmcache_controller ...
[A] controller pid=xxx
[A] starting vllm (port 8010) ...
... Loading safetensors ...        ← 10~20 分钟正常
Application startup complete
```

**验证 controller 成功**（另开 A 机终端）：

```bash
ss -lntp | grep -E "9000|9800|9900"
# 期望: 3 个端口都在 LISTEN
grep -i "error\|traceback" /mnt/sdb/zj/p2p/logs/controller.log
# 期望: 无输出
```

**验证 vLLM 就绪**：

```bash
curl http://<A机内网IP>:8010/v1/models
# 期望: 返回含 "id": "dsv4" 的 JSON
```

### 6.3 B 机启动（A ready 之后）

```bash
bash /mnt/sdb/zj/p2p/start_p2p_b.sh
```

**期望**：同样加载 10~20 分钟后 `Application startup complete`，然后：

```bash
curl http://<B机内网IP>:8011/v1/models
```

**验证 B 注册到了 A 的 controller**（A 机另开终端）：

```bash
grep -Ei "register|colocated" /mnt/sdb/zj/p2p/logs/controller.log | tail -5
# 期望: lmcache_colocated_a 和 lmcache_colocated_b 都 Registered
```

---

## 第七部分：Smoke 测试（验证跨机 P2P KV 复用）

### 7.1 发起测试

在任意能访问两机的机器上执行（A 机容器内即可）：

```bash
# 构造一个 ~1024 token 的长 prompt（重复拼接）
PROMPT=$(python3 -c "print('Explain KV cache P2P sharing across Ascend instances. '*80)")

# 第1步: 打 A（A 算完并存 KV）
curl -s http://<A机内网IP>:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"dsv4\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee /tmp/p2p_a.json | tail -c 300
echo

# 第2步: 打 B（相同 prompt，B 应通过 controller 查到 A 有这段 KV，跨机 pull 复用）
curl -s http://<B机内网IP>:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"dsv4\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":16,\"temperature\":0}" \
  | tee /tmp/p2p_b.json | tail -c 300
echo
```

**期望**：两次都返回 JSON 且带 `"finish_reason":"stop"`。

### 7.2 验证 P2P 真正跨机生效（关键）

```bash
# A 机: 看 Stored
grep -Ei "store|saved|register|colocated_a" /mnt/sdb/zj/p2p/logs/instance_a.log | tail -20

# B 机: 看 Retrieved / pull（关键证据）
grep -Ei "retriev|pull|lookup|hit|p2p|colocated_b" /mnt/sdb/zj/p2p/logs/instance_b.log | tail -20
```

**成功标志**：

| 日志 | 期望关键词 |
|---|---|
| A `instance_a.log` | `Stored ...`、`lmcache_colocated_a registered` |
| B `instance_b.log` | `Retrieved ...`、`P2P lookup hit`、**pull 源是 A 机内网 IP** |
| `controller.log` | `lmcache_colocated_a` 和 `lmcache_colocated_b` 都 registered |

> B 日志里 pull 源 IP 是 **A 机的内网 IP** —— 这就是跨机 P2P 生效的铁证。

---

## 第八部分：停止与清理

两机各执行：

```bash
pkill -9 -f vllm
pkill -9 -f lmcache_controller
npu-smi info   # 确认卡空闲
```

---

## 附录

### A. 版本对应

| 组件 | 版本/来源 |
|---|---|
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.22.1rc1-a3` |
| LMCache-Ascend | `LMCache/LMCache-Ascend` @ `dsv4_support_045` 分支 |
| 子模块 kvcache-ops | `atomgit.com/openeuler/kvcache-ops`（随主仓库 submodule 拉取） |
| 子模块 hcomm | `gitcode.com/cann/hcomm`（随主仓库 submodule 拉取） |
| LMCache 上游 | `LMCache/LMCache` @ `v0.4.5` tag |

### B. Troubleshooting

| 现象 | 原因 / 处理 |
|---|---|
| **controller 起不来：`Address already in use (9800)`** | 残留 controller 占端口。`pkill -9 -f lmcache_controller` 后重启。**这是 KV 命中率归零的头号原因** |
| **B 连不上 controller（超时）** | 端口 9000/9800/9900 没放行（云安全组 + firewalld 都查） |
| **报 `cannot find IP for NIC <网卡名>`** | 网卡名写错，或容器没装 iproute2。`ip addr` 核对实际网卡名 |
| **编译报找不到 CANN 头文件** | `find /usr/local/Ascend -name rt_external_device.h` 为空 → CANN 安装异常，找运维 |
| **`ModuleNotFoundError`** | `pip install xxx -i https://mirrors.aliyun.com/pypi/simple/` 补装，**不要去掉 `--no-deps`** |
| **smoke 两条都 200 但 B `Retrieved 0`** | ① 查 `lmcache-p2p-b.yaml` 里 `enable_p2p: True`；② 查 controller 日志两 instance 都注册；③ 确认两次请求 prompt **完全一致** |
| **加载 15 分钟没 ready** | 看 `logs/instance_a.log` 末尾，还在 `Loading safetensors`/`Compiling` 就继续等 |

### C. 执行顺序速查

```
两机各做: 查IP(第0步) → 放端口(一) → 起容器+clone+子模块(二/三) → 安装(四)
A 机:    bash /mnt/sdb/zj/p2p/start_p2p_a.sh → 等 ready + 验证 controller
B 机:    bash /mnt/sdb/zj/p2p/start_p2p_b.sh → 等 ready
smoke:   打A → 打B(相同prompt) → grep日志验证 Retrieved
清理:    两机 pkill vllm / lmcache_controller
```
