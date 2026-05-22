
# Deployment guide

## for vLLM-Ascend

### Docker
<!-- Pull pre built image-->
To build the container image from the dockerfile, run: 
```
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:latest .
```
Once you have built the image, you can run it with:
```
docker run -it \
--shm-size=200g --privileged --net=host \
--cap-add=SYS_RESOURCE \
--cap-add=IPC_LOCK \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
--device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /var/log/npu:/var/log/npu \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/localtime:/etc/localtime \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/src/kernels:/usr/src/kernels:ro \
-v /data:/data \
--name lmcache-ascend-dev \
--entrypoint /bin/bash \
lmcache-ascend:latest
```

You can optionally modify the command to add LMCache configurations through env variables if that is your preferred way (we encourage the LMCache configurationfile), for example:
```
  --env "LMCACHE_CHUNK_SIZE=256" \
```
### Kubernetes
An example yaml file is:
```
apiVersion: v1
kind: Pod
metadata:
  name: lmcache-ascend
spec:
  containers:
  - name: lmcache-ascend
    image: lmcache-ascend:latest	# Replace with your image
    command: ["/bin/bash"]			# Replace with LLM serving command
    securityContext:
       allowPrivilegeEscalation: false
       capabilities:
          add: ["SYS_RESOURCE", "IPC_LOCK"]
    env:
    - name: ASCEND_VISIBLE_DEVICES  # Only if on Ascend Docker
      value: "0,1,2,3,4,5,6,7"        # Replace with your devices
    - name: ASCEND_RT_VISIBLE_DEVICES
      value: "0,1,2,3,4,5,6,7"
    - name: ASCEND_TOTAL_MEMORY_GB
      value: "32"
    - name: VLLM_TARGET_DEVICE
      value: "npu"
    ports:
    - containerPort: 8000
    - containerPort: 8001
    volumeMounts:
    - name: ascend-driver
      mountPath: /usr/local/Ascend/driver
    - name: localtime
      mountPath: /etc/localtime
      readOnly: true
    - name: npu-log
      mountPath: /var/log/npu
    - name: davinci-manager
      mountPath: /dev/davinci_manager
    - name: devmm-svm
      mountPath: /dev/devmm_svm
    - name: ascend-install-info
      mountPath: /etc/ascend_install.info
      subPath: ascend_install.info
    - name: hccn-conf
      mountPath: /etc/hccn.conf
      subPath: hccn.conf
  volumes:
  - name: ascend-driver
    hostPath:
      path: /usr/local/Ascend/driver
  - name: localtime
    hostPath:
      path: /etc/localtime
  - name: npu-log
    hostPath:
      path: /var/log/npu
  - name: davinci-manager
    hostPath:
      path: /dev/davinci_manager
  - name: devmm-svm
    hostPath:
      path: /dev/devmm_svm
  - name: ascend-install-info
    hostPath:
      path: /etc/ascend_install.info
      type: File
  - name: hccn-conf
    hostPath:
      path: /etc/hccn.conf
      type: File
```

Notes:
* Allocating the NPUs to the pod/container is possible through the environmental variables ASCEND_VISIBLE_DEVICES and ASCEND_RT_VISIBLE_DEVICES only when K8s is relying on Ascend Docker. Nevertheless, when the Ascend device plugin available in the cluster, it is preferable to assign NPUs through the dedicated resource field. If K8s is not relying on Ascend Docker and the Ascend device plugin is not available, please mount the devices /dev/davinci[0-7] one by one in the traditional way.
* The capabilities "SYS_RESOURCE" and "IPC_LOCK" are not required for Ascend driver v25.
* The capability SYS_RESOURCE is required to allow the container to lock an amount of memory beyond the standard. When such capability is given to the pod, a user within the pod can change the soft and hard resource limits (RLIMITS, and in particular RLIMIT_MEMLOCK) of the process that will be started in the container and can lock more memory than the limits.
While the pod is running, the pod user can run the following command within the pod to check the current RLIMITS:
```
ulimit -l
```
The user can also change the RLIMITS with the following command:
```
ulimit -l unlimited # Update with the amount of memory you need to lock in KBs
```
Locking a large amount of memory is required when the version of the Ascend driver is < 25. We warmly encourage the user to update the driver version to 25. 

## for vLLM-MindSpore

### Docker

1. Clone LMCache-Ascend Repo
Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

2. Build Docker Image
```bash
cd /workspace/LMCache-Ascend
docker build -f docker/mindspore/Dockerfile.a2.openEuler -t lmcache-ascend:v0.4.4-mindspore2.7.1.post1-openeuler .
```

3. Start Container
Once that is built, run it with the following cmd
```bash
docker run -itd \
    --shm-size 200g --privileged \
    --net=host \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /var/log/npu/:/var/log/npu \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /lib/modules:/lib/modules:ro \
    -v /usr/src/kernels:/usr/src/kernels:ro \
    -v /mnt/storage1/data:/data \
    -v /home:/home \
    --name lmcache-ascend-ms \
    --entrypoint /bin/bash \
    lmcache-ascend:v0.4.4-mindspore2.7.1.post1-openeuler

docker exec -it -u root lmcache-ascend-ms bash
```

For further info about deployment notes, please refer to the [guide about deployment](docs/deployment.md)

### Manual Installation

1. Start the base container
```bash
docker run -itd \
--shm-size 200g --privileged \
--net=host \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
--device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /var/log/npu/:/var/log/npu \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /lib/modules:/lib/modules:ro \
-v /usr/src/kernels:/usr/src/kernels:ro \
-v /mnt/storage1/data:/data \
-v /home/:/home \
--name lmcache-ascend-ms \
--entrypoint /bin/bash \
hub.oepkgs.net/oedeploy/openeuler/aarch64/intelligence_boom:0.2.0-aarch64-800I-A2-mindspore2.7.1.post1-openeuler24.03-lts-sp2-20260116

docker exec -it -u root lmcache-ascend-ms bash
```

2. Install LMCache

```bash
NO_CUDA_EXT=1 pip install lmcache==0.4.4 --no-deps
```

3. Install LMCache-Ascend

```bash
git clone --recurse-submodules -b v0.4.4 https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
USE_MINDSPORE=1 pip install -r requirement_ms.txt --no-build-isolation -v -e .
```

### Usage

We introduce a dynamic KVConnector via LMCacheAscendConnectorV1Dynamic, therefore LMCache-Ascend Connector can be used via the kv transfer config in the two following setting.

#### Online serving
```bash
python \
    -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model /data/models/Qwen/Qwen3-32B \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'
```

#### Offline
```python
ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
```
