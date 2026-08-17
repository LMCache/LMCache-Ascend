
# Deployment guide

## Docker
<!-- Pull pre built image-->
To build the container image from the dockerfile, run: 
```
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
docker build -t lmcache-ascend:latest --file docker/Dockerfile.a2.openEuler .
```
Once you have built the image, you can run it with:
```
export IMAGE=lmcache-ascend:latest  ## Replace with your image name
DEVICE_LIST="0,1,2,3,4,5,6,7"
docker run -it \
    --privileged \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    -p 8000:8000 \
    -p 8001:8001 \
    --name lmcache-ascend-dev \
    -e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_TOTAL_MEMORY_GB=32 \
    -e VLLM_TARGET_DEVICE=npu \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /var/log/npu:/var/log/npu \
    -v /dev/davinci_manager:/dev/davinci_manager \
    -v /dev/devmm_svm:/dev/devmm_svm \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -it $IMAGE bash
```
This command requires the Ascend Docker runtime. Alternatively you can use the following flags for the docker command:
```
# Update DEVICE according to your devices
--device  /dev/davinci0  \
--device  /dev/davinci1  \
--device  /dev/davinci_manager  \
--device  /dev/devmm_svm  \
--device  /dev/hisi_hdc  \
```
You can optionally modify the command to add LMCache configurations through env variables if that is your preferred way (we encourage the LMCache configurationfile), for example:
```
  --env "LMCACHE_CHUNK_SIZE=256" \
```
## Kubernetes
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

