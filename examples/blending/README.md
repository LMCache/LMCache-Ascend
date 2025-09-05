# ad-hoc modification to vllm-ascend

```
diff --git a/vllm_ascend/worker/worker.py b/vllm_ascend/worker/worker.py
index bffc6a8..d4de172 100644
--- a/vllm_ascend/worker/worker.py
+++ b/vllm_ascend/worker/worker.py
@@ -238,6 +238,7 @@ class NPUWorker(LocalOrDistributedWorkerBase):
             context = nullcontext()  # type: ignore
         with context:
             self.model_runner.load_model()
+        ensure_kv_transfer_initialized(self.vllm_config)
 
     def start_profile(self):
         if self.profiler is None:
diff --git a/vllm_ascend/worker/worker_v1.py b/vllm_ascend/worker/worker_v1.py
index f59daea..a777984 100644
--- a/vllm_ascend/worker/worker_v1.py
+++ b/vllm_ascend/worker/worker_v1.py
@@ -49,6 +49,8 @@ from vllm_ascend.utils import (check_kv_cache_bytes_cache_exist,
                                read_kv_cache_bytes_from_file,
                                sleep_mode_enabled, try_register_lib)
 from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
+from lmcache.v1.compute.models.utils import VLLMModelTracker
+from lmcache.integration.vllm.utils import ENGINE_NAME
 
 
 class NPUWorker(WorkerBase):
@@ -241,6 +243,9 @@ class NPUWorker(WorkerBase):
             context = nullcontext()  # type: ignore
         with context:
             self.model_runner.load_model()
+        
+        VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
+        ensure_kv_transfer_initialized(self.vllm_config)
 
     def compile_or_warm_up_model(self) -> None:
         warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
@@ -323,7 +328,7 @@ class NPUWorker(WorkerBase):
             parallel_config.expert_tensor_parallel_size,
             parallel_config.world_size_across_dp,
         )
-        ensure_kv_transfer_initialized(self.vllm_config)
+        # ensure_kv_transfer_initialized(self.vllm_config)
 
     def _init_profiler(self):
         # Torch profiler. Enabled and configured through env vars:
```

# Usage

`python script.py model_path 0.05` should yield no connector (ie `# #`) while `python script.py model_path 1.0` should.

The latter should yield the same result as `python script.py model_path 1.0 --no-blend`, although some precision issues may cause some difference.