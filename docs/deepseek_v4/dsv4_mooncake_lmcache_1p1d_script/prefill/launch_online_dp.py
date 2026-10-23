import argparse
import multiprocessing
import os
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default=12345, help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="Starting port for the engine.")
    parser.add_argument(
        "--device-start",
        type=int,
        default=0,
        help="First device index assigned to this local DP group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without starting vLLM.",
    )
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
device_start = args.device_start
dry_run = args.dry_run
script_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(script_dir, "run_dp_template.sh")


def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        template_path,
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        str(dp_rpc_port),
        str(tp_size),
    ]
    if dry_run:
        print(shlex.join(command), flush=True)
        return
    subprocess.run(command, check=True, cwd=script_dir)


if __name__ == "__main__":
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)
    processes = []
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        first_device = device_start + i * tp_size
        visible_devices = ",".join(
            str(x) for x in range(first_device, first_device + tp_size)
        )
        if dry_run:
            run_command(visible_devices, dp_rank, vllm_engine_port)
            continue
        process = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_engine_port))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
