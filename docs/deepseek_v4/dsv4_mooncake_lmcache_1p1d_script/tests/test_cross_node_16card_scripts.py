import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def run_script(path, *args, env=None, cwd=None):
    return subprocess.run(
        ["bash", str(path), *args],
        cwd=str(cwd or SCRIPT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_dry_run_commands(stdout):
    return [shlex.split(line) for line in stdout.splitlines() if line.strip()]


class CrossNodeLauncherTest(unittest.TestCase):
    def test_prefill_start_maps_dp8_tp2_to_all_16_npus(self):
        result = run_script(
            SCRIPT_ROOT / "prefill" / "start_p_sever.sh", "--dry-run"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = parse_dry_run_commands(result.stdout)
        self.assertEqual(len(commands), 8)
        self.assertEqual(
            [command[-7] for command in commands],
            ["0,1", "2,3", "4,5", "6,7", "8,9", "10,11", "12,13", "14,15"],
        )
        self.assertEqual([command[-6] for command in commands], [str(x) for x in range(7100, 7108)])
        self.assertEqual([command[-5] for command in commands], ["8"] * 8)
        self.assertEqual([command[-4] for command in commands], [str(x) for x in range(8)])
        self.assertEqual([command[-1] for command in commands], ["2"] * 8)

    def test_decode_start_maps_dp16_tp1_to_all_16_npus(self):
        result = run_script(
            SCRIPT_ROOT / "decode" / "start_d_sever.sh", "--dry-run"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = parse_dry_run_commands(result.stdout)
        self.assertEqual(len(commands), 16)
        self.assertEqual([command[-7] for command in commands], [str(x) for x in range(16)])
        self.assertEqual([command[-6] for command in commands], [str(x) for x in range(7100, 7116)])
        self.assertEqual([command[-5] for command in commands], ["16"] * 16)
        self.assertEqual([command[-4] for command in commands], [str(x) for x in range(16)])
        self.assertEqual([command[-1] for command in commands], ["1"] * 16)


class CrossNodeTemplateTest(unittest.TestCase):
    def _run_template(self, role_dir, arguments):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_bin = tmp_path / "bin"
            fake_home = tmp_path / "home"
            fake_bin.mkdir()
            fake_home.mkdir()
            fake_vllm = fake_bin / "vllm"
            fake_vllm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "print('VISIBLE=' + os.environ.get('ASCEND_RT_VISIBLE_DEVICES', ''))\n"
                "print('LMCACHE=' + os.environ.get('LMCACHE_CONFIG_FILE', ''))\n"
                "print('VLLM_ARGS=' + json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_vllm.chmod(0o755)
            env = os.environ.copy()
            env["HOME"] = str(fake_home)
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
            env.pop("LMCACHE_CONFIG_FILE", None)

            result = run_script(
                SCRIPT_ROOT / role_dir / "run_dp_template.sh",
                *arguments,
                env=env,
                cwd=tmp_path,
            )
            lines = result.stdout.splitlines()
            args_line = next(
                (line for line in lines if line.startswith("VLLM_ARGS=")), None
            )
            vllm_args = json.loads(args_line.removeprefix("VLLM_ARGS=")) if args_line else []
            return result, vllm_args

    def test_prefill_passes_rank_and_dp8_tp2_connector_topology(self):
        result, vllm_args = self._run_template(
            "prefill",
            ("14,15", "7107", "8", "7", "7.246.92.163", "12321", "2"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VISIBLE=14,15", result.stdout)
        self.assertIn("/lmcache-prefill-config.yaml", result.stdout)
        self.assertEqual(vllm_args[vllm_args.index("--data-parallel-rank") + 1], "7")
        self.assertEqual(vllm_args[vllm_args.index("--tensor-parallel-size") + 1], "2")
        kv_config = json.loads(vllm_args[vllm_args.index("--kv-transfer-config") + 1])
        self.assertEqual(kv_config["engine_id"], "0")
        connectors = kv_config["kv_connector_extra_config"]["connectors"]
        self.assertEqual(connectors[0]["kv_port"], "36000")
        self.assertEqual(
            connectors[0]["kv_connector_extra_config"],
            {
                "prefill": {"dp_size": 8, "tp_size": 2},
                "decode": {"dp_size": 16, "tp_size": 1},
            },
        )
        self.assertEqual(connectors[1]["kv_connector"], "LMCacheAscendConnectorV1Dynamic")

    def test_decode_passes_rank_and_dp16_tp1_connector_topology(self):
        result, vllm_args = self._run_template(
            "decode",
            ("15", "7115", "16", "15", "7.246.92.165", "12322", "1"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VISIBLE=15", result.stdout)
        self.assertIn("LMCACHE=", result.stdout)
        self.assertEqual(vllm_args[vllm_args.index("--data-parallel-rank") + 1], "15")
        self.assertEqual(vllm_args[vllm_args.index("--tensor-parallel-size") + 1], "1")
        kv_config = json.loads(vllm_args[vllm_args.index("--kv-transfer-config") + 1])
        self.assertEqual(kv_config["engine_id"], "1")
        self.assertEqual(kv_config["kv_port"], "36100")
        self.assertEqual(
            kv_config["kv_connector_extra_config"],
            {
                "prefill": {"dp_size": 8, "tp_size": 2},
                "decode": {"dp_size": 16, "tp_size": 1},
            },
        )
        self.assertEqual(kv_config["kv_connector"], "MooncakeHybridConnector")


class CrossNodeProxyTest(unittest.TestCase):
    def test_proxy_starts_with_8_prefillers_and_16_decoders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_bin = Path(tmp_dir) / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8"
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]

            result = run_script(
                SCRIPT_ROOT / "prefill" / "start_proxy.sh", env=env
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        prefill_hosts = arguments[
            arguments.index("--prefiller-hosts") + 1 : arguments.index("--prefiller-ports")
        ]
        prefill_ports = arguments[
            arguments.index("--prefiller-ports") + 1 : arguments.index("--decoder-hosts")
        ]
        decode_hosts = arguments[
            arguments.index("--decoder-hosts") + 1 : arguments.index("--decoder-ports")
        ]
        decode_ports = arguments[arguments.index("--decoder-ports") + 1 :]
        self.assertEqual(len(prefill_hosts), 8)
        self.assertEqual(prefill_ports, [str(x) for x in range(7100, 7108)])
        self.assertEqual(len(decode_hosts), 16)
        self.assertEqual(decode_ports, [str(x) for x in range(7100, 7116)])


class ShellSyntaxTest(unittest.TestCase):
    def test_all_shell_scripts_parse(self):
        scripts = sorted(SCRIPT_ROOT.glob("**/*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
