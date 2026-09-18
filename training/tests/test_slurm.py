"""Offline launch tests: simulate srun and GPU libraries, never submit jobs."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("slurm_training", TRAINING_DIR / "training.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


@pytest.fixture
def launcher_env(exported_training_env, tmp_path):
    interpreter = tmp_path / "python capture"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import contextlib, io, json, os, sys\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    Path(os.environ['PREFLIGHT_CAPTURE']).write_text(json.dumps(sys.argv[3:]))\n"
        "    gpu = SimpleNamespace(name='fake H100', total_memory=80 * 2**30)\n"
        "    cuda = SimpleNamespace(is_available=lambda: os.environ.get('FAKE_CUDA', '1') == '1',\n"
        "        device_count=lambda: int(os.environ.get('FAKE_GPUS', '8')),\n"
        "        get_device_properties=lambda i: gpu)\n"
        "    sys.modules['torch'] = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='fake'), __version__='fake')\n"
        "    sys.modules['deepspeed'] = SimpleNamespace(__version__='fake') if os.environ.get('FAKE_DEEPSPEED', '1') == '1' else None\n"
        "    code = sys.argv[2]\n"
        "    sys.argv = [sys.argv[0], *sys.argv[3:]]\n"
        "    with contextlib.redirect_stdout(io.StringIO()): exec(code)\n"
        "    sys.exit(0)\n"
        "keys = ['OUTPUT_DIR', 'WANDB_NAME', 'CUDA_VISIBLE_DEVICES', 'CC', 'CXX', 'NPROC_PER_NODE']\n"
        "print(json.dumps({'interpreter': sys.argv[0], 'argv': sys.argv[1:],\n"
        "    'env': {key: os.environ.get(key) for key in keys}}))\n"
    )
    interpreter.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "Path(os.environ['SRUN_CAPTURE']).write_text(json.dumps(args))\n"
        "command = args[args.index('bash'):]\n"
        "os.execvp(command[0], command)\n"
    )
    srun.chmod(0o755)
    return exported_training_env | {
        "PYTHON_BIN": str(interpreter), "NPROC_PER_NODE": "8", "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SRUN_CAPTURE": str(tmp_path / "srun.json"), "PREFLIGHT_CAPTURE": str(tmp_path / "preflight.json"),
    }


def invoke(path, env, cwd, *args):
    return subprocess.run(["bash", str(path), *args], env=env, cwd=cwd, text=True, capture_output=True)


def slurm_env(env):
    return env | {"SLURM_JOB_ID": "1234", "SLURM_SUBMIT_DIR": str(TRAINING_DIR.parents[1]),
                  "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1"}


def parse(args=()):
    return training.parse_args(["--model", "test", "--dataset", "/data", "--output-dir", "/output",
                                "--report-to", "none", *args])


def test_source_overwrites_existing_values(exported_training_env, tmp_path):
    template = TRAINING_DIR / "envs/tp1/.env.example.sh"
    env = exported_training_env | {key: "old value" for key in exported_training_env
                                  if key in {"MODEL", "DATASET", "OUTPUT_DIR", "REPORT_TO",
                                             "WANDB_NAME", "CC", "CXX", "PYTHON_BIN"}}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; "$2" -c "import json,os; print(json.dumps(dict(os.environ)))"',
         "source-test", str(template), sys.executable],
        env=env, cwd=tmp_path, check=True, text=True, capture_output=True,
    )
    actual = json.loads(result.stdout)
    for key in ["MODEL", "DATASET", "OUTPUT_DIR", "REPORT_TO", "WANDB_NAME", "CC", "CXX", "PYTHON_BIN"]:
        assert actual[key] == exported_training_env[key]
    assert actual["CC"] == "/cm/local/apps/gcc/14.2.0/bin/gcc"
    assert actual["WANDB_NAME"] == "experiment-001"
    assert "NPROC_PER_NODE" not in actual


@pytest.mark.parametrize("key", ["MODEL", "DATASET", "OUTPUT_DIR", "REPORT_TO", "PYTHON_BIN"])
@pytest.mark.parametrize("empty", [True, False])
def test_missing_inputs_fail_without_defaults(launcher_env, tmp_path, key, empty):
    env = launcher_env.copy()
    if empty:
        env[key] = ""
    else:
        del env[key]
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, "--dry-run")
    assert result.returncode != 0
    assert key in result.stderr
    assert not Path(env["PREFLIGHT_CAPTURE"]).exists()


@pytest.mark.parametrize("key", ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME"])
def test_wandb_required_only_when_enabled(launcher_env, tmp_path, key):
    env = launcher_env | {"REPORT_TO": "wandb"}
    del env[key]
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, "--dry-run")
    assert result.returncode != 0
    assert key in result.stderr
    env["REPORT_TO"] = "none"
    assert invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, "--dry-run").returncode == 0


def test_spooled_slurm_has_no_environment_directory_dependency(launcher_env, tmp_path):
    workspace = tmp_path / "server workspace"
    directory = workspace / "scripts/training"
    directory.mkdir(parents=True)
    (directory / "run_training.sh").write_text((TRAINING_DIR / "run_training.sh").read_text())
    spooled = tmp_path / "slurm_script"
    spooled.write_text((TRAINING_DIR / "submit_training.sbatch").read_text())
    env = slurm_env(launcher_env) | {"SLURM_SUBMIT_DIR": str(workspace), "NPROC_PER_NODE": "3"}
    result = invoke(spooled, env, tmp_path, "--smoke")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["argv"][:6] == [
        "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0",
    ]
    args = training.parse_args(payload["argv"][7:])
    assert (args.max_steps, args.max_train_samples) == (2, 32)
    assert args.batch_size * args.gradient_accumulation_steps * 8 == 16
    assert args.output_dir == Path(env["OUTPUT_DIR"])
    assert not (directory / "envs").exists()
    assert json.loads(Path(env["SRUN_CAPTURE"]).read_text())[:4] == [
        "--nodes=1", "--ntasks=1", "--gpu-bind=none", "--kill-on-bad-exit=1",
    ]


@pytest.mark.parametrize("launcher", ["run_training.sh", "submit_training.sbatch"])
def test_launcher_preserves_manual_settings_and_does_not_source(launcher_env, tmp_path, launcher):
    forbidden = tmp_path / "no-source.sh"
    forbidden.write_text("exit 99\n")
    env = slurm_env(launcher_env) | {
        "TRAINING_ENV_FILE": str(forbidden), "MODEL": "model with spaces",
        "CC": "/custom/gcc", "CXX": "/custom/g++", "CUDA_VISIBLE_DEVICES": "7,6,5,4,3,2,1,0",
        "MAX_STEPS": "999", "DTYPE": "fp32", "LEARNING_RATE": "123",
    }
    result = invoke(TRAINING_DIR / launcher, env, tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    args = training.parse_args(payload["argv"][7:])
    assert args.model == "model with spaces"
    assert args.max_steps == -1 and args.dtype == "bf16" and args.learning_rate == 2e-5
    for key in ["CC", "CXX", "CUDA_VISIBLE_DEVICES", "OUTPUT_DIR", "WANDB_NAME"]:
        assert payload["env"][key] == env[key]


@pytest.mark.parametrize("flag", ["--dry-run", "--help", "-h"])
def test_preview_needs_no_workers_or_gpu(launcher_env, tmp_path, flag):
    env = launcher_env | {"FAKE_GPUS": "0"}
    del env["NPROC_PER_NODE"]
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, flag)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"][0] == str(TRAINING_DIR / "training.py")
    assert not Path(env["PREFLIGHT_CAPTURE"]).exists()


@pytest.mark.parametrize("workers,visible", [("8", "0"), ("8", "4"), ("4", "8")])
def test_gpu_worker_mismatch_blocks_training(launcher_env, tmp_path, workers, visible):
    env = slurm_env(launcher_env) | {"NPROC_PER_NODE": workers, "FAKE_GPUS": visible}
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path)
    assert result.returncode != 0
    assert "GPU/worker mismatch" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("value", ["", "0", "-1", "eight"])
def test_invalid_worker_count(launcher_env, tmp_path, value):
    result = invoke(TRAINING_DIR / "run_training.sh", launcher_env | {"NPROC_PER_NODE": value}, tmp_path)
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_nested_torchrun_is_rejected(launcher_env, tmp_path):
    result = invoke(TRAINING_DIR / "run_training.sh", launcher_env | {"LOCAL_RANK": "0"}, tmp_path)
    assert result.returncode == 2
    assert "nested torchrun" in result.stderr


def test_slurm_rejects_wrong_context(launcher_env, tmp_path):
    for env in [launcher_env, slurm_env(launcher_env) | {"SLURM_NTASKS": "8"},
                slurm_env(launcher_env) | {"SLURM_JOB_NUM_NODES": "2"}]:
        assert invoke(TRAINING_DIR / "submit_training.sbatch", env, tmp_path).returncode != 0


def test_zero2_config_has_auto_batch_and_no_offload():
    config = json.loads((TRAINING_DIR / "deepspeed_zero2.json").read_text())
    assert config["zero_optimization"]["stage"] == 2
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]
    for key in ["train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "gradient_clipping"]:
        assert config[key] == "auto"


def test_fixed_recipe_forwards_deepspeed(monkeypatch):
    import trl
    monkeypatch.setattr(trl, "SFTConfig", lambda **kwargs: kwargs)
    config = training.build_training_config(parse())
    assert config["deepspeed"] == str(TRAINING_DIR / "deepspeed_zero2.json")
    assert config["gradient_accumulation_steps"] == 2
    assert config["model_init_kwargs"]["device_map"] is None


@pytest.mark.parametrize("rank", [0, 1])
def test_output_errors_are_shared_with_every_rank(tmp_path, monkeypatch, rank):
    import accelerate.utils

    (tmp_path / "existing").write_text("keep")
    args = parse(["--output-dir", str(tmp_path)])
    calls = []

    def broadcast(values, from_process):
        calls.append(from_process)
        if rank == 0:
            assert "nonempty" in values[0]
        else:
            assert values == [None]
            values[0] = "Output directory is nonempty"

    monkeypatch.setattr(accelerate.utils, "broadcast_object_list", broadcast)
    if rank != 0:
        monkeypatch.setattr(training, "check_output_directory", lambda *a: pytest.fail("Only rank zero checks"))
    with pytest.raises(ValueError, match="nonempty"):
        training.check_distributed_output_directory(args, SimpleNamespace(world_size=2, process_index=rank))
    assert calls == [0]


def test_worker_ignores_files_created_after_main_check(tmp_path, monkeypatch):
    import accelerate.utils

    (tmp_path / "created-by-rank-zero").write_text("keep")
    monkeypatch.setattr(accelerate.utils, "broadcast_object_list", lambda values, from_process: values)
    args = parse(["--output-dir", str(tmp_path)])
    training.check_distributed_output_directory(args, SimpleNamespace(world_size=2, process_index=1))
