"""Offline launcher/config tests. No sbatch submission, CUDA, or DeepSpeed init."""

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
def launcher_env(tmp_path):
    """Replace Python and srun with local argv captures, not training processes."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "WANDB_")) and key not in {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "NPROC_PER_NODE", "DEEPSPEED_CONFIG",
            "WORKSPACE_DIR", "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "OUTPUT_DIR",
            "REPORT_TO", "DTYPE", "CUDA_VISIBLE_DEVICES", "CHAT_TEMPLATE",
        }
    }
    interpreter = tmp_path / "python capture"
    interpreter.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if sys.argv[1:2] == ['-c']: sys.exit(0)  # simulate GPU/dependency preflight\n"
        "keys = ['OUTPUT_DIR', 'WANDB_NAME', 'CUDA_VISIBLE_DEVICES', 'BATCH_SIZE', "
        "'GRADIENT_ACCUMULATION_STEPS', 'DEEPSPEED_CONFIG', 'NPROC_PER_NODE']\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}))\n"
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
    env.update({
        "PYTHON_BIN": str(interpreter),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SRUN_CAPTURE": str(tmp_path / "srun.json"),
    })
    return env


def invoke(path, env, cwd, *args):
    return subprocess.run(["bash", str(path), *args], env=env, cwd=cwd, text=True, capture_output=True)


def test_torchrun_is_launched_once_with_deepspeed_and_cli_overrides(launcher_env, tmp_path):
    config = TRAINING_DIR / "deepspeed_zero2.json"
    env = launcher_env | {"NPROC_PER_NODE": "8", "DEEPSPEED_CONFIG": str(config)}
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, "--max-steps", "2")
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert argv[:6] == ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0"]
    assert argv[6] == str(TRAINING_DIR / "training.py")
    args = training.parse_args(argv[7:])
    assert args.deepspeed == config
    assert args.max_steps == 2
    assert "--chat-template" not in argv
    assert args.chat_template is None


@pytest.mark.parametrize("value", [None, "", "/path with spaces/custom.jinja"])
def test_launcher_only_passes_optional_template_when_set(launcher_env, tmp_path, value):
    env = launcher_env.copy()
    if value is not None:
        env["CHAT_TEMPLATE"] = value
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert ("--chat-template" in argv) == bool(value)
    args = training.parse_args(argv[1:])
    assert args.chat_template == (Path(value) if value else None)


def test_cli_template_override_wins_over_environment(launcher_env, tmp_path):
    env = launcher_env | {"CHAT_TEMPLATE": "/env/template.jinja"}
    result = invoke(TRAINING_DIR / "run_training.sh", env, tmp_path, "--chat-template", "/cli/template.jinja")
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert training.parse_args(argv[1:]).chat_template == Path("/cli/template.jinja")


@pytest.mark.parametrize("value", ["0", "-1", "eight"])
def test_invalid_worker_count_is_rejected(launcher_env, tmp_path, value):
    result = invoke(TRAINING_DIR / "run_training.sh", launcher_env | {"NPROC_PER_NODE": value}, tmp_path)
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_nested_torchrun_is_rejected(launcher_env, tmp_path):
    result = invoke(TRAINING_DIR / "run_training.sh", launcher_env | {"NPROC_PER_NODE": "8", "LOCAL_RANK": "0"}, tmp_path)
    assert result.returncode == 2
    assert "nested torchrun" in result.stderr


def test_slurm_spooled_script_launches_one_task_eight_workers(launcher_env, tmp_path):
    # sbatch executes a copied script, not the original file in the repository.
    spooled = tmp_path / "slurm_script"
    spooled.write_text((TRAINING_DIR / "submit_training.sbatch").read_text())
    env = launcher_env | {
        "SLURM_JOB_ID": "1234", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
        "SLURM_SUBMIT_DIR": str(TRAINING_DIR.parents[1]),
        "CUDA_VISIBLE_DEVICES": "7,6,5,4,3,2,1,0",
    }
    result = invoke(spooled, env, tmp_path, "--max-steps", "2")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    argv = payload["argv"]
    assert argv.count("torch.distributed.run") == 1
    assert "--nproc-per-node=8" in argv
    args = training.parse_args(argv[7:])
    assert args.batch_size * args.gradient_accumulation_steps * 8 == 16
    assert args.dtype == "bf16"
    assert args.deepspeed == TRAINING_DIR / "deepspeed_zero2.json"
    assert args.chat_template is None
    assert args.output_dir.name == "qwen-zero2-1234"
    assert payload["env"]["WANDB_NAME"] == "qwen-zero2-1234"
    assert payload["env"]["CUDA_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"]
    srun_args = json.loads(Path(env["SRUN_CAPTURE"]).read_text())
    assert srun_args[:4] == ["--nodes=1", "--ntasks=1", "--gpu-bind=none", "--kill-on-bad-exit=1"]


def test_slurm_keeps_user_run_paths_and_parameters(launcher_env, tmp_path):
    output = tmp_path / "my run"
    env = launcher_env | {
        "SLURM_JOB_ID": "88", "WORKSPACE_DIR": str(TRAINING_DIR.parents[1]),
        "OUTPUT_DIR": str(output), "WANDB_NAME": "my run name",
        "BATCH_SIZE": "2", "GRADIENT_ACCUMULATION_STEPS": "4",
    }
    result = invoke(TRAINING_DIR / "submit_training.sbatch", env, tmp_path)
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    args = training.parse_args(captured["argv"][7:])
    assert args.output_dir == output
    assert args.batch_size == 2 and args.gradient_accumulation_steps == 4
    assert captured["env"]["WANDB_NAME"] == "my run name"


@pytest.mark.parametrize("overrides,expected", [
    ({}, "Submit with sbatch"),
    ({"SLURM_JOB_ID": "1", "SLURM_NTASKS": "8"}, "one Slurm task"),
    ({"SLURM_JOB_ID": "1", "SLURM_JOB_NUM_NODES": "2"}, "one node"),
])
def test_slurm_rejects_wrong_execution_context(launcher_env, tmp_path, overrides, expected):
    result = invoke(TRAINING_DIR / "submit_training.sbatch", launcher_env | overrides, tmp_path)
    assert result.returncode == 2
    assert expected in result.stderr


def test_zero2_config_has_auto_batch_and_no_offload():
    config = json.loads((TRAINING_DIR / "deepspeed_zero2.json").read_text())
    assert config["zero_optimization"]["stage"] == 2
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]
    for key in ["train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps", "gradient_clipping"]:
        assert config[key] == "auto"
    assert config["bf16"]["enabled"] == "auto"


def test_failed_preflight_never_starts_srun(launcher_env, tmp_path):
    env = launcher_env | {
        "SLURM_JOB_ID": "1", "WORKSPACE_DIR": str(TRAINING_DIR.parents[1]),
        "PYTHON_BIN": "/bin/false",
    }
    result = invoke(TRAINING_DIR / "submit_training.sbatch", env, tmp_path)
    assert result.returncode != 0
    assert not Path(env["SRUN_CAPTURE"]).exists()


def test_config_forwards_deepspeed_without_initializing_it(monkeypatch):
    import trl

    path = TRAINING_DIR / "deepspeed_zero2.json"
    args = training.parse_args(["--deepspeed", str(path), "--gradient-accumulation-steps", "2"])
    monkeypatch.setattr(trl, "SFTConfig", lambda **kwargs: kwargs)
    config = training.build_training_config(args)
    assert config["deepspeed"] == str(path)
    assert config["gradient_accumulation_steps"] == 2
    assert config["model_init_kwargs"]["device_map"] is None


def test_missing_deepspeed_config_fails_before_training(tmp_path):
    with pytest.raises(SystemExit):
        training.parse_args(["--deepspeed", str(tmp_path / "missing.json")])


@pytest.mark.parametrize("rank", [0, 1])
def test_output_errors_are_shared_with_every_rank(tmp_path, monkeypatch, rank):
    import accelerate.utils

    (tmp_path / "existing").write_text("keep")
    args = training.parse_args(["--output-dir", str(tmp_path)])
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
    args = training.parse_args(["--output-dir", str(tmp_path)])
    training.check_distributed_output_directory(args, SimpleNamespace(world_size=2, process_index=1))
