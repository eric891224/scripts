"""Shared offline shell environment: manually source the public TP1 template."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def exported_training_env(tmp_path):
    template = Path(__file__).resolve().parents[1] / "envs/tp1/.env.example.sh"
    keys = {line.split("=", 1)[0].removeprefix("export ")
            for line in template.read_text().splitlines() if line.startswith("export ")}
    env = {key: value for key, value in os.environ.items()
           if key not in keys and not key.startswith(("SLURM_", "WANDB_"))
           and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "WORKSPACE_DIR", "TRAINING_ENV_FILE",
                          "CUDA_VISIBLE_DEVICES", "SBATCH_EXPORT", "SLURM_EXPORT_ENV", "NPROC_PER_NODE"}}
    # Simulate the site's inherited compiler choice. Only manual sourcing may change it.
    env.update({"CC": "/cluster/nvc", "CXX": "/cluster/nvc++"})
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; "$2" -c "import json, os; print(json.dumps(dict(os.environ)))"',
         "source-test", str(template), sys.executable],
        cwd=tmp_path, env=env, check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)
