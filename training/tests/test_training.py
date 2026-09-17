"""Offline tests: no downloaded models, GPU, or real training artifact required.

Run from workspace root:
uv run --project scripts/training/envs/tp1 --locked --with pytest python -m pytest scripts/training/tests -q
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from trl.chat_template_utils import qwen3_5_think_chat_template

TRAINING_DIR = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("training", TRAINING_DIR / "training.py")
training = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(training)


@pytest.fixture
def sample():
    return {
        "id": "sample-1",
        "source": "siliconmind",
        "messages": [
            {"role": "system", "content": "SYSTEM_SENTINEL", "reasoning": None},
            {"role": "user", "content": "USER_SENTINEL", "reasoning": None},
            {"role": "assistant", "content": "ANSWER_ONE", "reasoning": "REASON_ONE"},
            {"role": "user", "content": "USER_SECOND", "reasoning": None},
            {"role": "assistant", "content": "ANSWER_TWO", "reasoning": "REASON_TWO"},
        ],
    }


@pytest.fixture
def tokenizer():
    # Byte-level tokens keep offsets reversible so assistant-mask tests exercise
    # the real Transformers/Jinja path, without accessing the Hugging Face Hub.
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
        chat_template=qwen3_5_think_chat_template,
    )


def test_adapter_renames_reasoning_without_mutating_source(sample):
    original = copy.deepcopy(sample)
    result = training.adapt_sample(sample)
    assert sample == original
    assert result["messages"][2]["reasoning_content"] == "REASON_ONE"
    assert result["messages"][2]["content"] == "ANSWER_ONE"
    assert all("reasoning" not in message for message in result["messages"])
    assert "reasoning_content" not in result["messages"][0]


@pytest.mark.parametrize("reasoning", [None, ""])
def test_absent_reasoning_renders_empty_think_block(tokenizer, reasoning):
    sample = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "reasoning": reasoning},
    ]}
    template = training.resolve_training_template(tokenizer)
    result = training.preview_sample(tokenizer, sample, template, 256)
    assert "<think>\n\n</think>\n\nHello<|im_end|>" in result["loss_preview"]
    assert "None" not in result["loss_preview"]


def test_training_template_preserves_all_reasoning_and_masks_prompts(tokenizer, sample):
    template = training.resolve_training_template(tokenizer)
    result = training.preview_sample(tokenizer, sample, template, 4096)
    assert tokenizer.chat_template == qwen3_5_think_chat_template  # inference template unchanged
    assert template != tokenizer.chat_template
    for text in ["REASON_ONE", "REASON_TWO", "ANSWER_ONE", "ANSWER_TWO"]:
        assert text in result["loss_preview"]
    for text in ["SYSTEM_SENTINEL", "USER_SENTINEL", "USER_SECOND", "<|im_start|>"]:
        assert text not in result["loss_preview"]
    assert result["loss_preview"].count("<|im_end|>") == 2
    assert 0 < result["loss_tokens"] < result["tokens"]
    assert result["truncated"] is False


def test_truncation_can_remove_all_loss_tokens(tokenizer, sample):
    template = training.resolve_training_template(tokenizer)
    result = training.preview_sample(tokenizer, sample, template, 4)
    assert result["truncated"] is True
    assert result["retained_tokens"] == 4
    assert result["loss_tokens"] == 0


def test_unsupported_template_fails_explicitly(tokenizer, tmp_path):
    path = tmp_path / "unsupported.jinja"
    path.write_text("{% for message in messages %}{{ message.content }}{% endfor %}")
    with pytest.raises(ValueError, match="not training-compatible"):
        training.resolve_training_template(tokenizer, path)


def test_cli_defaults_to_tokenizer_template():
    assert training.parse_args([]).chat_template is None


@pytest.mark.parametrize("native", [None, ""])
def test_missing_native_template_has_actionable_error(tokenizer, native):
    tokenizer.chat_template = native
    with pytest.raises(ValueError, match="--chat-template"):
        training.resolve_training_template(tokenizer)


def test_local_template_override_is_used(tokenizer, tmp_path):
    path = tmp_path / "custom template.jinja"
    path.write_text(qwen3_5_think_chat_template)
    tokenizer.chat_template = "unsupported native template"
    template = training.resolve_training_template(tokenizer, path)
    assert tokenizer.chat_template == qwen3_5_think_chat_template
    assert template != tokenizer.chat_template


def test_missing_local_template_override_is_not_ignored(tokenizer, tmp_path):
    with pytest.raises(FileNotFoundError):
        training.resolve_training_template(tokenizer, tmp_path / "missing.jinja")


@pytest.mark.parametrize("contents", ["", " \n\t"])
def test_empty_override_does_not_fall_back_to_native_template(tokenizer, tmp_path, contents):
    path = tmp_path / "empty.jinja"
    path.write_text(contents)
    with pytest.raises(ValueError, match="--chat-template"):
        training.resolve_training_template(tokenizer, path)


def test_unsupported_native_template_fails_explicitly(tokenizer):
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    with pytest.raises(ValueError, match="not training-compatible"):
        training.resolve_training_template(tokenizer)


def test_already_training_compatible_template_is_kept(tokenizer):
    template = training.resolve_training_template(tokenizer)
    tokenizer.chat_template = template
    assert training.resolve_training_template(tokenizer) == template
    assert tokenizer.chat_template == template


def test_load_saved_dataset_preserves_provenance_and_limits_rows(sample, tmp_path):
    Dataset.from_list([sample, sample]).save_to_disk(str(tmp_path / "data"))
    loaded = training.load_training_dataset(tmp_path / "data", limit=1)
    assert len(loaded) == 1
    assert loaded[0] == sample


def test_dataset_dict_is_not_silently_treated_as_training_data(sample, tmp_path):
    DatasetDict(train=Dataset.from_list([sample])).save_to_disk(str(tmp_path / "data"))
    with pytest.raises(ValueError, match="single saved Dataset"):
        training.load_training_dataset(tmp_path / "data")


def test_offline_model_resolution_uses_only_cached_snapshot(monkeypatch, tmp_path):
    import transformers.utils.hub

    calls = []

    def cached_config(model, filename, **kwargs):
        calls.append((model, filename, kwargs))
        return str(tmp_path / "config.json")

    monkeypatch.setattr(transformers.utils.hub, "cached_file", cached_config)
    assert training.resolve_model_path("Qwen/example", True) == str(tmp_path)
    assert calls == [("Qwen/example", "config.json", {"local_files_only": True})]
    assert training.resolve_model_path(str(tmp_path), True) == str(tmp_path)
    assert training.resolve_model_path("Qwen/example", False) == "Qwen/example"
    assert len(calls) == 1


@pytest.mark.parametrize("option,value", [("--batch-size", "0"), ("--max-length", "-2"), ("--max-steps", "0")])
def test_invalid_cli_values_are_rejected(option, value):
    with pytest.raises(SystemExit):
        training.parse_args([option, value])


def test_nonempty_output_requires_explicit_resume(tmp_path):
    args = training.parse_args(["--output-dir", str(tmp_path)])
    (tmp_path / "existing.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="nonempty"):
        training.check_output_directory(args)
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    with pytest.raises(ValueError, match="Trainer checkpoint"):
        training.check_output_directory(args)
    args.resume_from_checkpoint.mkdir()
    (args.resume_from_checkpoint / "trainer_state.json").write_text("{}")
    training.check_output_directory(args)


@pytest.fixture
def single_process_env(tmp_path):
    """Use the public template, never a developer's private .env.sh."""
    template = (TRAINING_DIR / "envs/tp1/.env.example.sh").read_text()
    keys = {line.split("=", 1)[0].removeprefix("export ")
            for line in template.splitlines() if line.startswith("export ")}
    env = {key: value for key, value in os.environ.items()
           if key not in keys and not key.startswith(("SLURM_", "WANDB_"))
           and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}}
    config = tmp_path / "training config.sh"
    config.write_text(template)
    return env | {"TRAINING_ENV_FILE": str(config), "NPROC_PER_NODE": "1", "DEEPSPEED_CONFIG": ""}


def test_launcher_preserves_spaces_and_cli_overrides(tmp_path, single_process_env):
    # Capture argv as JSON using a stand-in interpreter; no training is started.
    interpreter = tmp_path / "capture args"
    interpreter.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o755)
    env = single_process_env | {"PYTHON_BIN": str(interpreter), "MODEL": "model with spaces", "MAX_STEPS": "7"}
    result = subprocess.run(
        ["bash", str(TRAINING_DIR / "run_training.sh"), "--max-steps", "2", "--dry-run"],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    forwarded = json.loads(result.stdout)
    assert forwarded[0] == str(TRAINING_DIR / "training.py")
    args = training.parse_args(forwarded[1:])
    assert args.model == "model with spaces"
    assert args.max_steps == 2
    assert args.dry_run


@pytest.mark.parametrize("overrides,extra_args,expected_report", [
    ({}, [], "none"),
    ({"REPORT_TO": "wandb"}, [], "wandb"),
    ({
        "REPORT_TO": "wandb",
        "WANDB_ENTITY": "another-team",
        "WANDB_PROJECT": "another-project",
        "WANDB_NAME": "run with spaces",
        "WANDB_LOG_MODEL": "checkpoint",
    }, [], "wandb"),
    ({"REPORT_TO": "wandb"}, ["--report-to", "none"], "none"),
])
def test_launcher_exports_wandb_settings(tmp_path, single_process_env, overrides, extra_args, expected_report):
    """Inspect only selected settings in a stand-in process; never contact W&B."""
    defaults = {
        "WANDB_ENTITY": "s96006730-siliconmind",
        "WANDB_PROJECT": "sm-dp-training",
        "WANDB_NAME": "qwen-zero2-local",
        "WANDB_LOG_MODEL": "false",
    }
    interpreter = tmp_path / "capture wandb settings"
    interpreter.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"settings = {{key: os.environ.get(key) for key in {list(defaults)!r}}}\n"
        "print(json.dumps({'argv': sys.argv[2:], 'settings': settings}))\n"
    )
    interpreter.chmod(0o755)
    env = single_process_env.copy()
    env.update(overrides)
    env["PYTHON_BIN"] = str(interpreter)
    result = subprocess.run(
        ["bash", str(TRAINING_DIR / "run_training.sh"), *extra_args],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    captured = json.loads(result.stdout)
    assert captured["settings"] == {key: overrides.get(key, value) for key, value in defaults.items()}
    assert training.parse_args(captured["argv"]).report_to == expected_report


def test_dry_run_never_constructs_trainer(sample, tokenizer, tmp_path, monkeypatch):
    import transformers
    import trl

    Dataset.from_list([sample]).save_to_disk(str(tmp_path / "data"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not instantiate the trainer or load model weights")

    monkeypatch.setattr(trl, "SFTTrainer", forbidden)
    monkeypatch.setattr(training, "build_training_config", forbidden)
    training.main([
        "--dataset", str(tmp_path / "data"), "--output-dir", str(tmp_path / "output"),
        "--dry-run",
        "--deepspeed", str(TRAINING_DIR / "deepspeed_zero2.json"),
    ])
    assert not (tmp_path / "output").exists()


def test_one_cpu_training_step_and_save(sample, tokenizer, tmp_path, monkeypatch):
    """Exercise real TRL preprocessing, labels, backward pass, and main's saves."""
    import torch
    import transformers
    import trl
    import trl.trainer.sft_trainer as trainer_module

    torch.set_num_threads(1)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_positions=256, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=False,
    ))
    Dataset.from_list([sample, sample]).save_to_disk(str(tmp_path / "data"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(trainer_module, "create_model_from_path", lambda *a, **kw: model)
    real_config = trl.SFTConfig
    monkeypatch.setattr(trl, "SFTConfig", lambda **kw: real_config(use_cpu=True, **kw))
    output = tmp_path / "output"
    training.main([
        "--model", str(tmp_path), "--dataset", str(tmp_path / "data"),
        "--output-dir", str(output),
        "--max-steps", "1", "--max-length", "256", "--dtype", "fp32",
        "--gradient-accumulation-steps", "1", "--logging-steps", "1",
        "--save-steps", "1", "--no-gradient-checkpointing", "--local-files-only",
    ])
    assert (output / "final/model.safetensors").exists()
    assert (output / "final/tokenizer.json").exists()
    assert (output / "checkpoint-1/trainer_state.json").exists()
    assert (output / "training_chat_template.jinja").exists()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    metrics = json.loads((output / "train_results.json").read_text())
    assert metrics["train_loss"] > 0
    metadata = json.loads((output / "run_arguments.json").read_text())
    assert metadata["world_size"] == 1
    assert metadata["global_batch_size"] == 1
    assert (output / "final/chat_template.jinja").read_text() == qwen3_5_think_chat_template
