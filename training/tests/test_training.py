"""Offline tests: no downloaded models, GPU, or real training artifact required.

Run from workspace root:
uv run --project scripts/training/envs/tp1 --locked --with pytest python -m pytest scripts/training/tests -q
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from trl.chat_template_utils import qwen3_5_think_chat_template

TRAINING_DIR = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("training", TRAINING_DIR / "training.py")
training = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(training)


def parse(args=()):
    return training.parse_args(["--model", "test-model", "--dataset", "/test/data",
                                "--output-dir", "/test/output", "--report-to", "none", *args])


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


@pytest.mark.parametrize("native", [None, ""])
def test_missing_native_template_has_actionable_error(tokenizer, native):
    tokenizer.chat_template = native
    with pytest.raises(ValueError, match="supported tokenizer"):
        training.resolve_training_template(tokenizer)


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
        parse([option, value])


@pytest.mark.parametrize("nested", [False, True])
def test_disable_training_cache_targets_text_config(nested):
    from types import SimpleNamespace
    from transformers import GPT2Config, Qwen3_5Config

    config = Qwen3_5Config() if nested else GPT2Config()
    assert config.get_text_config().use_cache is True
    training.disable_training_cache(SimpleNamespace(config=config))
    assert config.get_text_config().use_cache is False
    if nested:
        assert not hasattr(config, "use_cache")  # Do not invent a top-level setting.


@pytest.mark.parametrize("checkpointing", [True, False])
def test_qwen35_loads_with_training_kwargs_without_constructor_cache_arg(tmp_path, monkeypatch, checkpointing):
    """Exercise real TRL/Transformers loading on a tiny local Qwen3.5 checkpoint."""
    from types import SimpleNamespace
    import transformers
    import trl
    from trl.trainer.utils import create_model_from_path

    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "layer_types": ["full_attention"],
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "out_hidden_size": 16, "num_position_embeddings": 16,
        },
    )
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(tmp_path)
    monkeypatch.setattr(trl, "SFTConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(training.RECIPE, "dtype", "fp32")
    monkeypatch.setitem(training.RECIPE, "gradient_checkpointing", checkpointing)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    args = parse()
    settings = training.build_training_config(args)
    # No loader mocking: the previous use_cache kwarg must fail here.
    model = create_model_from_path(str(tmp_path), **settings.model_init_kwargs)
    assert "use_cache" not in settings.model_init_kwargs
    assert settings.gradient_checkpointing is checkpointing
    training.disable_training_cache(model)
    assert model.config.text_config.use_cache is False
    assert model.model.language_model.config.use_cache is False


def test_nonempty_output_requires_explicit_resume(tmp_path):
    args = parse(["--output-dir", str(tmp_path)])
    (tmp_path / "existing.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="nonempty"):
        training.check_output_directory(args)
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    with pytest.raises(ValueError, match="Trainer checkpoint"):
        training.check_output_directory(args)
    args.resume_from_checkpoint.mkdir()
    (args.resume_from_checkpoint / "trainer_state.json").write_text("{}")
    training.check_output_directory(args)


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
        "--model", str(tmp_path), "--dataset", str(tmp_path / "data"),
        "--output-dir", str(tmp_path / "output"), "--report-to", "none", "--dry-run",
    ])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("smoke", [False, True])
def test_cpu_training_save_and_smoke_resume(sample, tokenizer, tmp_path, monkeypatch, smoke):
    """Exercise real TRL preprocessing, labels, backward pass, and main's saves."""
    import torch
    import transformers
    import trl
    import trl.trainer.sft_trainer as trainer_module

    torch.set_num_threads(1)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_positions=256, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    ))
    Dataset.from_list([sample, sample]).save_to_disk(str(tmp_path / "data"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(trainer_module, "create_model_from_path", lambda *a, **kw: model)
    real_config = trl.SFTConfig
    monkeypatch.setattr(trl, "SFTConfig", lambda **kw: real_config(use_cpu=True, **kw))
    output = tmp_path / "output"
    # Test-only tiny CPU recipe; the public API remains fixed to BF16/ZeRO-2.
    for key, value in dict(dtype="fp32", deepspeed=None, max_steps=1, max_length=256,
                           gradient_accumulation_steps=1, logging_steps=1,
                           save_steps=1, gradient_checkpointing=False).items():
        monkeypatch.setitem(training.RECIPE, key, value)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    argv = [
        "--model", str(tmp_path), "--dataset", str(tmp_path / "data"),
        "--output-dir", str(output), "--report-to", "none",
        *(["--smoke"] if smoke else []),
    ]
    training.main(argv)
    assert (output / "final/model.safetensors").exists()
    assert model.config.use_cache is False
    assert (output / "final/tokenizer.json").exists()
    assert (output / "checkpoint-1/trainer_state.json").exists()
    assert (output / "training_chat_template.jinja").exists()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == (2 if smoke else 1)
    metrics = json.loads((output / "train_results.json").read_text())
    assert metrics["train_loss"] > 0
    metadata = json.loads((output / "run_arguments.json").read_text())
    assert metadata["world_size"] == 1
    assert metadata["global_batch_size"] == 1
    assert (output / "final/chat_template.jinja").read_text() == qwen3_5_think_chat_template
    if smoke:
        original_arguments = (output / "run_arguments.json").read_text()
        training.main([*argv, "--resume-from-checkpoint", str(output / "checkpoint-1")])
        assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 2
        assert (output / "resume_arguments.json").exists()
        assert (output / "run_arguments.json").read_text() == original_arguments


@pytest.mark.parametrize("option", ["--model", "--dataset", "--output-dir", "--report-to"])
@pytest.mark.parametrize("missing", [True, False])
def test_required_inputs_have_no_fallback(option, missing):
    argv = ["--model", "model", "--dataset", "/data", "--output-dir", "/output", "--report-to", "none"]
    index = argv.index(option)
    if missing:
        del argv[index:index + 2]
    else:
        argv[index + 1] = ""
    with pytest.raises(SystemExit):
        training.parse_args(argv)


def test_fixed_recipe_ignores_stale_training_environment(monkeypatch):
    for key in ["EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "MAX_STEPS", "MAX_TRAIN_SAMPLES", "DTYPE"]:
        monkeypatch.setenv(key, "999")
    args = parse()
    assert args.epochs == 1 and args.learning_rate == 2e-5
    assert args.batch_size == 1 and args.gradient_accumulation_steps == 2
    assert args.dtype == "bf16" and args.max_steps == -1
    assert args.max_train_samples is None
    smoke = parse(["--smoke"])
    assert (smoke.max_steps, smoke.max_train_samples, smoke.logging_steps, smoke.save_steps) == (2, 32, 1, 1)
    with pytest.raises(SystemExit):
        parse(["--smoke", "--dry-run"])


@pytest.mark.parametrize("key", ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME"])
def test_wandb_identity_required_only_when_enabled(monkeypatch, key):
    for name in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME"]:
        monkeypatch.setenv(name, "test")
    monkeypatch.delenv(key)
    assert parse().report_to == "none"
    with pytest.raises(SystemExit):
        parse(["--report-to", "wandb"])
