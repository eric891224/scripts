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


@pytest.fixture
def tool_sample() -> dict:
    """Returns a saved tool conversation with a call and a response."""
    return {
        "schema_version": "2.0",
        "tools": [
            {
                "name": "lookup_weather",
                "description": "DEFINITION_SENTINEL",
                "parameters": json.dumps({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                }),
            },
        ],
        "messages": [
            {
                "role": "system",
                "content": "SYSTEM_SENTINEL",
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": "USER_SENTINEL",
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning": "REASON_SENTINEL",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "name": "lookup_weather",
                        "arguments": json.dumps({"city": "Taipei"}),
                    },
                ],
            },
            {
                "role": "tool",
                "content": "RESULT_SENTINEL",
                "tool_calls": [],
                "tool_call_id": "call_weather",
            },
            {
                "role": "assistant",
                "content": "ANSWER_SENTINEL",
                "reasoning": None,
                "tool_calls": [],
            },
        ],
    }


@pytest.fixture
def mixed_samples(sample: dict, tool_sample: dict) -> list[dict]:
    """Returns compatible saved chat and tool rows."""
    chat = copy.deepcopy(sample)
    chat["tools"] = []
    chat["schema_version"] = "2.0"
    for message in chat["messages"]:
        message["tool_calls"] = []
    tools = copy.deepcopy(tool_sample)
    tools["id"] = "tool-sample-1"
    tools["source"] = "smoltalk"
    return [chat, tools]


def _make_cpu_trainer(dataset, tokenizer, output_dir, max_length=4096):
    """Builds a tiny real SFTTrainer for preprocessing and collator tests."""
    import torch
    import transformers
    import trl

    torch.set_num_threads(1)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=4096,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    return trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=trl.SFTConfig(
            output_dir=str(output_dir),
            use_cpu=True,
            bf16=False,
            assistant_only_loss=True,
            packing=False,
            max_length=max_length,
            report_to="none",
        ),
    )


@pytest.mark.parametrize("num_proc", [None, 2])
def test_tokenization_map_preserves_dynamic_json_and_disk_rows(
    tokenizer,
    tool_sample: dict,
    tmp_path: Path,
    num_proc: int | None,
) -> None:
    arguments = [
        {"city": "Taipei"},
        {"zone": "UTC", "values": [1, None]},
        {},
        {"city": None},
    ]
    rows = []
    for index, payload in enumerate(arguments):
        row = copy.deepcopy(tool_sample)
        row["id"] = f"tool:{index}"
        row["messages"][2]["tool_calls"][0]["arguments"] = json.dumps(payload)
        row["tools"][0]["parameters"] = json.dumps({
            "type": "object",
            "properties": {key: {} for key in payload},
        })
        rows.append(row)
    raw = Dataset.from_list(rows)
    original = raw.to_list()
    template = training.resolve_training_template(tokenizer)

    tokenized = raw.map(
        training.tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer, "template": template},
        num_proc=num_proc,
    )
    tokenized.save_to_disk(str(tmp_path / "tokens"))
    restored = Dataset.load_from_disk(str(tmp_path / "tokens"))

    assert raw.to_list() == original
    assert restored.to_list() == tokenized.to_list()
    for index, row in enumerate(restored):
        for column in raw.column_names:
            assert row[column] == original[index][column]
        stored = row["messages"][2]["tool_calls"][0]["arguments"]
        assert isinstance(stored, str)
        assert json.loads(stored) == arguments[index]
        assert len(row["input_ids"]) == len(row["assistant_masks"])
        assert any(row["assistant_masks"])


def test_preview_uses_the_training_tokenization_path(
    tokenizer,
    tool_sample: dict,
    monkeypatch,
) -> None:
    template = training.resolve_training_template(tokenizer)
    tokenize_sample = training.tokenize_sample
    calls = []

    def track_tokenization(sample, **kwargs):
        encoded = tokenize_sample(sample, **kwargs)
        calls.append(encoded)
        return encoded

    monkeypatch.setattr(training, "tokenize_sample", track_tokenization)
    result = training.preview_sample(tokenizer, tool_sample, template, 8192)

    assert len(calls) == 1
    encoded = calls[0]
    expected = [
        token
        for token, mask in zip(
            encoded["input_ids"][1:],
            encoded["assistant_masks"][1:],
            strict=True,
        )
        if mask
    ]
    assert result["tokens"] == len(encoded["input_ids"])
    assert result["loss_tokens"] == len(expected)
    assert result["loss_preview"] == tokenizer.decode(
        expected, skip_special_tokens=False
    )[:1500]


def test_tokenization_rejects_a_template_without_assistant_mask(
    tokenizer,
    tool_sample: dict,
) -> None:
    template = "{% for message in messages %}{{ message.role }}{% endfor %}"

    with pytest.raises(ValueError, match="no usable assistant loss mask"):
        training.tokenize_sample(
            tool_sample,
            tokenizer=tokenizer,
            template=template,
        )


def test_trainer_builds_and_pads_assistant_labels_without_retokenizing(
    tokenizer,
    mixed_samples: list[dict],
    tmp_path: Path,
) -> None:
    template = training.resolve_training_template(tokenizer)
    tokenized = Dataset.from_list(mixed_samples).map(
        training.tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer, "template": template},
    )

    trainer = _make_cpu_trainer(tokenized, tokenizer, tmp_path)

    assert len(trainer.train_dataset) == 2
    for index, row in enumerate(trainer.train_dataset):
        assert row["input_ids"] == tokenized[index]["input_ids"]
        expected = [
            token if mask else -100
            for token, mask in zip(
                tokenized[index]["input_ids"],
                tokenized[index]["assistant_masks"],
                strict=True,
            )
        ]
        assert row["labels"] == expected
        loss_text = tokenizer.decode(
            [label for label in row["labels"] if label != -100]
        )
        for marker in (
            "USER_SENTINEL",
            "SYSTEM_SENTINEL",
            "DEFINITION_SENTINEL",
            "RESULT_SENTINEL",
        ):
            assert marker not in loss_text
        if index == 1:
            assert "<function=lookup_weather>" in loss_text
            assert "<parameter=city>\nTaipei\n</parameter>" in loss_text
            assert "ANSWER_SENTINEL" in loss_text
        else:
            assert "ANSWER_ONE" in loss_text

    batch = trainer.data_collator(trainer.train_dataset.to_list())
    for index, row in enumerate(trainer.train_dataset):
        length = len(row["input_ids"])
        assert batch["labels"][index, :length].tolist() == row["labels"]
        assert batch["labels"][index, length:].eq(-100).all().item()
        assert batch["attention_mask"][index, :length].eq(1).all().item()
        assert batch["attention_mask"][index, length:].eq(0).all().item()


def test_trainer_truncates_tokens_and_labels_and_drops_masked_rows(
    tokenizer,
    sample: dict,
    tmp_path: Path,
) -> None:
    short = copy.deepcopy(sample)
    short["id"] = "short"
    long = copy.deepcopy(sample)
    long["id"] = "long"
    long["messages"][0]["content"] = "SYSTEM_SENTINEL" * 100
    template = training.resolve_training_template(tokenizer)
    tokenized = Dataset.from_list([short, long]).map(
        training.tokenize_sample,
        fn_kwargs={"tokenizer": tokenizer, "template": template},
    )
    max_length = 128
    preview = training.preview_sample(tokenizer, short, template, max_length)
    assert preview["loss_tokens"] > 0
    assert not any(tokenized[1]["assistant_masks"][:max_length])

    trainer = _make_cpu_trainer(
        tokenized,
        tokenizer,
        tmp_path,
        max_length=max_length,
    )

    assert trainer.train_dataset["id"] == ["short"]
    retained = trainer.train_dataset[0]
    assert retained["input_ids"] == tokenized[0]["input_ids"][:max_length]
    expected = [
        token if mask else -100
        for token, mask in zip(
            tokenized[0]["input_ids"][:max_length],
            tokenized[0]["assistant_masks"][:max_length],
            strict=True,
        )
    ]
    assert retained["labels"] == expected
    assert preview["loss_tokens"] == sum(
        label != -100 for label in retained["labels"][1:]
    )


@pytest.mark.parametrize("stored", [True, False])
def test_tool_adapter_preserves_payloads_and_links_without_aliasing(
    tool_sample: dict,
    stored: bool,
) -> None:
    if not stored:
        tool_sample["tools"][0]["parameters"] = json.loads(
            tool_sample["tools"][0]["parameters"]
        )
        call = tool_sample["messages"][2]["tool_calls"][0]
        call["arguments"] = json.loads(call["arguments"])
    original = copy.deepcopy(tool_sample)

    adapted = training.adapt_sample(tool_sample)

    function = adapted["tools"][0]["function"]
    assert function["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert adapted["tools"][0]["type"] == "function"
    assert adapted["messages"][2]["tool_calls"] == [
        {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": {"city": "Taipei"},
            },
        },
    ]
    assert adapted["messages"][2]["content"] is None
    assert adapted["messages"][2]["reasoning_content"] == "REASON_SENTINEL"
    assert adapted["messages"][3]["tool_call_id"] == "call_weather"
    assert "reasoning_content" not in adapted["messages"][3]
    assert tool_sample == original

    function["parameters"]["properties"]["city"]["type"] = "number"
    arguments = adapted["messages"][2]["tool_calls"][0]["function"]["arguments"]
    arguments["city"] = "Osaka"
    assert tool_sample == original


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"city": None},
        {
            "city": "台北",
            "filters": {"enabled": False, "limit": 3, "missing": None},
            "items": [1, "text", None, {"nested": True}],
        },
    ],
)
def test_tool_adapter_preserves_missing_null_and_nested_arguments(
    tool_sample: dict,
    arguments: dict,
) -> None:
    tool_sample["messages"][2]["tool_calls"][0]["arguments"] = json.dumps(
        arguments, ensure_ascii=False
    )

    adapted = training.adapt_sample(tool_sample)

    assert (
        adapted["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )


@pytest.mark.parametrize("field", ["arguments", "parameters"])
@pytest.mark.parametrize("payload", ["invalid json", "[]", "null", "42"])
def test_tool_adapter_rejects_invalid_payloads(
    tool_sample: dict,
    field: str,
    payload: str,
) -> None:
    if field == "arguments":
        tool_sample["messages"][2]["tool_calls"][0][field] = payload
    else:
        tool_sample["tools"][0][field] = payload

    with pytest.raises(ValueError):
        training.adapt_sample(tool_sample)


def test_tool_adapter_preserves_unspecified_parameters_and_missing_call_id(
    tool_sample: dict,
) -> None:
    tool_sample["tools"][0]["description"] = None
    tool_sample["tools"][0]["parameters"] = None
    call = tool_sample["messages"][2]["tool_calls"][0]
    call["id"] = None
    call["arguments"] = "{}"
    tool_sample["messages"][3]["tool_call_id"] = None

    adapted = training.adapt_sample(tool_sample)

    assert adapted["tools"] == [
        {"type": "function", "function": {"name": "lookup_weather"}},
    ]
    assert "id" not in adapted["messages"][2]["tool_calls"][0]
    assert adapted["messages"][3]["tool_call_id"] is None
    assert (
        adapted["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == {}
    )


@pytest.mark.parametrize("call_only", [True, False])
@pytest.mark.parametrize("reasoning", [None, "REASON_SENTINEL"])
def test_tool_template_renders_definitions_and_masks_assistant_output(
    tokenizer,
    tool_sample: dict,
    call_only: bool,
    reasoning: str | None,
) -> None:
    tool_sample["messages"][2]["reasoning"] = reasoning
    if call_only:
        tool_sample["messages"] = tool_sample["messages"][:3]
    template = training.resolve_training_template(tokenizer)
    adapted = training.adapt_sample(tool_sample)

    encoded = tokenizer.apply_chat_template(
        adapted["messages"],
        tools=adapted["tools"],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    rendered = tokenizer.decode(
        encoded["input_ids"], skip_special_tokens=False
    )
    assert "DEFINITION_SENTINEL" in rendered
    assert '"required": ["city"]' in rendered
    assert '<tool_call>\n<function=lookup_weather>\n' in rendered
    assert '<parameter=city>\nTaipei\n</parameter>' in rendered

    result = training.preview_sample(tokenizer, tool_sample, template, 8192)
    loss_preview = result["loss_preview"]
    for text in (
        "<function=lookup_weather>",
        "<parameter=city>\nTaipei\n</parameter>",
        "</tool_call>",
    ):
        assert text in loss_preview
    if reasoning is None:
        assert "<think>\n\n</think>\n\n<tool_call>" in loss_preview
    else:
        assert reasoning in loss_preview
    for text in (
        "DEFINITION_SENTINEL",
        "SYSTEM_SENTINEL",
        "USER_SENTINEL",
        "RESULT_SENTINEL",
        "<|im_start|>",
        "<tools>",
    ):
        assert text not in loss_preview
    if not call_only:
        assert "RESULT_SENTINEL" in rendered
        assert "ANSWER_SENTINEL" in loss_preview
    assert result["truncated"] is False
    assert 0 < result["loss_tokens"] < result["tokens"]

    masked_ids = [
        token
        for token, mask in zip(
            encoded["input_ids"][1:],
            encoded["assistant_masks"][1:],
            strict=True,
        )
        if mask
    ]
    assert result["loss_tokens"] == len(masked_ids)
    assert loss_preview == tokenizer.decode(
        masked_ids, skip_special_tokens=False
    )[:1500]
    assert tokenizer.chat_template == qwen3_5_think_chat_template


def test_tool_preview_preserves_multiple_calls_in_order(
    tokenizer,
    tool_sample: dict,
) -> None:
    tool_sample["messages"][2]["tool_calls"].append({
        "id": None,
        "name": "lookup_time",
        "arguments": '{"zone": "Asia/Taipei"}',
    })
    tool_sample["tools"].append({
        "name": "lookup_time",
        "description": "SECOND_DEFINITION_SENTINEL",
        "parameters": json.dumps({
            "type": "object",
            "properties": {"zone": {"type": "string"}},
        }),
    })
    template = training.resolve_training_template(tokenizer)

    result = training.preview_sample(tokenizer, tool_sample, template, 8192)

    loss_preview = result["loss_preview"]
    assert loss_preview.index("<function=lookup_weather>") < loss_preview.index(
        "<function=lookup_time>"
    )
    assert "<parameter=zone>\nAsia/Taipei\n</parameter>" in loss_preview
    assert loss_preview.count("</tool_call>") == 2
    assert "SECOND_DEFINITION_SENTINEL" not in loss_preview


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


@pytest.mark.parametrize("sample_name", ["sample", "tool_sample"])
def test_dry_run_never_constructs_trainer(
    request, sample_name, tokenizer, tmp_path, monkeypatch,
):
    import transformers
    import trl

    sample = request.getfixturevalue(sample_name)
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
@pytest.mark.parametrize("dataset_kind", ["chat", "mixed"])
def test_cpu_training_save_and_smoke_resume(
    sample,
    mixed_samples,
    tokenizer,
    tmp_path,
    monkeypatch,
    smoke,
    dataset_kind,
):
    """Exercise real TRL preprocessing, labels, backward pass, and main's saves."""
    import torch
    import transformers
    import trl
    import trl.trainer.sft_trainer as trainer_module

    torch.set_num_threads(1)
    max_length = 4096 if dataset_kind == "mixed" else 256
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(tokenizer), n_positions=max_length, n_embd=16,
        n_layer=1, n_head=2,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    ))
    rows = mixed_samples if dataset_kind == "mixed" else [sample, sample]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(trainer_module, "create_model_from_path", lambda *a, **kw: model)
    real_config = trl.SFTConfig
    monkeypatch.setattr(trl, "SFTConfig", lambda **kw: real_config(use_cpu=True, **kw))
    output = tmp_path / "output"
    # Test-only tiny CPU recipe; the public API remains fixed to BF16/ZeRO-2.
    for key, value in dict(dtype="fp32", deepspeed=None, max_steps=1, max_length=max_length,
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
    assert metadata["input_rows"] == metadata["prepared_rows"] == len(rows)
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
    expected_recipe = dict(training.RECIPE)

    for key in (
        "EPOCHS",
        "LEARNING_RATE",
        "BATCH_SIZE",
        "MAX_STEPS",
        "MAX_TRAIN_SAMPLES",
        "DTYPE",
    ):
        monkeypatch.setenv(key, "999")

    args = parse()

    actual_recipe = {
        name: getattr(args, name)
        for name in expected_recipe
    }
    assert actual_recipe == expected_recipe
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
