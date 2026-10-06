"""Tests candidate-first Dolci sampling in the B and D scripts."""

import copy
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

import datasets
import numpy as np
import pytest

from sm_dp import schema, storage

PREPARATION_DIR = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Loads a standalone script without changing the import path."""
    spec = importlib.util.spec_from_file_location(
        name, PREPARATION_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPTS = [
    _load_script("dp_b_reasoning"),
    _load_script("dp_d_reasoning_tools"),
]


@pytest.fixture(params=SCRIPTS, ids=lambda module: module.__name__)
def script(request: pytest.FixtureRequest) -> Any:
    """Runs each sampling test against both scripts."""
    return request.param


def _row(
    sample_id: str = "example",
    *,
    content: str = "<think>Plan.</think>Answer.",
    category: str = "math",
) -> dict[str, Any]:
    """Builds a raw Dolci conversation."""
    return {
        "id": sample_id,
        "dataset_source": category,
        "messages": [
            {"role": "user", "content": "Question."},
            {"role": "assistant", "content": content},
        ],
    }


def test_sampling_stops_at_quota_and_preserves_original_indices(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [_row(str(index)) for index in range(10)]
    order = np.random.default_rng(42).permutation(len(rows)).tolist()
    rows[order[0]]["messages"][0]["content"] = ""
    rows[order[1]]["messages"][1]["content"] = "<think>Plan.</think>"
    original = copy.deepcopy(rows)
    raw = mock.MagicMock()
    raw.__len__.return_value = len(rows)
    raw.__getitem__.side_effect = rows.__getitem__
    raw.filter.side_effect = AssertionError("Do not filter the full dataset.")
    raw.map.side_effect = AssertionError("Do not map the full dataset.")
    loader = mock.Mock(return_value=raw)
    converter = mock.Mock(wraps=script.convert_dolci_sample)
    monkeypatch.setattr(script, "load_dataset", loader)
    monkeypatch.setattr(script, "convert_dolci_sample", converter)

    selected = script.sample_dolci(Path("unused"), size=3, seed=42)

    loader.assert_called_once_with("unused", split="train")
    visited = [call.args[0] for call in raw.__getitem__.call_args_list]
    assert visited == order[:5]
    assert len(set(visited)) == len(visited)
    assert [call.kwargs["index"] for call in converter.call_args_list] == (
        visited
    )
    assert all(type(index) is int for index in visited)
    assert selected["id"] == [f"dolci:{index}" for index in order[2:5]]
    assert selected.features == schema.CONVERSATION_FEATURES
    assert selected[0]["messages"][1]["reasoning"] == "Plan."
    assert selected[0]["messages"][1]["content"] == "Answer."
    assert rows == original
    output = capsys.readouterr().out
    assert "Dolci: selected 3; rejected 2" in output
    assert "Dolci sources:" in output
    assert "'math': 3" in output


@pytest.mark.parametrize(
    "invalid_case",
    [
        "empty_user", "empty_assistant", "reasoning_only", "unclosed_think",
        "unknown_role", "missing_id", "missing_category", "incomplete_turn",
    ],
)
def test_sampling_skips_invalid_conversations_and_refills(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    invalid_case: str, capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = _row("invalid")
    if invalid_case == "empty_user":
        invalid["messages"][0]["content"] = ""
    elif invalid_case == "empty_assistant":
        invalid["messages"][1]["content"] = ""
    elif invalid_case == "reasoning_only":
        invalid["messages"][1]["content"] = "<think>Plan.</think>"
    elif invalid_case == "unclosed_think":
        invalid["messages"][1]["content"] = "<think>Unclosed"
    elif invalid_case == "unknown_role":
        invalid["messages"][1]["role"] = "unknown"
    elif invalid_case == "missing_id":
        del invalid["id"]
    elif invalid_case == "missing_category":
        del invalid["dataset_source"]
    elif invalid_case == "incomplete_turn":
        invalid["messages"].extend([
            {"role": "user", "content": "Next question."},
            {"role": "assistant", "content": "<think>Next plan.</think>"},
        ])
    rows = [invalid, _row("valid")]
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: rows)

    # Seed 1 visits the invalid row first, so the quota needs a replacement.
    assert np.random.default_rng(1).permutation(2).tolist() == [0, 1]
    selected = script.sample_dolci(Path("unused"), size=1, seed=1)

    assert selected["id"] == ["dolci:valid"]
    assert "selected 1; rejected 1" in capsys.readouterr().out


def test_sampling_keeps_and_encodes_reasoning_with_tool_calls(
    script: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row(content="<think>Check status.</think>")
    row["messages"][1]["tool_calls"] = [
        {"name": "status", "arguments": {"target": "rtl"}},
    ]
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: [row])

    selected = script.sample_dolci(Path("unused"), size=1, seed=42)

    call = selected[0]["messages"][1]["tool_calls"][0]
    assert isinstance(call["arguments"], str)
    decoded = storage.decode_sample(selected[0])
    assert decoded["messages"][1]["content"] == ""
    assert decoded["messages"][1]["reasoning"] == "Check status."
    assert decoded["messages"][1]["tool_calls"][0]["arguments"] == {
        "target": "rtl",
    }


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_quota_is_rejected_before_loading(
    script: Any, monkeypatch: pytest.MonkeyPatch, size: int,
) -> None:
    loader = mock.Mock(side_effect=AssertionError("Must not load rows."))
    monkeypatch.setattr(script, "load_dataset", loader)

    with pytest.raises(ValueError, match="sample size must be positive"):
        script.sample_dolci(Path("unused"), size=size, seed=42)

    loader.assert_not_called()


@pytest.mark.parametrize("row_count", [0, 2])
def test_sampling_fails_when_too_few_valid_rows_exist(
    script: Any, monkeypatch: pytest.MonkeyPatch, row_count: int,
) -> None:
    rows = [_row("valid"), _row("invalid", content="<think>Plan.</think>")]
    rows = rows[:row_count]
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: rows)

    with pytest.raises(ValueError, match=f"found {int(row_count > 0)}"):
        script.sample_dolci(Path("unused"), size=2, seed=42)


@pytest.mark.parametrize("error_type", [KeyError, RuntimeError])
def test_unexpected_converter_errors_propagate(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    monkeypatch.setattr(
        script, "load_dataset", lambda *args, **kwargs: [_row()],
    )
    converter = mock.Mock(side_effect=error_type("Unexpected failure."))
    monkeypatch.setattr(script, "convert_dolci_sample", converter)

    with pytest.raises(error_type, match="Unexpected failure"):
        script.sample_dolci(Path("unused"), size=1, seed=42)


def test_storage_errors_are_not_skipped(
    script: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        script, "load_dataset", lambda *args, **kwargs: [_row()],
    )
    encoder = mock.Mock(side_effect=ValueError("Storage failure."))
    monkeypatch.setattr(script.storage, "encode_sample", encoder)

    with pytest.raises(ValueError, match="Storage failure"):
        script.sample_dolci(Path("unused"), size=1, seed=42)


def test_other_sources_keep_their_original_conversion_path(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = datasets.Dataset.from_list([_row(content="<think>Plan.</think>")])
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: raw)

    converted = script.load_and_convert(
        Path("unused"), converter=script.convert_smoltalk_sample,
        category="chat",
    )

    assert len(converted) == 1
    assert converted[0]["messages"][1]["content"] == "<think>Plan.</think>"
    assert "Dolci:" not in capsys.readouterr().out


def test_same_seed_gives_b_and_d_the_same_valid_2250_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(f"{category}-{index}", category=category)
        for category, count in [("math", 1617), ("code", 462), ("chat", 231)]
        for index in range(count)
    ]
    rows.extend(
        _row(f"invalid-{index}", content="<think>Plan.</think>")
        for index in range(3)
    )
    raw = datasets.Dataset.from_list(rows)
    expected_indices = np.random.default_rng(42).permutation(len(rows))
    expected_ids = [
        f"dolci:{rows[index]['id']}"
        for index in expected_indices
        if not rows[index]["id"].startswith("invalid-")
    ][:2250]
    results = []
    for module in SCRIPTS:
        monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: raw)
        selected = module.sample_dolci(Path("unused"), size=2250, seed=42)
        assert len(selected) == len(set(selected["id"])) == 2250
        assert selected["id"] == expected_ids
        assert selected.features == schema.CONVERSATION_FEATURES
        assert sum(Counter(selected["category"]).values()) == 2250
        results.append(selected["id"])

    assert results[0] == results[1]
