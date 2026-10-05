"""Tests Dolci filtering in the standalone B and D preparation scripts."""

import copy
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import pydantic
import pytest

from sm_dp import schema
from sm_dp.mixing import (
    DatasetGroupSpec,
    DatasetSpec,
    MixtureSpec,
    ReplacementMode,
    mix,
)

PREPARATION_DIR = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Loads one standalone script without changing the import path."""
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


@pytest.fixture(params=SCRIPTS, ids=lambda script: script.__name__)
def script(request: pytest.FixtureRequest) -> Any:
    """Runs each filter test against both preparation scripts."""
    return request.param


def _row(
    content: str | None,
    *,
    sample_id: str = "example",
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


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Answer.", True),
        ("<think>Plan.</think>Answer.", True),
        ("  <think>Plan.</think>\n Answer. ", True),
        ("<think></think>Answer.", True),
        ("<think>Plan.</think>", False),
        ("  <think>Plan.</think>\n\t", False),
        ("<think></think>", False),
        ("<think>Unclosed", True),
        ("Quoted <think>Plan.</think>", True),
        ("", True),
        (None, True),
    ],
)
def test_filter_only_rejects_closed_think_blocks_without_answers(
    script: Any, content: str | None, expected: bool,
) -> None:
    raw = _row(content)
    original = copy.deepcopy(raw)

    assert script.has_complete_assistant_responses(raw) is expected
    assert raw == original


def test_filter_keeps_reasoning_with_structured_tool_calls(script: Any) -> None:
    raw = _row("<think>Check status.</think>")
    raw["messages"][1]["tool_calls"] = [
        {"name": "status", "arguments": {}},
    ]

    assert script.has_complete_assistant_responses(raw)
    converted = script.convert_dolci_sample(raw, index=0)
    assert converted["messages"][1]["content"] == ""
    assert converted["messages"][1]["reasoning"] == "Check status."
    assert converted["messages"][1]["tool_calls"][0]["name"] == "status"


def test_filter_rejects_whole_conversation_if_any_assistant_has_no_answer(
    script: Any,
) -> None:
    raw = _row("Answer.")
    raw["messages"].extend([
        {"role": "user", "content": "Next question."},
        {"role": "assistant", "content": "<think>Next plan.</think>"},
    ])

    assert not script.has_complete_assistant_responses(raw)


def test_filter_does_not_inspect_non_assistant_think_blocks(
    script: Any,
) -> None:
    raw = _row("Answer.")
    raw["messages"].insert(
        0, {"role": "system", "content": "<think>System text.</think>"}
    )
    raw["messages"][1]["content"] = "<think>User text.</think>"
    raw["messages"].append(
        {"role": "tool", "content": "<think>Tool text.</think>"}
    )

    assert script.has_complete_assistant_responses(raw)


def test_load_and_convert_filters_dolci_and_reports_counts(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = datasets.Dataset.from_list([
        _row("<think>Plan.</think>Answer.", sample_id="keep"),
        _row("<think>Plan.</think>", sample_id="remove"),
    ])
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: raw)

    converted = script.load_and_convert(
        Path("unused"), converter=script.convert_dolci_sample,
    )

    assert converted["id"] == ["dolci:keep"]
    assert converted[0]["messages"][1]["reasoning"] == "Plan."
    assert converted[0]["messages"][1]["content"] == "Answer."
    assert converted.features == schema.CONVERSATION_FEATURES
    assert "Dolci: kept 1; removed 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("content", "error_type", "match"),
    [
        ("<think>Unclosed", ValueError, "unclosed"),
        ("", pydantic.ValidationError, "content or tool calls"),
    ],
)
def test_other_dolci_format_errors_still_propagate(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    content: str, error_type: type[Exception], match: str,
) -> None:
    raw = datasets.Dataset.from_list([_row(content)])
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: raw)

    with pytest.raises(error_type, match=match):
        script.load_and_convert(
            Path("unused"), converter=script.convert_dolci_sample,
        )


def test_other_sources_are_not_filtered(
    script: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = datasets.Dataset.from_list([_row("<think>Plan.</think>")])
    monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: raw)

    converted = script.load_and_convert(
        Path("unused"), converter=script.convert_smoltalk_sample,
        category="chat",
    )

    assert len(converted) == 1
    assert converted[0]["messages"][1]["content"] == "<think>Plan.</think>"
    assert "Dolci: kept" not in capsys.readouterr().out


def test_b_and_d_keep_the_same_2250_row_cohort_after_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(
            "<think>Plan.</think>Answer.",
            sample_id=f"{category}-{index}",
            category=category,
        )
        for category, count in [("math", 1617), ("code", 462), ("chat", 231)]
        for index in range(count)
    ]
    rows.extend(
        _row("<think>Plan.</think>", sample_id=f"remove-{index}")
        for index in range(3)
    )
    raw = datasets.Dataset.from_list(rows)
    selected_ids = []
    for script in SCRIPTS:
        monkeypatch.setattr(script, "load_dataset", lambda *args, **kwargs: raw)
        dolci = script.load_and_convert(
            Path("unused"), converter=script.convert_dolci_sample,
        )
        counts = Counter(dolci["category"])
        assert counts == {"math": 1617, "code": 462, "chat": 231}
        sources = {
            category: dolci.filter(
                lambda value: value == category, input_columns="category",
            )
            for category in sorted(counts)
        }
        selected = mix(
            datasets=sources,
            spec=MixtureSpec(
                size=2250,
                seed=42,
                replacement_strategy=ReplacementMode.NEVER,
                groups={
                    "reasoning": DatasetGroupSpec(
                        weight=1,
                        datasets=[
                            DatasetSpec(name=category, weight=counts[category])
                            for category in sorted(counts)
                        ],
                    ),
                },
            ),
        )
        assert len(selected) == len(set(selected["id"])) == 2250
        assert Counter(selected["category"]) == {
            "math": 1575, "code": 450, "chat": 225,
        }
        assert not any("remove-" in sample_id for sample_id in selected["id"])
        selected_ids.append(selected["id"])

    assert selected_ids[0] == selected_ids[1]
