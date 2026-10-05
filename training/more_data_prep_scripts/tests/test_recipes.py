"""Tests experiment recipes with small source fixtures."""

import importlib
import json
import pathlib
import sys
from collections import Counter

import datasets
import numpy as np
import pytest

from sm_dp import schema
from sm_dp import storage

PREPARATION_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREPARATION_DIR))
preparation = importlib.import_module("_common")
RECIPES = [
    importlib.import_module(name)
    for name in (
        "dp_a_general",
        "dp_b_reasoning",
        "dp_c_tools",
        "dp_d_reasoning_tools",
    )
]
_FORMAT_SUFFIX = (
    "\n\nThe output MUST strictly adhere to the following format, and NO other "
    "text MUST be included.\n"
    "The example format is as follows. Please make sure the parameter type "
    "is correct. If no function call is needed, please make the tool calls "
    "an empty list '[]'.\n"
    "<tool_call>[\n"
    '{"name": "func_name1", "arguments": {"argument1": "value1", '
    '"argument2": "value2"}},\n'
    "... (more tool calls as required)\n"
    "]</tool_call>"
)


def _chat_row(index: int) -> dict:
    """Builds a source message row."""
    return {
        "messages": [
            {"role": "user", "content": f"Question {index}."},
            {"role": "assistant", "content": "Answer."},
        ],
    }


@pytest.fixture
def small_sources(monkeypatch) -> dict:
    """Supplies raw sources and scales shared pools to a 400-row mixture."""
    rows = [_chat_row(index) for index in range(400)]
    sources = {
        name: datasets.Dataset.from_list(rows)
        for name in preparation.GENERAL_WEIGHTS
    }
    sources["spec2rtl"] = datasets.Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": f"Build module {index}."}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "<think>Design.</think>RTL.",
                },
            ],
        }
        for index in range(400)
    ])
    sources["dolci"] = datasets.Dataset.from_list([
        _chat_row(index) | {
            "id": f"dolci-{index}",
            "dataset_source": "math" if index < 56 else (
                "code" if index < 72 else "chat"
            ),
        }
        for index in range(80)
    ])
    tool = {
        "name": "status",
        "parameters": {"type": "object", "properties": {}},
    }
    system = (
        "Choose a tool.\n\nYou have access to the following tools:\n"
        f"<tools>{json.dumps([tool])}</tools>{_FORMAT_SUFFIX}"
    )
    sources["apigen-80k"] = datasets.Dataset.from_list([
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Get status {index}."},
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>[{"name":"status","arguments":{}}]</tool_call>'
                    ),
                },
            ],
        }
        for index in range(40)
    ])
    by_path = {
        preparation.SOURCES[name][0]: raw for name, raw in sources.items()
    }
    monkeypatch.setattr(preparation, "_load_raw", lambda path: by_path[path])
    monkeypatch.setattr(preparation, "POOL_SIZES", {
        "spec2rtl": 320,
        "smol-magpie-ultra": 40,
        "openhermes-100k": 20,
        "smol-constraints": 12,
        "everyday-conversations": 8,
        "dolci": 20,
        "apigen-80k": 20,
    })
    return sources


def test_recipes_generate_controlled_cohorts(
    small_sources, tmp_path,
) -> None:
    outputs = []
    expected_counts = [
        {"domain": 320, "general": 80},
        {"domain": 320, "general": 60, "reasoning": 20},
        {"domain": 320, "general": 60, "tools": 20},
        {"domain": 320, "general": 40, "reasoning": 20, "tools": 20},
    ]
    for recipe, counts in zip(RECIPES, expected_counts, strict=True):
        assert recipe.SPEC.size == 45_000
        spec = recipe.SPEC.model_copy(update={"size": 400})
        path = tmp_path / recipe.OUTPUT_PATH.name
        preparation.generate(path, spec)
        restored = datasets.load_from_disk(str(path))
        stats = json.loads(path.with_suffix(".stats.json").read_text())
        assert stats["groups"] == counts
        assert stats["repeated_rows"] == 0
        assert len(restored) == 400
        assert restored.features == schema.CONVERSATION_FEATURES
        assert json.loads(path.with_suffix(".mixture.json").read_text()) == (
            spec.model_dump(mode="json")
        )
        rows = restored.to_list()
        outputs.append(rows)
        for row in rows:
            if row["source"] == "siliconmind":
                source_index = int(row["id"].split(":")[-1])
                assert row["messages"][0]["content"] == (
                    f"Build module {source_index}."
                )
            if row["category"] == "apigen-80k":
                decoded = storage.decode_sample(row)
                assert decoded["messages"][2]["content"] is None
                call = decoded["messages"][2]["tool_calls"][0]
                assert call["arguments"] == {}
                assert decoded["tools"][0]["name"] == "status"

    def ids(rows, *, source=None, category=None):
        return {
            row["id"] for row in rows
            if (source is None or row["source"] == source)
            and (category is None or row["category"] == category)
        }

    assert all(
        ids(rows, source="siliconmind") == ids(outputs[0], source="siliconmind")
        for rows in outputs
    )
    assert ids(outputs[1], source="dolci") == ids(outputs[3], source="dolci")
    assert ids(outputs[2], category="apigen-80k") == ids(
        outputs[3], category="apigen-80k"
    )
    for category in preparation.GENERAL_WEIGHTS:
        a, b, c, d = [ids(rows, category=category) for rows in outputs]
        assert b == c
        assert d <= b <= a
    dolci_counts = Counter(
        row["category"] for row in outputs[1] if row["source"] == "dolci"
    )
    assert dolci_counts == {"math": 14, "code": 4, "chat": 2}


def test_source_selection_uses_original_indices_and_exact_strata() -> None:
    raw = datasets.Dataset.from_dict({"value": list(range(20))})
    assert preparation._select_indices(raw, 8) == (
        np.random.default_rng(preparation.SEED).permutation(20)[:8].tolist()
    )
    stratified = datasets.Dataset.from_dict({
        "dataset_source": ["a"] * 3 + ["b"] * 3 + ["c"] * 4,
    })
    indices = preparation._select_indices(stratified, 5)
    assert len(set(indices)) == 5
    assert Counter(stratified.select(indices)["dataset_source"]) == {
        "a": 2, "b": 1, "c": 2,
    }
    assert preparation._select_indices(stratified, 5) == indices
    with pytest.raises(ValueError, match="Need 11 rows"):
        preparation._select_indices(stratified, 11)


def test_existing_output_is_rejected_before_loading(tmp_path, monkeypatch) -> None:
    path = tmp_path / "existing"
    path.with_suffix(".mixture.json").write_text("keep")

    def fail_if_loaded(path):
        pytest.fail("Existing outputs must be checked before loading sources.")

    monkeypatch.setattr(preparation, "_load_raw", fail_if_loaded)
    with pytest.raises(FileExistsError):
        preparation.generate(path, RECIPES[0].SPEC)
    assert path.with_suffix(".mixture.json").read_text() == "keep"
