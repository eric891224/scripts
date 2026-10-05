"""Builds experiment B: RTL 80%, general 15%, reasoning 5%."""

from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from sm_dp.adapters.dolci import convert_sample as convert_dolci_sample
from sm_dp.adapters.siliconmind import (
    convert_sample as convert_siliconmind_sample,
)
from sm_dp.adapters.smoltalk import convert_sample as convert_smoltalk_sample
from sm_dp.converters import SampleConverter, convert_dataset
from sm_dp.mixing import (
    DatasetGroupSpec,
    DatasetSpec,
    MixtureSpec,
    ReplacementMode,
    mix,
)
from sm_dp.parsers.reasoning import split_reasoning

DATASET_ROOT = Path("/home/siliconmind/cl/dataset")
OUTPUT_PATH = (
    DATASET_ROOT / "mixed" / "siliconmind-b-rtl80-general15-reasoning5-v2"
)
RECIPE_PATH = OUTPUT_PATH.with_suffix(".mixture.json")


def has_complete_assistant_responses(raw: dict[str, Any]) -> bool:
    """Checks for think blocks without an answer or tool calls.

    Rejects the full conversation if an assistant has a closed think
    block but no answer or tool calls. Does not change the input.
    Other format errors remain subject to converter validation.

    Args:
        raw: A source row with a messages field.

    Returns:
        False if an assistant think block has no answer or tool calls.
        Otherwise, True.
    """
    for message in raw["messages"]:
        if message["role"] != "assistant":
            continue

        content = message.get("content") or ""
        if (
            content.lstrip().startswith("<think>")
            and "</think>" in content
        ):
            _, answer = split_reasoning(content)
            if not answer and not message.get("tool_calls"):
                return False

    return True


def load_and_convert(
    path: Path,
    *,
    converter: SampleConverter,
    category: str | None = None,
) -> Dataset:
    """Loads, filters Dolci responses, and converts training rows.

    For Dolci, removes conversations with closed assistant think blocks
    but no answer or tool calls. Reports the kept and removed row counts.
    Does not filter other sources.

    Args:
        path: The source dataset directory.
        converter: The source sample converter.
        category: The category, or None to use source categories.

    Returns:
        A dataset with canonical storage features.
    """
    raw = load_dataset(str(path), split="train")
    if converter is convert_dolci_sample:
        before = len(raw)
        raw = raw.filter(
            has_complete_assistant_responses,
            desc="Filtering Dolci responses without answers",
        )
        print(f"Dolci: kept {len(raw):,}; removed {before - len(raw):,}")

    return convert_dataset(
        raw,
        converter=converter,
        category=category,
        desc=f"Converting {category or path.name}",
    )


def main() -> None:
    """Converts sources, mixes exact quotas, and saves experiment B."""
    if OUTPUT_PATH.exists() or RECIPE_PATH.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_PATH}")

    # 1. Load and convert every source dataset.
    datasets = {
        "siliconmind-spec2rtl": load_and_convert(
            DATASET_ROOT / "siliconmind_oss-38k",
            converter=convert_siliconmind_sample,
            category="spec2rtl",
        ),
        "smoltalk-magpie": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "smol-magpie-ultra",
            converter=convert_smoltalk_sample,
            category="smol-magpie-ultra",
        ),
        "smoltalk-openhermes": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "openhermes-100k",
            converter=convert_smoltalk_sample,
            category="openhermes-100k",
        ),
        "smoltalk-constraints": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "smol-constraints",
            converter=convert_smoltalk_sample,
            category="smol-constraints",
        ),
        "smoltalk-everyday": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "everyday-conversations",
            converter=convert_smoltalk_sample,
            category="everyday-conversations",
        ),
    }

    # Keep the same 2,250-row, source-stratified Dolci cohort in B and D.
    dolci = load_and_convert(
        DATASET_ROOT / "Dolci-Think-SFT-7B",
        converter=convert_dolci_sample,
    )
    dolci_counts = Counter(dolci["category"])
    dolci_sources = {
        category: dolci.filter(
            lambda value: value == category,
            input_columns="category",
            desc=f"Selecting Dolci {category}",
        )
        for category in sorted(dolci_counts)
    }
    datasets["dolci-think"] = mix(
        datasets=dolci_sources,
        spec=MixtureSpec(
            size=2_250,
            seed=42,
            replacement_strategy=ReplacementMode.NEVER,
            groups={
                "reasoning": DatasetGroupSpec(
                    weight=1,
                    datasets=[
                        DatasetSpec(
                            name=category,
                            weight=dolci_counts[category],
                        )
                        for category in sorted(dolci_counts)
                    ],
                ),
            },
        ),
    )

    # 2. Define the mixture recipe. Weights count samples, not tokens.
    spec = MixtureSpec(
        size=45_000,
        seed=42,
        replacement_strategy=ReplacementMode.NEVER,
        groups={
            "domain": DatasetGroupSpec(
                weight=80,
                datasets=[
                    DatasetSpec(name="siliconmind-spec2rtl", weight=1),
                ],
            ),
            "general": DatasetGroupSpec(
                weight=15,
                datasets=[
                    DatasetSpec(name="smoltalk-magpie", weight=50),
                    DatasetSpec(name="smoltalk-openhermes", weight=25),
                    DatasetSpec(name="smoltalk-constraints", weight=15),
                    DatasetSpec(name="smoltalk-everyday", weight=10),
                ],
            ),
            "reasoning": DatasetGroupSpec(
                weight=5,
                datasets=[
                    DatasetSpec(name="dolci-think", weight=1),
                ],
            ),
        },
    )

    # 3. Produce the fixed-quota mixture.
    mixed = mix(datasets=datasets, spec=spec)

    # 4. Inspect the result before saving.
    source_counts = Counter(mixed["source"])
    category_counts = Counter(mixed["category"])
    unique_ids = len(set(mixed["id"]))
    repeated_rows = len(mixed) - unique_ids
    general_rows = sum(
        category_counts[category]
        for category in (
            "smol-magpie-ultra",
            "openhermes-100k",
            "smol-constraints",
            "everyday-conversations",
        )
    )

    print(f"Total rows:    {len(mixed):,}")
    print(f"Sources:       {source_counts}")
    print(f"Categories:    {category_counts}")
    print(f"Unique IDs:    {unique_ids:,}")
    print(f"Repeated rows: {repeated_rows:,}")

    assert len(mixed) == spec.size
    assert category_counts["spec2rtl"] == 36_000
    assert general_rows == 6_750
    assert source_counts["dolci"] == 2_250
    assert category_counts["apigen-80k"] == 0

    # 5. Save the Hugging Face Dataset and its recipe.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(OUTPUT_PATH)
    RECIPE_PATH.write_text(
        spec.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 6. Reload the artifact to verify it.
    reloaded = load_from_disk(OUTPUT_PATH)
    assert len(reloaded) == len(mixed)
    assert reloaded.features == mixed.features
    assert reloaded["id"] == mixed["id"]

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Recipe saved to:  {RECIPE_PATH}")


if __name__ == "__main__":
    main()
