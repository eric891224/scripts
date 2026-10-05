"""Builds experiment C: RTL 80%, general 15%, tools 5%."""

from collections import Counter
from functools import partial
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

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

DATASET_ROOT = Path("/home/siliconmind/cl/dataset")
OUTPUT_PATH = DATASET_ROOT / "mixed" / "siliconmind-c-rtl80-general15-tools5-v2"
RECIPE_PATH = OUTPUT_PATH.with_suffix(".mixture.json")


def load_and_convert(
    path: Path,
    *,
    converter: SampleConverter,
    category: str | None = None,
) -> Dataset:
    """Loads and converts all training rows from a local source.

    Args:
        path: The source dataset directory.
        converter: The source sample converter.
        category: The category, or None to use source categories.

    Returns:
        A dataset with canonical storage features.
    """
    raw = load_dataset(str(path), split="train")
    return convert_dataset(
        raw,
        converter=converter,
        category=category,
        desc=f"Converting {category or path.name}",
    )


def main() -> None:
    """Converts sources, mixes exact quotas, and saves experiment C."""
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

    datasets["smoltalk-apigen"] = load_and_convert(
        DATASET_ROOT / "smoltalk" / "data" / "apigen-80k",
        converter=partial(convert_smoltalk_sample, parse_tool_calls=True),
        category="apigen-80k",
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
            "tools": DatasetGroupSpec(
                weight=5,
                datasets=[
                    DatasetSpec(name="smoltalk-apigen", weight=1),
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
    assert source_counts["dolci"] == 0
    assert category_counts["apigen-80k"] == 2_250

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
