"""Builds experiment D: RTL 80%, general 10%, reasoning 5%, tools 5%."""

from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from sm_dp import schema, storage
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

DATASET_ROOT = Path("/home/siliconmind/cl/dataset")
OUTPUT_PATH = (
    DATASET_ROOT / "mixed"
    / "siliconmind-d-rtl80-general10-reasoning5-tools5-v2"
)
RECIPE_PATH = OUTPUT_PATH.with_suffix(".mixture.json")


def sample_dolci(
    path: Path,
    *,
    size: int,
    seed: int,
) -> Dataset:
    """Samples valid Dolci conversations without replacement.

    Visits raw rows in a random order. Stops after size valid rows.
    Skips rows with adapter or schema ValueError exceptions.
    Does not enforce quotas for individual Dolci sources.

    Args:
        path: The local Dolci source directory.
        size: The number of valid conversations to select.
        seed: The seed for the random candidate order.

    Returns:
        A dataset with canonical storage features.

    Raises:
        ValueError: If size is not positive or too few valid rows exist.
        Exception: Other loading, conversion, or storage errors propagate.
    """
    if size <= 0:
        raise ValueError("Dolci sample size must be positive.")

    raw = load_dataset(str(path), split="train")
    indices = np.random.default_rng(seed).permutation(len(raw))
    rows: list[dict[str, Any]] = []
    rejected = 0

    for index in indices:
        index = int(index)
        try:
            sample = convert_dolci_sample(raw[index], index=index)
        except ValueError:
            # Includes schema validation and adapter data errors.
            rejected += 1
            continue

        rows.append(storage.encode_sample(sample))
        if len(rows) == size:
            break

    if len(rows) != size:
        raise ValueError(
            f"Need {size} valid Dolci rows; found {len(rows)}."
        )

    print(f"Dolci: selected {len(rows):,}; rejected {rejected:,}")
    print(f"Dolci sources: {Counter(row['category'] for row in rows)}")
    return Dataset.from_list(rows, features=schema.CONVERSATION_FEATURES)


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
    """Converts sources, mixes exact quotas, and saves experiment D."""
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

    # Sample candidates first. Convert only enough valid rows for the quota.
    datasets["dolci-think"] = sample_dolci(
        DATASET_ROOT / "Dolci-Think-SFT-7B",
        size=2_250,
        seed=42,
    )

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
                weight=10,
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
    assert general_rows == 4_500
    assert source_counts["dolci"] == 2_250
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
