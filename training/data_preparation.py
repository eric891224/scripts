from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from sm_dp.converters import convert_dataset
from sm_dp.converters.siliconmind import (
    convert_sample as convert_siliconmind_sample,
)
from sm_dp.converters.smoltalk import (
    convert_sample as convert_smoltalk_sample,
)
from sm_dp.mixing import (
    DatasetGroupSpec,
    DatasetSpec,
    MixtureSpec,
    ReplacementMode,
    mix,
)

DATASET_ROOT = Path("/home/siliconmind/cl/dataset")

OUTPUT_PATH = DATASET_ROOT / "mixed" / "siliconmind-retention-v1"

RECIPE_PATH = OUTPUT_PATH.with_suffix(".mixture.json")


def load_and_convert(
    path: Path,
    *,
    converter,
    category: str,
) -> Dataset:
    raw = load_dataset(
        str(path),
        split="train",
    )

    return convert_dataset(
        raw,
        converter=converter,
        category=category,
        desc=f"Converting {category}",
    )


def main() -> None:
    # 1. Load and convert every source dataset.
    datasets = {
        "siliconmind-spec2rtl": load_and_convert(
            DATASET_ROOT / "siliconmind_oss-38k",
            converter=convert_siliconmind_sample,
            category="spec2rtl",
        ),
        "smoltalk-everyday": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "everyday-conversations",
            converter=convert_smoltalk_sample,
            category="everyday-conversations",
        ),
        "smoltalk-math": load_and_convert(
            DATASET_ROOT / "smoltalk" / "data" / "metamathqa-50k",
            converter=convert_smoltalk_sample,
            category="metamathqa-50k",
        ),
    }

    # 2. Define the mixture recipe.
    spec = MixtureSpec(
        size=45_000,
        seed=42,
        replacement_strategy=ReplacementMode.AUTO,
        max_repeat_factor=4.0,
        groups={
            "domain": DatasetGroupSpec(
                weight=80,
                datasets=[
                    DatasetSpec(
                        name="siliconmind-spec2rtl",
                        weight=1,
                    ),
                ],
            ),
            "retention": DatasetGroupSpec(
                weight=20,
                datasets=[
                    DatasetSpec(
                        name="smoltalk-everyday",
                        weight=1,
                    ),
                    DatasetSpec(
                        name="smoltalk-math",
                        weight=1,
                    ),
                ],
            ),
        },
    )

    # 3. Produce the fixed-quota mixture.
    mixed = mix(
        datasets=datasets,
        spec=spec,
    )

    # 4. Inspect the result before saving.
    source_counts = Counter(mixed["source"])
    category_counts = Counter(mixed["category"])
    unique_ids = len(set(mixed["id"]))
    repeated_rows = len(mixed) - unique_ids

    print(f"Total rows:   {len(mixed):,}")
    print(f"Sources:      {source_counts}")
    print(f"Categories:   {category_counts}")
    print(f"Unique IDs:   {unique_ids:,}")
    print(f"Repeated rows:{repeated_rows:,}")

    assert len(mixed) == spec.size

    # Expected quotas:
    assert category_counts["spec2rtl"] == 36_000
    assert category_counts["everyday-conversations"] == 4_500
    assert category_counts["metamathqa-50k"] == 4_500

    # 5. Save the Hugging Face Dataset and its recipe.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
