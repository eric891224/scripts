"""Checks sm-dp converters with local datasets."""

import pathlib

import datasets

from sm_dp import adapters
from sm_dp import converters

_DATASET_ROOT = pathlib.Path("/home/siliconmind/cl/dataset")
_DOLCI_SHARD_PATH = (
    _DATASET_ROOT
    / "Dolci-Think-SFT-7B"
    / "data"
    / "train-00000-of-00156.parquet"
)


def main() -> None:
    """Converts and prints samples from three local datasets."""
    siliconmind_raw = datasets.load_dataset(
        str(_DATASET_ROOT / "siliconmind_oss-38k"),
        split="train",
    )
    siliconmind_dataset = converters.convert_dataset(
        siliconmind_raw,
        converter=adapters.siliconmind.convert_sample,
        category="spec2rtl",
    )

    smoltalk_raw = datasets.load_dataset(
        str(
            _DATASET_ROOT
            / "smoltalk"
            / "data"
            / "everyday-conversations"
        ),
        split="train",
    )
    smoltalk_dataset = converters.convert_dataset(
        smoltalk_raw,
        converter=adapters.smoltalk.convert_sample,
        category="conversations",
    )

    dolci_raw = datasets.load_dataset(
        "parquet",
        data_files=str(_DOLCI_SHARD_PATH),
        split="train[:8]",
    )
    dolci_dataset = converters.convert_dataset(
        dolci_raw,
        converter=adapters.dolci.convert_sample,
    )

    assert (
        siliconmind_dataset.features
        == smoltalk_dataset.features
        == dolci_dataset.features
    )
    assert all(
        "<think>" not in message["content"]
        for sample in dolci_dataset
        for message in sample["messages"]
        if message["role"] == "assistant"
    )

    print("SiliconMind dataset sample:")
    print(siliconmind_dataset[0])
    print("SmolTalk dataset sample:")
    print(smoltalk_dataset[0])
    print("Dolci Think dataset sample:")
    print(dolci_dataset[0])


if __name__ == "__main__":
    main()
