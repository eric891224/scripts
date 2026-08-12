from pathlib import Path

from datasets import load_dataset
from sm_dp.converters.siliconmind import convert_sample as convert_siliconmind_sample
from sm_dp.converters.smoltalk import convert_sample as convert_smoltalk_sample

SILICONMIND_DATASET_PATH = Path("/home/siliconmind/cl/dataset/siliconmind_oss-38k")
SMOLTALK_DATASET_PATH = Path(
    "/home/siliconmind/cl/dataset/smoltalk/data/everyday-conversations"
)

siliconmind_raw = load_dataset(str(SILICONMIND_DATASET_PATH), split="train")
siliconmind = siliconmind_raw.map(
    lambda row, index: convert_siliconmind_sample(
        row, index=index, category="spec2rtl"
    ),
    with_indices=True,
    remove_columns=siliconmind_raw.column_names,
)

smoltalk_raw = load_dataset(str(SMOLTALK_DATASET_PATH), split="train")
smoltalk = smoltalk_raw.map(
    lambda row, index: convert_smoltalk_sample(row, index=index, category="smoltalk"),
    with_indices=True,
    remove_columns=smoltalk_raw.column_names,
)

print("SiliconMind dataset sample:")
print(siliconmind[0])
print("Smoltalk dataset sample:")
print(smoltalk[0])
