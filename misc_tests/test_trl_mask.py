from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_name = "Qwen/Qwen3.5-9B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds = load_dataset(
    "/home/siliconmind/cl/dataset/smoltalk/data/everyday-conversations",
    split="test",
)

trainer = SFTTrainer(
    model=model_name,
    processing_class=tokenizer,
    train_dataset=train_ds,
    args=SFTConfig(
        output_dir="outputs/qwen-smoltalk",
        max_length=4096,
        assistant_only_loss=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        use_cpu=True,
        loss_type="nll",  # avoid the faulty chunked-loss patch
    ),
)

batch = next(iter(trainer.get_train_dataloader()))

input_ids = batch["input_ids"][0]
labels = batch["labels"][0]

print("=== Full conversation ===")
print(tokenizer.decode(input_ids, skip_special_tokens=False))

print("\n=== Tokens included in loss ===")
trained_ids = input_ids[labels != -100]
print(tokenizer.decode(trained_ids, skip_special_tokens=False))
