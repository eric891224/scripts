"""Full-parameter, text-only Qwen SFT on a preprocessed Hugging Face Dataset.

Run through run_training.sh (or use --help). --dry-run loads only the tokenizer
and previews a few examples; it does not load model weights or start training.
No validation split is manufactured from the oversampled mixture: repeated IDs
could otherwise leak between train and validation. Evaluation is a separate step.
"""

import argparse
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse run settings; defaults are an initial experiment, not a tuned recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", type=Path, default=WORKSPACE / "dataset/mixed/siliconmind-retention-v1")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "outputs/qwen-domain-retention")
    parser.add_argument(
        "--chat-template", type=Path,
        help="Optional Jinja file override; defaults to the tokenizer's own chat template.",
    )
    parser.add_argument("--deepspeed", type=Path, help="DeepSpeed JSON config; requires DeepSpeed in this Python environment.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides --epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=16)
    parser.add_argument("--max-length", type=positive_int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=positive_int, default=10)
    parser.add_argument("--save-steps", type=positive_int, default=250)
    parser.add_argument("--save-total-limit", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-num-proc", type=positive_int, default=1)
    parser.add_argument("--max-train-samples", type=positive_int, help="Use the first N rows for a smoke test; changes mixture ratios.")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download tokenizer or model files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-samples", type=positive_int, default=3)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive")
    if args.max_steps != -1 and args.max_steps <= 0:
        parser.error("--max-steps must be -1 or a positive integer")
    if args.deepspeed is not None and not args.deepspeed.is_file():
        parser.error("--deepspeed must point to an existing JSON config file")
    return args


def adapt_sample(sample: dict) -> dict:
    """Copy canonical messages and expose assistant reasoning to Qwen's template.

    The saved canonical schema and input dictionaries are not mutated. None becomes
    an empty reasoning_content only in this model-facing view: Qwen then emits
    an empty <think> block, rather than inventing reasoning for retention data.
    Tool/system/user messages are not given an assistant reasoning field.
    """
    messages = []
    for message in sample["messages"]:
        adapted = dict(message)
        reasoning = adapted.pop("reasoning", None)
        if adapted["role"] == "assistant":
            adapted["reasoning_content"] = reasoning if reasoning is not None else ""
        messages.append(adapted)
    return {"messages": messages}


def load_training_dataset(path: Path, limit: int | None = None):
    """Load one save_to_disk Dataset, not a raw Hub dataset or DatasetDict."""
    from datasets import Dataset, load_from_disk

    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise ValueError("Expected a single saved Dataset; select/save the training split first.")
    if not len(dataset) or "messages" not in dataset.column_names:
        raise ValueError("Training dataset must be nonempty and contain canonical messages.")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def resolve_model_path(model: str, local_files_only: bool) -> str:
    """Resolve a cached Hub snapshot for offline runs, including TRL config reads.

    Passing a local snapshot also prevents internal config loaders from issuing
    Hub requests without forwarding local_files_only. Missing weights still fail
    at training time; dry-run needs only the cached config and tokenizer files.
    """
    if not local_files_only or Path(model).is_dir():
        return model
    from transformers.utils.hub import cached_file

    # Do not require a complete snapshot (README/license/vision assets included).
    config_path = cached_file(model, "config.json", local_files_only=True)
    return str(Path(config_path).parent)


def resolve_training_template(tokenizer, path: Path | None = None) -> str:
    """Resolve the tokenizer's template, optionally overridden by a local file.

    No separate Jinja file is required when the tokenizer supplies a template.
    A requested file is read strictly: missing/empty overrides never fall back
    to the native template. TRL patches supported Qwen templates to preserve
    earlier reasoning and mark assistant bodies for loss; unsupported templates
    fail rather than silently switching to full-sequence loss. The tokenizer
    keeps the selected inference template for checkpoint serialization; the
    training variant is returned separately. Source files are never modified.
    """
    from trl.chat_template_utils import get_training_chat_template

    if path is not None:
        tokenizer.chat_template = path.read_text(encoding="utf-8")
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
        raise ValueError(
            "Expected a single nonempty tokenizer chat template. "
            "Use --chat-template PATH to supply a model-compatible Jinja template."
        )
    return get_training_chat_template(processing_class=tokenizer) or tokenizer.chat_template


def preview_sample(tokenizer, sample: dict, template: str, max_length: int) -> dict:
    """Preview the same keep-start truncation and assistant mask used by SFT.

    Counts exclude the first token because causal LM labels are shifted. The
    result describes one sample only; it is not a full-dataset quality audit.
    """
    encoded = tokenizer.apply_chat_template(
        adapt_sample(sample)["messages"],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, masks = encoded["input_ids"], encoded["assistant_masks"]
    if len(ids) != len(masks) or not any(masks):
        raise ValueError("Template produced no usable assistant loss mask.")
    retained_ids = ids[:max_length]
    retained_masks = masks[:max_length]
    loss_ids = [token for token, mask in zip(retained_ids[1:], retained_masks[1:], strict=True) if mask]
    return {
        "tokens": len(ids),
        "retained_tokens": len(retained_ids),
        "loss_tokens": len(loss_ids),
        "truncated": len(ids) > max_length,
        "loss_preview": tokenizer.decode(loss_ids, skip_special_tokens=False)[:1500],
    }


def build_training_config(args: argparse.Namespace):
    """Translate launcher options into TRL settings, keeping masking explicit."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(args.output_dir),
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        truncation_mode="keep_start",
        assistant_only_loss=True,
        packing=False,
        loss_type="nll",
        eos_token="<|im_end|>",
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        dataset_num_proc=args.dataset_num_proc if args.dataset_num_proc > 1 else None,
        report_to=args.report_to,
        model_init_kwargs={
            "dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype],
            "attn_implementation": args.attn_implementation,
            "use_cache": False,
            "device_map": None,  # Let Trainer place the model; never inference-style auto sharding.
            "local_files_only": args.local_files_only,
        },
    )


def check_output_directory(args: argparse.Namespace) -> None:
    """Refuse accidental reuse of a run directory unless explicitly resuming."""
    if args.resume_from_checkpoint is not None:
        if not (args.resume_from_checkpoint / "trainer_state.json").is_file():
            raise ValueError("--resume-from-checkpoint must point to a Trainer checkpoint.")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is nonempty. Choose a new run name or --resume-from-checkpoint.")


def check_distributed_output_directory(args: argparse.Namespace, config) -> None:
    """Check once on rank zero and share errors before any Trainer creates files.

    All ranks must participate. Independently checking the directory is racy:
    a faster rank may create Trainer output before a slower rank checks it.
    """
    errors = [None]
    if config.process_index == 0:
        try:
            check_output_directory(args)
        except (OSError, ValueError) as exc:
            errors[0] = str(exc)
    if config.world_size > 1:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(errors, from_process=0)
    if errors[0] is not None:
        raise ValueError(errors[0])


def main(argv: list[str] | None = None) -> None:
    """Inspect data, run full SFT, then save model/tokenizer, state, and metrics."""
    args = parse_args(argv)
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    dataset = load_training_dataset(args.dataset, args.max_train_samples)
    model_path = resolve_model_path(args.model, args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
    template = resolve_training_template(tokenizer, args.chat_template)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Dataset: {args.dataset} ({len(dataset):,} rows)")
        print("Full fine-tuning; assistant-only loss; reasoning + answer; packing disabled.")
        for index in range(min(args.preview_samples, len(dataset))):
            preview = preview_sample(tokenizer, dataset[index], template, args.max_length)
            print(f"Preview {index}: {json.dumps(preview, ensure_ascii=False)}")
            if preview["loss_tokens"] == 0:
                print("WARNING: this row has no retained assistant tokens; TRL will drop it.")
    if args.dry_run:
        print("Dry run complete: preview only, no model weights loaded or training started.")
        return

    from trl import SFTTrainer

    # SFTConfig initializes distributed state before model construction.
    config = build_training_config(args)
    check_distributed_output_directory(args, config)
    # On a shared filesystem, rank zero builds the adapter cache first. TRL
    # separately coordinates its own tokenization/preparation inside SFTTrainer.
    with config.main_process_first(desc="Adapting reasoning for Qwen"):
        dataset = dataset.map(
            adapt_sample,
            num_proc=args.dataset_num_proc if args.dataset_num_proc > 1 else None,
            desc="Adapting reasoning for Qwen",
        )
    trainer = SFTTrainer(
        model=model_path,
        processing_class=tokenizer,  # Text path, not a vision processor.
        train_dataset=dataset,
        args=config,
    )
    if not len(trainer.train_dataset):
        raise ValueError("No trainable samples remain after truncation; increase --max-length.")
    if trainer.is_world_process_zero():
        global_batch = config.world_size * args.batch_size * args.gradient_accumulation_steps
        print(f"Prepared rows: {len(trainer.train_dataset):,}/{len(dataset):,}. Truncation can change mixture ratios.")
        print(f"World size: {config.world_size}; global batch size: {global_batch}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = vars(args) | {
            "resolved_model": model_path,
            "input_rows": len(dataset),
            "prepared_rows": len(trainer.train_dataset),
            "world_size": config.world_size,
            "global_batch_size": global_batch,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        # Keep original run arguments when resuming.
        metadata_name = "resume_arguments.json" if args.resume_from_checkpoint else "run_arguments.json"
        (args.output_dir / metadata_name).write_text(json.dumps(metadata, default=str, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "training_chat_template.jinja").write_text(template, encoding="utf-8")
        if args.deepspeed:
            # Snapshot the input settings; "auto" values are resolved by Trainer.
            config_name = "resume_deepspeed_config.json" if args.resume_from_checkpoint else "deepspeed_config.json"
            (args.output_dir / config_name).write_text(args.deepspeed.read_text(encoding="utf-8"), encoding="utf-8")
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
    )
    trainer.save_model(str(args.output_dir / "final"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
