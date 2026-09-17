# Training：訓練、續訓與參數

[回 manual 索引](../README.md)

本頁：[Smoke test](#smoke-test) · [正式訓練](#training) · [續訓](#resume) · [輸出](#outputs) · [參數表](#parameters)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

本頁的直接執行範例適用於已允許直接使用 GPU 的環境，且已完成 [Quickstart](quickstart.md)。**Slurm 使用者請用 [Slurm manual](slurm.md)，不要在 login node 直接跑以下訓練指令。**

<a id="smoke-test"></a>

## 1. 一般環境的 smoke test

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_STEPS=2 \
MAX_TRAIN_SAMPLES=32 \
GRADIENT_ACCUMULATION_STEPS=1 \
LOGGING_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=outputs/qwen-smoke \
bash scripts/training/run_training.sh
```

這會載入**完整模型**，取資料前 32 筆，做 2 個 optimizer steps，測試 forward/backward 與 checkpoint 儲存。它不是小模型測試；少量樣本／steps 不會免除 9B full fine-tuning 的模型與 optimizer 記憶體需求。

`MAX_TRAIN_SAMPLES` 是取前 N 筆，不是分層抽樣，因此不保證維持 80/20。它適合 smoke test，不應直接當成正式混合比例實驗。

<a id="training"></a>

## 2. 正式訓練

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=Qwen/Qwen3.5-9B \
DATASET=dataset/mixed/siliconmind-retention-v1 \
OUTPUT_DIR=outputs/qwen-domain-retention-v1 \
EPOCHS=1 \
LEARNING_RATE=2e-5 \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
MAX_LENGTH=4096 \
DTYPE=bf16 \
bash scripts/training/run_training.sh
```

正式執行前確認 shell 沒有殘留之前 `export` 的 `MAX_STEPS` 或 `MAX_TRAIN_SAMPLES`；正的 `MAX_STEPS` 會優先於 epochs，樣本上限也會繼續生效。上例採用單 GPU；launcher 不會因為看到多張 GPU 就自動啟動 DDP 或切分模型。

單 GPU 預設的有效 batch size 約為 `1 × 16 = 16` 筆／optimizer step（尾批可能不足）。`MAX_STEPS`、`LOGGING_STEPS` 與 `SAVE_STEPS` 都以 optimizer steps 計算，不是單筆資料或 gradient accumulation 的 micro-batch 次數。

<a id="resume"></a>

## 3. 中斷後續訓

假設 `checkpoint-250` 已存在：

```bash
OUTPUT_DIR=outputs/qwen-domain-retention-v1 \
RESUME_FROM_CHECKPOINT=outputs/qwen-domain-retention-v1/checkpoint-250 \
bash scripts/training/run_training.sh
```

請沿用原本的模型、dataset、template、batch 等設定；上例假設其他設定都使用預設值。程式不會自動從 `run_arguments.json` 恢復 CLI，也不會比對本次參數是否與原 run 一致。

新的訓練若發現輸出目錄非空，會拒絕執行，避免不小心混用結果。續訓路徑需指向含有 `trainer_state.json` 的 `checkpoint-N/`；這只是基本檢查，是否有完整狀態仍由 Trainer 載入時確認。

`final/` 是供載入模型的產物，不等同包含 optimizer/scheduler 等狀態的續訓 checkpoint。若設定 `MAX_STEPS`，它代表整個 run 的目標總步數，不是「再跑幾步」。

<a id="outputs"></a>

## 4. 輸出內容

正常完成後，輸出目錄大致包含：

```text
OUTPUT_DIR/
  run_arguments.json             # CLI 設定、resolved model、處理前後筆數
                                # 另含 world size、global batch、Slurm job ID
  resume_arguments.json          # 有續訓時才寫出；再次續訓會更新此檔
  deepspeed_config.json          # 啟用時保存的輸入設定，保留 auto 值
  resume_deepspeed_config.json   # 使用 DeepSpeed 續訓時另存本次設定
  training_chat_template.jinja   # training 用的 template
  checkpoint-N/                 # 依 save interval 產生的續訓 checkpoint
  final/                        # 最終 model、tokenizer 及 inference template
  trainer_state.json
  train_results.json
```

實際模型檔名／分片方式由 Transformers 決定。還沒到儲存間隔就結束的 run，不一定有中途 checkpoint，但正常完成仍會寫出 `final/`。Trainer 也可能另外產生其他標準 metadata 檔案。

`run_arguments.json` 不是完整實驗封存：不會複製 dataset、mixture recipe、原始碼或依賴 lockfile，也不包含所有未顯式設定的 Trainer 預設值。需要重現實驗時，請另外保留 dataset 版本、recipe、模型 revision、程式 commit 與 `sm-dp/uv.lock`。

<a id="parameters"></a>

## 5. 參數怎麼設定

可以修改 bash 中的預設值，也可以只在執行時傳入環境變數。額外 CLI 參數放在最後，會覆蓋 launcher 傳入的同名設定，例如：

```bash
MAX_LENGTH=4096 bash scripts/training/run_training.sh --max-length 2048
bash scripts/training/run_training.sh --help
```

上例實際使用 2048。Launcher 會依腳本位置找 workspace，因此預設路徑不依賴目前目錄；但**自行指定的相對路徑**仍相對於你執行指令時的工作目錄。

| 環境變數 | Python CLI | 預設值／用途 |
| --- | --- | --- |
| `PYTHON_BIN` | 無，僅供 launcher 使用 | `<workspace>/sm-dp/.venv/bin/python`。 |
| `MODEL` | `--model` | `Qwen/Qwen3.5-9B`；也接受本地模型目錄。 |
| `DATASET` | `--dataset` | `<workspace>/dataset/mixed/siliconmind-retention-v1`。 |
| `OUTPUT_DIR` | `--output-dir` | `<workspace>/outputs/qwen-domain-retention`。 |
| `CHAT_TEMPLATE` | `--chat-template` | 預設未設定，使用 `MODEL` 的 tokenizer template；指定 Jinja 檔案時才覆蓋，空環境變數視為未設定。 |
| `EPOCHS` | `--epochs` | `1`。 |
| `MAX_STEPS` | `--max-steps` | `-1` 表示由 epochs 控制；正整數會覆蓋 epochs。 |
| `LEARNING_RATE` | `--learning-rate` | `2e-5`。 |
| `BATCH_SIZE` | `--batch-size` | `1`，每張 GPU 的 micro-batch 大小。 |
| `GRADIENT_ACCUMULATION_STEPS` | `--gradient-accumulation-steps` | `16`。 |
| `MAX_LENGTH` | `--max-length` | `4096`，整段 conversation 的上限，包含 template、prompt、reasoning、答案。 |
| `DTYPE` | `--dtype` | `bf16`；可選 `fp16`、`fp32`。 |
| `ATTN_IMPLEMENTATION` | `--attn-implementation` | `sdpa`；其他 backend 需模型與環境支援。 |
| `LOGGING_STEPS` | `--logging-steps` | `10`。 |
| `SAVE_STEPS` | `--save-steps` | `250`。 |
| `SAVE_TOTAL_LIMIT` | `--save-total-limit` | `2`，限制保留的中途 checkpoints，不包含另存的 `final/`。 |
| `SEED` | `--seed` | `42`，同時設定訓練與資料 seed；不保證跨硬體完全一致。 |
| `DATASET_NUM_PROC` | `--dataset-num-proc` | `1`；單程序時在目前 process 處理，大於 1 才使用多程序。 |
| `NPROC_PER_NODE` | 無，控制 launcher | `1`；大於 1 時用單節點 torchrun 啟動對應數量的 workers。Slurm template 固定設為 `8`。 |
| `DEEPSPEED_CONFIG` | `--deepspeed` | 一般執行預設未啟用；Slurm template 預設 `deepspeed_zero2.json`。 |
| `MAX_TRAIN_SAMPLES` | `--max-train-samples` | 未設定，使用全部資料。 |
| `RESUME_FROM_CHECKPOINT` | `--resume-from-checkpoint` | 未設定，從指定 checkpoint 恢復。 |
| `REPORT_TO` | `--report-to` | `none`；也支援 `tensorboard`、`wandb`，需另外備妥套件與設定；W&B 會啟用外部記錄。 |

以下選項沒有對應的 bash 環境變數，直接附在指令後面：

- `--dry-run`：只預覽，不訓練。
- `--preview-samples N`：預覽前 N 筆，預設 3；正式訓練前也會預覽。
- `--local-files-only`：只使用本地／快取模型檔案，缺檔時報錯。
- `--gradient-checkpointing`／`--no-gradient-checkpointing`：開啟／關閉 gradient checkpointing，預設開啟；以重算換取較低 activation 記憶體使用量。

W&B 專用的 `WANDB_*` 設定在 [W&B manual](wandb.md#settings)；Slurm 與一般執行不同的預設值在 [Slurm manual](slurm.md#defaults)。
