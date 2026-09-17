# Slurm：單節點 8 × H100 80GB

[回 manual 索引](../README.md)

本頁：[Server 準備](#setup) · [兩步 smoke test](#smoke-test) · [正式提交](#training) · [續訓](#resume) · [離線](#offline) · [架構與預設值](#defaults)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

操作入口：[submit_training.sbatch](../submit_training.sbatch)。先完成 [Quickstart](quickstart.md)，再確認下列 cluster 設定；全量參數請查 [Training 參數表](training.md#parameters)。

<a id="setup"></a>

## 1. Server 上需先確認

1. **Queue/resource 規則**：partition、account、必要的 QoS／constraint。`--gres=gpu:8` 只要求 GPU 數量，不能保證拿到 H100；必須由站台提供的 partition、constraint 或 typed GRES 選到正確卡型。若使用 typed GRES，要依站台實際名稱調整，不能猜測。
2. **Python/CUDA 環境**：必要時在 `.sbatch` 標示的位置加入該 cluster 的 `module load`／環境啟用；目前不包含 container launcher。腳本不會自動安裝依賴或猜 module 名稱。
3. **共享路徑**：compute node 需能讀取 workspace、模型、tokenizer 與 Dataset，並能寫入資料 cache 與輸出目錄。Template 預設隨 tokenizer 載入；只有指定 `CHAT_TEMPLATE`／`--chat-template` 時才需額外準備該 Jinja 檔案。
4. **資源與儲存空間**：8 GPU 的 full fine-tuning、optimizer checkpoints 與 `final/` 都需要實測容量與寫入時間；本文件列出的 CPU/RAM/time 是可調的起始值，不是已驗證容量保證。

提交時必須指定 `TRAINING_ENV_FILE=/absolute/path/to/experiment.sh`。Slurm 不搜尋 `envs/`，也不選擇 Python 環境；設定檔中的 `PYTHON_BIN` 應指向已安裝訓練依賴與 DeepSpeed 的 interpreter。可沿用 [TP1 設定範本](../envs/tp1/README.md)，也可將設定與 Python 環境放在其他共享路徑。

Slurm script 要求 `NPROC_PER_NODE=8` 與有效的 `DEEPSPEED_CONFIG`，不會覆寫你的訓練參數。資料上傳方式見 [Quickstart](quickstart.md#setup)。

腳本執行前會檢查 DeepSpeed 可 import、CUDA 可用、剛好看見 8 張 GPU，並列印 PyTorch/CUDA/DeepSpeed 版本與 GPU 名稱／記憶體。這不代表完整 backward 或 checkpoint 已驗證。

<a id="smoke-test"></a>

## 2. 第一次：只提交兩步 smoke test

以下 `YOUR_H100_PARTITION` 與 `YOUR_ACCOUNT` 必須替換成你的 cluster 設定；若站台不需要 account，移除該選項。從 workspace 根目錄提交：

```bash
TRAINING_ENV_FILE=/absolute/path/to/experiment.sh \
MAX_STEPS=2 \
MAX_TRAIN_SAMPLES=32 \
LOGGING_STEPS=1 \
SAVE_STEPS=1 \
sbatch --partition=YOUR_H100_PARTITION --account=YOUR_ACCOUNT \
  --time=00:30:00 scripts/training/submit_training.sbatch
```

使用範本設定時 global batch 為 16。第一次還包含模型載入、preprocessing，以及可能的 DeepSpeed 編譯；30 分鐘只是起始測試時間，可依站台調整。中途與最後儲存仍是完整模型／訓練狀態，並不因只跑兩步就變成小檔案。

檢查 `slurm-<job-name>-<job-id>.out`／`.err`：

- GPU preflight 確認拿到預期的卡。
- 訓練資訊包含 `World size: 8; global batch size: 16`。
- loss／gradient 等指標正常，8 張卡都有工作，峰值記憶體可接受。
- `checkpoint-1/`、`checkpoint-2/` 與正常完成時的 `final/` 可建立。
- 再用 checkpoint 做短續訓，確認不只可以存，也能恢復。

Slurm logs 預設寫入提交目錄；若自行用 `--output`／`--error` 指向子目錄，須先建立父目錄，Slurm 會在腳本執行前開啟 log。

<a id="training"></a>

## 3. 正式提交

Smoke test 成功後，使用全部資料跑一個 epoch，例如：

```bash
TRAINING_ENV_FILE=/absolute/path/to/experiment.sh \
REPORT_TO=wandb \
MAX_STEPS=-1 \
MAX_TRAIN_SAMPLES= \
EPOCHS=1 \
sbatch --partition=YOUR_H100_PARTITION --account=YOUR_ACCOUNT \
  --time=12:00:00 scripts/training/submit_training.sbatch
```

12 小時是範例，不是訓練時間估計；依 smoke test 的吞吐量、資料量和 queue 上限選擇。`MAX_STEPS=-1` 與空的樣本上限避免沿用先前 shell 的 smoke 設定。

Slurm 會將 `.sbatch` 複製到 spool 目錄，因此腳本透過 `SLURM_SUBMIT_DIR` 找 workspace，不依賴腳本本身所在位置。若不是從根目錄提交，請先設定絕對路徑 `WORKSPACE_DIR=/path/to/workspace`。不要直接在 login node 用 `bash submit_training.sbatch` 啟動訓練。

<a id="resume"></a>

## 4. 中斷後續訓

續訓時指定原 run 目錄與 checkpoint（`1234` 為示例 job ID）：

```bash
TRAINING_ENV_FILE=/absolute/path/to/experiment.sh \
OUTPUT_DIR=outputs/qwen-zero2-1234 \
RESUME_FROM_CHECKPOINT=outputs/qwen-zero2-1234/checkpoint-250 \
sbatch --partition=YOUR_H100_PARTITION --account=YOUR_ACCOUNT \
  scripts/training/submit_training.sbatch
```

續訓應保留相同 dataset、模型、ZeRO 配置與 8 GPU world size，也要重新提供原本的其他自訂參數。不要只複製 checkpoint 的 rank 0 檔案：DeepSpeed 的 optimizer 等狀態是分片保存的，必須保留完整 checkpoint 目錄。`final/` 供模型載入，不是完整續訓狀態。

目前沒有自動 requeue 或在時間到期前自動存檔的 signal handler。Job 被時間限制中止後，只能從最後**完整寫入**的 checkpoint 恢復，請選合適的 `SAVE_STEPS`。

W&B 是否延續同一個 run 是另一件事：Trainer checkpoint 的續訓不代表 W&B run 也自動續接；不要只靠相同的顯示名稱判斷它們是同一個 run。 詳見 [W&B manual](wandb.md#resume)。

<a id="offline"></a>

## 5. Compute node 不能連外

預先備妥模型與 tokenizer，訓練時使用本地路徑／快取並附上 `--local-files-only`。若要保留 W&B 記錄，另設 `WANDB_MODE=offline REPORT_TO=wandb`，之後在可連網環境同步：模型離線與 W&B 離線是獨立設定。

若只想先檢查資料/template，使用 [Quickstart 的單 process dry-run](quickstart.md#dry-run) 即可，不必提交 8 GPU job。這個模式不初始化 DeepSpeed，也不驗證分散式訓練。

Slurm 行為依站台設定而異，可參考官方 [sbatch](https://slurm.schedmd.com/sbatch.html) 與 [srun](https://slurm.schedmd.com/srun.html) 文件。

<a id="defaults"></a>

## 架構與 Slurm 預設值

```text
sbatch：1 node、8 GPUs、1 task
  → srun：1 task，能看見分配到的 8 GPUs
  → run_training.sh
  → python -m torch.distributed.run：8 workers
  → 每個 worker 執行 training.py，DeepSpeed 協調訓練
```

**不要改成 8 個 Slurm tasks 再各自啟動 torchrun。** 這會重複建立 workers。腳本會拒絕多節點／多 Slurm task 配置，也不會覆寫 Slurm 提供的 `CUDA_VISIBLE_DEVICES`。

| 設定 | `.sbatch` 資源／`.env.example.sh` 訓練初始值 |
| --- | --- |
| 節點／GPU／Slurm tasks | 1／8／1 |
| CPU／host RAM／時間 | 32 CPUs／256 GiB (`--mem=256G`)／2 小時；均需依 cluster 與實際工作量調整。 |
| 訓練方式 | BF16 full fine-tuning、ZeRO-2、gradient checkpointing。 |
| 每卡 micro-batch／gradient accumulation | 1／2，所以 global batch = `8 × 1 × 2 = 16`。 |
| 輸出目錄 | `<workspace>/outputs/qwen-zero2-<job-id>`；可用 `OUTPUT_DIR` 覆蓋。 |
| W&B 名稱 | `qwen-zero2-<job-id>`；可用 `WANDB_NAME` 覆蓋。 |
| W&B 記錄 | 仍預設關閉，用 `REPORT_TO=wandb` 啟用；只有主 rank 初始化 run。 |

ZeRO-2 分攤梯度與 optimizer state，各 GPU 仍保留完整模型參數。JSON 不啟用 CPU/NVMe offload，也不另指定 optimizer/scheduler，沿用 Trainer 的選擇。precision、batch 與 gradient clipping 設為 `auto`，避免在 bash、Python、JSON 三處維護不同數字。參考 [Transformers DeepSpeed 文件](https://huggingface.co/docs/transformers/deepspeed)。
