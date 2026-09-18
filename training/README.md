# Qwen domain + retention SFT

單節點 8 × H100、full fine-tuning、DeepSpeed ZeRO-2。輸入是已預處理的 Hugging Face Dataset。

## 從這裡開始

1. [準備環境與 dry-run](docs/quickstart.md)
2. [Slurm：smoke、正式訓練、續訓](docs/slurm.md)
3. [固定配方與輸出](docs/training.md)

其他：[W&B](docs/wandb.md) · [實作與測試](docs/internals.md) · [TP1 安裝](envs/tp1/README.md)

所有指令從 workspace 根目錄執行。先手動 source 設定；scripts 不會自動讀取 .env.sh。

- 改路徑／W&B／compiler：編輯 TP1 的 .env.sh，再 source。
- 改訓練配方：只改 training.py 的 RECIPE。
- 改 Slurm 資源：只改 submit_training.sbatch；GPU 數必須與其中的 NPROC_PER_NODE 一致。

舊的 MAX_STEPS、BATCH_SIZE 等環境變數已不再控制訓練；快速測試請改用 --smoke。
