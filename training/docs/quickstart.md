# 第一次使用

[回索引](../README.md)

## 1. 準備環境和設定

依 [TP1 安裝說明](../envs/tp1/README.md) 安裝套件，再執行：

```bash
cp -n scripts/training/envs/tp1/.env.example.sh scripts/training/envs/tp1/.env.sh
# 編輯 .env.sh 的 Python、model、dataset、output 與 W&B 設定
source scripts/training/envs/tp1/.env.sh
```

範本採用 TP1 server 的硬編碼路徑，使用前請核對。已有舊 .env.sh 請手動對照新範本更新；cp -n 不會覆寫它。

每次 source 都覆寫範本列出的環境變數。不再推導路徑、產生時間戳記或保留舊值。新實驗請明確修改 OUTPUT_DIR 與 WANDB_NAME。

Dataset 必須是 Dataset.save_to_disk() 儲存的完整目錄，包含 canonical messages；不是 JSONL 或 DatasetDict。Model 可用 Hub ID 或本地模型路徑。

## 2. 預覽

```bash
bash scripts/training/run_training.sh --dry-run
```

只載入 tokenizer、預覽前三筆資料，不載入模型權重、不需要 GPU、不啟動 W&B。

確認 loss_preview 包含 assistant reasoning／答案、不包含 user／system；loss_tokens 應大於 0。truncated 表示樣本超過固定的 4096 tokens。

預設使用 tokenizer 自帶的 chat template，不必準備 qwen.jinja。需要離線時使用本地 MODEL 路徑，或在備妥 cache 後設定 HF_HUB_OFFLINE=1。

下一步：[Slurm 兩步 smoke test](slurm.md)。Dry-run 不代表 GPU 記憶體、backward 或 checkpoint 已驗證。
